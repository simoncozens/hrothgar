"""Oracle AR probe: test G-Tok sequential structure within a single font.

Trains a small GPT (no conditioning, no image features) to autoregressively
predict token sequences from all glyphs of *one* font.  If the model achieves
high training accuracy (>80%), the tokenizer has strong per-font sequential
structure, and the full AR model's struggles are a conditioning/optimisation
problem.  If training accuracy stays low (<30%), the tokenizer is the
bottleneck — its codes lack the sequential regularity that autoregressive
models require.

This is the single most decisive diagnostic for whether the G-Tok tokenizer is
fit for purpose as an AR generation target.

Usage::

    python -m hrothgar.gtok.oracle_ar \\
        --gtok-model-path models/gtok_model.pth \\
        --dataset-path $GOOGLE_FONTS_REPO \\
        --font-index 0 \\
        --steps 20000
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from hrothgar.googlefonts import Font, GoogleFonts
from hrothgar.gtok.model import GtokModel, load_model
from hrothgar.upstream.gpt import GPTModelArgs
from hrothgar.upstream.gpt import Transformer as GPTTransformer
from hrothgar.utils import torch_setup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OracleARConfig:
    """Configuration for the oracle AR probe."""

    gtok_model_path: str = "models/gtok_model.pth"
    dataset_path: str = os.environ.get("GOOGLE_FONTS_REPO", "")
    font_index: int = 0
    steps: int = 20_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    seed: int = 42
    # GPT sizing — default GPT-90M.
    gpt_dim: int = 768
    gpt_layers: int = 12
    gpt_heads: int = 12

    # Derived — set after G-Tok is loaded.
    vocab_size: int = field(default=0, init=False)
    image_size: int = field(default=0, init=False)
    sequence_length: int = field(default=0, init=False)


# ---------------------------------------------------------------------------
# Dataset: all glyphs from one font
# ---------------------------------------------------------------------------


class SingleFontTokenDataset(torch.utils.data.Dataset):
    """All token sequences from one font — the "oracle" dataset."""

    def __init__(
        self,
        font: Font,
        gtok: GtokModel,
        image_size: int,
        device: torch.device,
    ) -> None:
        self.gtok = gtok
        self.image_size = image_size
        self.device = device

        # Collect all renderable codepoints from this font.
        self.codepoints: list[int] = sorted(
            cp for cp in font.codepoints if self._is_renderable(font, cp)
        )

        # Pre-tokenize everything.
        self.token_sequences: list[torch.Tensor] = []
        for cp in tqdm.tqdm(self.codepoints, desc="Tokenizing font glyphs"):
            tokens = self._tokenize(font, cp)
            if tokens is not None:
                self.token_sequences.append(tokens.cpu())

        if not self.token_sequences:
            raise RuntimeError(
                f"Font '{font.family}' has no renderable glyphs — "
                "choose a different font."
            )

        print(f"Font:   {font.family}")
        print(f"Glyphs: {len(self.token_sequences)}")
        print(f"Tokens per glyph: {self.token_sequences[0].shape[0]}")

    @staticmethod
    def _is_renderable(font: Font, cp: int) -> bool:
        """Quick check that the font can render this codepoint."""
        import uharfbuzz as hb

        hb_font = hb.Font(font.hb_face)  # type: ignore
        gid = hb_font.get_nominal_glyph(cp)
        extents = hb_font.get_glyph_extents(gid)
        if extents is None:
            return False
        return not all(x == 0 for x in extents)

    @torch.no_grad()
    def _tokenize(self, font: Font, cp: int) -> torch.Tensor | None:
        """Render + tokenize one glyph."""
        image = torch.tensor(font.render(cp, size=self.image_size), dtype=torch.float32)
        # Skip blank renderings.
        if float(image.max()) == float(image.min()):
            return None

        image = image.unsqueeze(0).to(self.device)
        cnn_out = self.gtok.cnn_encoder(image)
        tokens = self.gtok.proj_patch(cnn_out).flatten(2).transpose(1, 2)
        vit_out = self.gtok.vit_encoder(tokens)
        pre_quant = self.gtok.vit_encoder_to_quantizer(vit_out)

        _batch, _channels, _h, _w = cnn_out.shape
        pre_quant_4d = pre_quant.reshape(_batch, _h, _w, -1).permute(0, 3, 1, 2)
        _quantized, _loss, indices_info = self.gtok.quantizer(pre_quant_4d)
        return indices_info[2]  # (N,) flattened indices

    def __len__(self) -> int:
        return len(self.token_sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.token_sequences[idx]


def _collate_oracle(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


# ---------------------------------------------------------------------------
# Oracle runner
# ---------------------------------------------------------------------------


class OracleARProbe:
    """Train a conditionless GPT on one font's token sequences."""

    def __init__(self, config: OracleARConfig) -> None:
        self.config = config
        self.device = torch_setup()

        gtok, gtok_config = load_model(Path(config.gtok_model_path), device=self.device)
        self.gtok = gtok
        self.gtok.eval()
        for param in self.gtok.parameters():
            param.requires_grad = False

        config.vocab_size = gtok_config.quantizer_codebook_size
        config.image_size = gtok_config.image_size
        config.sequence_length = gtok_config.image_size // gtok.downsampling_factor
        config.sequence_length = config.sequence_length * config.sequence_length

        # Load one font.
        gf = GoogleFonts(config.dataset_path)
        all_fonts = sorted(gf.fonts, key=lambda f: f.family)
        if config.font_index >= len(all_fonts):
            raise ValueError(
                f"font_index {config.font_index} out of range "
                f"(max {len(all_fonts) - 1})"
            )
        font = all_fonts[config.font_index]
        self._font_name = font.family

        self.dataset = SingleFontTokenDataset(
            font, self.gtok, config.image_size, self.device
        )
        config.sequence_length = self.dataset.token_sequences[0].shape[0]

        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=_collate_oracle,
        )

        # Build a small GPT with NO conditioning path.
        gpt_config = GPTModelArgs(
            vocab_size=config.vocab_size,
            dim=config.gpt_dim,
            n_layer=config.gpt_layers,
            n_head=config.gpt_heads,
            img_feature_channel=0,  # No conditioning!
            img_feature_code_len=0,
            target_token_len=config.sequence_length,
            token_dropout_p=0.1,
            attn_dropout_p=0.0,
            resid_dropout_p=0.1,
            ffn_dropout_p=0.1,
        )
        self.model = GPTTransformer(gpt_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95)
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"GPT parameters:     {total_params:,}")
        print(f"Vocabulary size:    {config.vocab_size}")
        print(f"Random-chance acc:  {1 / config.vocab_size:.4%}")
        print(f"Training steps:     {config.steps}")

    def run(self) -> float:
        """Train the oracle GPT and report final token accuracy."""
        cfg = self.config
        loss_fn = nn.CrossEntropyLoss()
        data_iter = iter(self.loader)

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        log_every = 500
        best_acc = 0.0
        chance = 1.0 / cfg.vocab_size

        scaler = torch.amp.GradScaler("cuda", enabled=False)

        pbar = tqdm.tqdm(range(1, cfg.steps + 1), desc="Oracle AR training")
        for step in pbar:
            self.model.train()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)

            token_indices = batch.to(self.device)  # (B, N)

            # PrefixLM with empty conditioning.
            idx = token_indices[:, :-1]
            targets = token_indices

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Pass empty conditioning map.
                dummy_conditioning = torch.empty(
                    token_indices.shape[0], 0, 0, device=self.device
                )
                logits, _loss = self.model(
                    idx=idx,
                    imgs_feature_map=dummy_conditioning,
                    targets=targets,
                )

            loss = loss_fn(
                logits.reshape(-1, cfg.vocab_size),
                targets.reshape(-1),
            )

            self.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            preds = torch.argmax(logits, dim=-1)
            running_correct += (preds == targets).sum().item()
            running_total += targets.numel()
            running_loss += loss.item()

            if step % log_every == 0:
                acc = running_correct / max(running_total, 1)
                avg_loss = running_loss / log_every
                best_acc = max(best_acc, acc)
                pbar.set_postfix(
                    loss=f"{avg_loss:.3f}",
                    acc=f"{acc:.3f}",
                    best=f"{best_acc:.3f}",
                )
                running_loss = 0.0
                running_correct = 0
                running_total = 0

        print()
        print("=" * 56)
        print("=== Oracle AR Results ===")
        print(f"Font:              {self._font_name} (index {cfg.font_index})")
        print(f"Glyphs:            {len(self.dataset)}")
        print(f"Best token acc:    {best_acc:.4f}  ({best_acc / chance:.0f}× chance)")
        print(f"Random chance:     {chance:.4%}")
        print()

        if best_acc > 0.80:
            print("✓✓✓ TOKENIZER IS FINE — strong per-font sequential structure.")
            print(
                "    The full AR model's struggles are a conditioning/optimisation problem."
            )
        elif best_acc > 0.50:
            print("✓ Tokenizer is adequate — moderate sequential structure.")
            print("  Conditioning and exposure bias are likely the main bottlenecks.")
        elif best_acc > 0.20:
            print("⚠  Tokenizer has marginal sequential structure.")
            print(
                "   Consider: more training, larger codebook, or different VQ-VAE config."
            )
        else:
            print("✗✗✗ TOKENIZER IS THE BOTTLENECK — weak sequential structure.")
            print("    No AR model can succeed without stronger token codes.")
            print("    Consider: retraining the tokenizer with different objectives or")
            print("    architectural changes (e.g., spatial autoregressive ordering).")

        return best_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Oracle AR probe: test G-Tok sequential structure within one font"
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok_model.pth",
        help="Path to trained G-Tok weights (.pth)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO", ""),
        help="Path to the Google Fonts repository",
    )
    parser.add_argument(
        "--font-index",
        type=int,
        default=0,
        help="Index into sorted font list (0 = first alphabetically)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20_000,
        help="Training steps",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.dataset_path:
        parser.error(
            "--dataset-path is required (or set GOOGLE_FONTS_REPO environment variable)"
        )

    config = OracleARConfig(
        gtok_model_path=args.gtok_model_path,
        dataset_path=args.dataset_path,
        font_index=args.font_index,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    probe = OracleARProbe(config)
    probe.run()


if __name__ == "__main__":
    main()
