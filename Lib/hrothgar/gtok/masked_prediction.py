"""Masked-prediction probe: test G-Tok bidirectional predictability for MaskGIT.

Trains a small bidirectional transformer (no conditioning) to predict randomly
masked tokens from the unmasked context, over the glyphs of a single font.  This
is the MaskGIT training objective in miniature, and it measures the property the
AR probes (``oracle_ar``, ``autocorrelation``) do not: **bidirectional masked
predictability** — whether a partial token grid constrains the rest.

- High masked-token accuracy (>80%) → the codes are bidirectionally predictable;
  the tokenizer is fit for MaskGIT, and any generator failure is a
  conditioning / cold-start problem.
- Low accuracy (<20%) → the tokenizer is the bottleneck.

This is the MaskGIT analogue of ``oracle_ar`` (which measures *causal*
sequential structure instead).

Usage::

    python -m hrothgar.gtok.masked_prediction \\
        --gtok-model-path models/gtok.pth \\
        --dataset-path $GOOGLE_FONTS_REPO \\
        --steps 20000
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from hrothgar.glyph_rendering import crop_to_ink
from hrothgar.googlefonts import Font, GoogleFonts
from hrothgar.gtok.model import GtokModel, load_model
from hrothgar.utils import torch_setup


# ---------------------------------------------------------------------------
# Minimal bidirectional transformer (no conditioning path)
# ---------------------------------------------------------------------------


class BidirectionalMaskedDecoder(nn.Module):
    """A minimal bidirectional transformer for masked-token prediction.

    Unlike the causal decoder in ``oracle_ar``, this uses full (non-causal)
    self-attention, so every token can attend to every other token — matching
    MaskGIT's masked-prediction objective.  A single extra ``[MASK]`` embedding
    (index ``vocab_size``) marks the positions to predict.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size  # extra embedding slot for [MASK]

        self.token_embedding = nn.Embedding(vocab_size + 1, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        nn.init.constant_(self.head.weight, 0)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, masked_indices: torch.Tensor) -> torch.Tensor:
        """Return logits at every position given a partially-masked grid.

        Args:
            masked_indices: ``(B, N)`` integers, with ``mask_token_id`` at the
                positions to predict.

        Returns:
            ``(B, N, vocab_size)`` logits.
        """
        B, N = masked_indices.shape
        device = masked_indices.device
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        h = self.token_embedding(masked_indices) + self.position_embedding(positions)
        h = self.dropout(h)
        h = self.transformer(h)  # no causal mask -> bidirectional
        h = self.ln_f(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MaskedPredictionConfig:
    """Configuration for the masked-prediction probe."""

    gtok_model_path: str = "models/gtok.pth"
    dataset_path: str = os.environ.get("GOOGLE_FONTS_REPO", "")
    font_index: int = 0
    steps: int = 20_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    seed: int = 42
    # Probe transformer sizing.
    dim: int = 384
    layers: int = 6
    heads: int = 8

    # Derived — set after G-Tok is loaded.
    vocab_size: int = field(default=0, init=False)
    image_size: int = field(default=0, init=False)
    sequence_length: int = field(default=0, init=False)


# ---------------------------------------------------------------------------
# Dataset: all glyphs from one font
# ---------------------------------------------------------------------------


class SingleFontTokenDataset(torch.utils.data.Dataset):
    """All token sequences from one font."""

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

        self.codepoints: list[int] = sorted(
            cp for cp in font.codepoints if self._is_renderable(font, cp)
        )

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
        import uharfbuzz as hb

        hb_font = hb.Font(font.hb_face)  # type: ignore
        gid = hb_font.get_nominal_glyph(cp)
        extents = hb_font.get_glyph_extents(gid)
        if extents is None:
            return False
        return not all(x == 0 for x in extents)

    @torch.no_grad()
    def _tokenize(self, font: Font, cp: int) -> torch.Tensor | None:
        image = torch.tensor(font.render(cp, size=self.image_size), dtype=torch.float32)
        image = crop_to_ink(image, self.image_size)
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
        return indices_info[2]  # (N,) flattened codebook indices

    def __len__(self) -> int:
        return len(self.token_sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.token_sequences[idx]


def _collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------


class MaskedPredictionProbe:
    """Train a conditionless bidirectional transformer to predict masked tokens."""

    def __init__(self, config: MaskedPredictionConfig) -> None:
        self.config = config
        self.device = torch_setup()

        gtok, gtok_config = load_model(Path(config.gtok_model_path), device=self.device)
        self.gtok = gtok
        self.gtok.eval()
        for param in self.gtok.parameters():
            param.requires_grad = False

        config.vocab_size = gtok_config.quantizer_codebook_size
        config.image_size = gtok_config.image_size
        config.sequence_length = (
            gtok_config.image_size // gtok.downsampling_factor
        ) ** 2

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
            collate_fn=_collate,
        )

        self.model = BidirectionalMaskedDecoder(
            vocab_size=config.vocab_size,
            dim=config.dim,
            n_layers=config.layers,
            n_heads=config.heads,
            max_seq_len=config.sequence_length,
            dropout=0.1,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95)
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Probe parameters:   {total_params:,}")
        print(f"Vocabulary size:    {config.vocab_size}")
        print(f"Random-chance acc:  {1 / config.vocab_size:.4%}")
        print(f"Training steps:     {config.steps}")

    def _apply_mask(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly mask a cosine-sampled fraction of positions.

        Returns ``(masked_input, mask_bool)`` where ``masked_input`` has the
        ``[MASK]`` token at masked positions and ``mask_bool`` marks them.
        """
        B, N = tokens.shape
        device = tokens.device
        mask_token_id = self.model.mask_token_id

        # Same distribution as MaskGIT training (cosine / arcsine, mean ~64%).
        r = 1.0 - torch.rand(1, device=device).item()
        mask_ratio = math.cos(math.pi / 2 * r)
        num_mask = max(1, int(N * mask_ratio))

        rand = torch.rand(B, N, device=device)
        mask = rand.argsort(dim=-1)[:, :num_mask]  # (B, num_mask)
        mask_bool = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask_bool.scatter_(1, mask, True)

        masked_input = tokens.clone()
        masked_input[mask_bool] = mask_token_id
        return masked_input, mask_bool

    def run(self) -> float:
        """Train the probe and report final masked-token accuracy."""
        cfg = self.config
        loss_fn = nn.CrossEntropyLoss()
        data_iter = iter(self.loader)

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        log_every = 500
        best_acc = 0.0
        chance = 1.0 / cfg.vocab_size

        pbar = tqdm.tqdm(range(1, cfg.steps + 1), desc="Masked-prediction training")
        for step in pbar:
            self.model.train()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)

            tokens = batch.to(self.device)  # (B, N)
            masked_input, mask_bool = self._apply_mask(tokens)

            logits = self.model(masked_input)  # (B, N, vocab_size)
            loss = loss_fn(logits[mask_bool], tokens[mask_bool])

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            running_correct += (preds[mask_bool] == tokens[mask_bool]).sum().item()
            running_total += int(mask_bool.sum().item())
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
        print("=== Masked-Prediction Results ===")
        print(f"Font:              {self._font_name} (index {cfg.font_index})")
        print(f"Glyphs:            {len(self.dataset)}")
        print(f"Best token acc:    {best_acc:.4f}  ({best_acc / chance:.0f}× chance)")
        print(f"Random chance:     {chance:.4%}")
        print()

        if best_acc > 0.80:
            print("✓✓✓ TOKENIZER IS FINE — strong bidirectional masked predictability.")
            print(
                "    The generator's failures are a conditioning / cold-start problem."
            )
        elif best_acc > 0.50:
            print("✓ Tokenizer is adequate — moderate bidirectional predictability.")
            print("  Conditioning and cold-start are likely the main bottlenecks.")
        elif best_acc > 0.20:
            print("⚠  Tokenizer has marginal bidirectional predictability.")
            print("   Consider: more training, lower code dim, or a different VQ config.")
        else:
            print("✗✗✗ TOKENIZER IS THE BOTTLENECK — weak bidirectional predictability.")
            print("    No MaskGIT model can succeed without stronger token codes.")
            print("    Consider: disentangling content from position (lower code dim,")
            print("    removing position embeddings from the encoder, or a pure CNN VQ).")

        return best_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Masked-prediction probe: test G-Tok bidirectional predictability"
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok.pth",
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
    parser.add_argument("--steps", type=int, default=20_000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="AdamW learning rate"
    )
    parser.add_argument("--dim", type=int, default=384, help="Probe hidden dim")
    parser.add_argument("--layers", type=int, default=6, help="Probe transformer layers")
    parser.add_argument("--heads", type=int, default=8, help="Probe attention heads")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.dataset_path:
        parser.error(
            "--dataset-path is required (or set GOOGLE_FONTS_REPO environment variable)"
        )

    torch.manual_seed(args.seed)

    config = MaskedPredictionConfig(
        gtok_model_path=args.gtok_model_path,
        dataset_path=args.dataset_path,
        font_index=args.font_index,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
    )

    probe = MaskedPredictionProbe(config)
    probe.run()


if __name__ == "__main__":
    main()
