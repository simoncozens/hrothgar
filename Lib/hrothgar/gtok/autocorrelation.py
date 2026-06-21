"""Token autocorrelation: test sequential structure of G-Tok codes.

Trains a tiny 1-layer causal transformer to predict the next codebook index
from preceding indices.  High next-token accuracy indicates the tokenizer
produces sequentially structured codes — a prerequisite for successful
autoregressive generation.

Usage::

    python -m hrothgar.gtok.autocorrelation \\
        --gtok-model-path models/gtok_model.pth \\
        --dataset-path $GOOGLE_FONTS_REPO \\
        --epochs 5
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from hrothgar.googlefonts import Font, GoogleFonts
from hrothgar.gtok.model import GtokModel, load_model
from hrothgar.utils import torch_setup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
@dataclass
class AutocorrConfig:
    """Configuration for token autocorrelation probing."""

    gtok_model_path: str = "models/gtok_model.pth"
    dataset_path: str = os.environ.get("GOOGLE_FONTS_REPO", "")
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    max_samples: int = 50_000
    seed: int = 42
    # Per-position analysis: report accuracy for these position groups.
    per_position_buckets: tuple[int, ...] = (8,)


# ---------------------------------------------------------------------------
# 1-layer causal transformer for next-token prediction
# ---------------------------------------------------------------------------


class NextTokenProbe(nn.Module):
    """Single causal transformer layer for next-token prediction.

    Given a sequence of codebook indices ``s[0..i]``, predicts ``s[i+1]``.
    A tiny autoregressive model — if this can't predict the next token above
    chance, the full AR generator won't either.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(1024, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.head = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """Predict next-token logits for each position.

        Args:
            token_indices: ``(B, N)`` integer tensor of codebook indices.

        Returns:
            Logits of shape ``(B, N-1, vocab_size)`` — at each position
            ``i``, the logits predict the token at position ``i+1``.
        """
        B, N = token_indices.shape
        device = token_indices.device

        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(token_indices) + self.position_embedding(positions)
        x = self.dropout(x)

        # Causal mask: position i can attend to positions ≤ i.
        causal_mask = torch.triu(
            torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1
        )

        attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.layer_norm(x + self.dropout(attn_out))
        x = self.ffn_norm(x + self.ffn(x))

        # Predict token at position i+1 from context at position i.
        logits = self.head(x[:, :-1, :])  # (B, N-1, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Dataset: render glyphs → tokenize → collect index sequences
# ---------------------------------------------------------------------------


class TokenSequenceDataset(torch.utils.data.Dataset):
    """Dataset of (image, token_indices) pairs from G-Tok encoding."""

    def __init__(
        self,
        samples: List[Tuple[Font, int]],
        gtok: GtokModel,
        image_size: int,
        device: torch.device,
    ) -> None:
        self.samples = samples
        self.gtok = gtok
        self.image_size = image_size
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    @torch.no_grad()
    def __getitem__(self, idx: int) -> torch.Tensor:
        font, cp = self.samples[idx]
        image = torch.tensor(font.render(cp, size=self.image_size), dtype=torch.float32)
        image = image.unsqueeze(0).to(self.device)

        cnn_out = self.gtok.cnn_encoder(image)
        tokens = self.gtok.proj_patch(cnn_out).flatten(2).transpose(1, 2)
        vit_out = self.gtok.vit_encoder(tokens)
        pre_quant = self.gtok.vit_encoder_to_quantizer(vit_out)

        # Reshape and quantize to get discrete indices.
        _batch, _channels, _h, _w = cnn_out.shape
        pre_quant_4d = pre_quant.reshape(_batch, _h, _w, -1).permute(0, 3, 1, 2)
        _quantized, _loss, indices_info = self.gtok.quantizer(pre_quant_4d)
        token_indices = indices_info[2]  # (B*N,) flattened

        return token_indices.cpu()


def _collate_indices(batch: List[torch.Tensor]) -> torch.Tensor:
    """Stack variable-length token index sequences into a padded batch."""
    return torch.stack(batch, dim=0)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


class TokenAutocorrelation:
    """Measure sequential structure in G-Tok codebook index sequences.

    Trains a 1-layer causal transformer on next-token prediction and reports
    accuracy.  High accuracy means the codes are sequentially predictable,
    which is a prerequisite for autoregressive generation.
    """

    def __init__(self, config: AutocorrConfig) -> None:
        self.config = config
        self.device = torch_setup()

        gtok, gtok_config = load_model(Path(config.gtok_model_path), device=self.device)
        self.gtok = gtok
        self.gtok_config = gtok_config
        self.image_size: int = gtok_config.image_size
        self.gtok.eval()
        for param in self.gtok.parameters():
            param.requires_grad = False

        self.vocab_size = gtok_config.quantizer_codebook_size

        # Build dataset.
        self._build_dataset()

    def _build_dataset(self) -> None:
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

        gf = GoogleFonts(cfg.dataset_path)
        all_fonts = list(gf.fonts)
        rng.shuffle(all_fonts)

        # Use a small fixed character set for speed — we only care about
        # sequential structure, which should be character-agnostic.
        probe_chars = list(range(ord("A"), ord("Z") + 1))  # 26 uppercase

        samples: List[Tuple[Font, int]] = []
        for font in all_fonts:
            for cp in probe_chars:
                samples.append((font, cp))

        if cfg.max_samples > 0 and len(samples) > cfg.max_samples:
            indices = rng.choice(len(samples), cfg.max_samples, replace=False)
            samples = [samples[i] for i in indices]

        # 80/20 train/test split by font index (not family — we want to see
        # if sequential structure generalises to unseen fonts).
        split = int(len(samples) * 0.8)
        train_samples = samples[:split]
        test_samples = samples[split:]

        self.train_dataset = TokenSequenceDataset(
            train_samples, self.gtok, self.image_size, self.device
        )
        self.test_dataset = TokenSequenceDataset(
            test_samples, self.gtok, self.image_size, self.device
        )

        print(f"Vocabulary size:     {self.vocab_size}")
        print(f"Random-chance acc:   {1 / self.vocab_size:.4%}")
        print(f"Train samples:       {len(self.train_dataset)}")
        print(f"Test samples:        {len(self.test_dataset)}")

    def _make_loader(self, dataset: TokenSequenceDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=_collate_indices,
            num_workers=0,  # GPU encoding, can't fork
            pin_memory=False,
        )

    def run(self) -> float:
        """Train the probe and return best next-token accuracy."""
        cfg = self.config

        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        test_loader = self._make_loader(self.test_dataset, shuffle=False)

        probe = NextTokenProbe(
            vocab_size=self.vocab_size,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)

        optimizer = torch.optim.AdamW(probe.parameters(), lr=cfg.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0.0
        chance = 1.0 / self.vocab_size

        for epoch in range(cfg.epochs):
            probe.train()
            total_loss = 0.0
            total_tokens = 0

            for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                token_indices = batch.to(self.device)  # (B, N)
                B, N = token_indices.shape

                logits = probe(token_indices)  # (B, N-1, vocab_size)
                targets = token_indices[:, 1:]  # (B, N-1)

                loss = loss_fn(
                    logits.reshape(B * (N - 1), -1),
                    targets.reshape(B * (N - 1)),
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * B * (N - 1)
                total_tokens += B * (N - 1)

            # --- Validation ---
            probe.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    token_indices = batch.to(self.device)
                    B, N = token_indices.shape
                    logits = probe(token_indices)
                    targets = token_indices[:, 1:]
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == targets).sum().item()
                    total += B * (N - 1)

            acc = correct / total if total > 0 else 0.0
            best_acc = max(best_acc, acc)
            ratio = acc / chance if chance > 0 else float("inf")
            print(
                f"  Epoch {epoch + 1:2d}  "
                f"train loss: {total_loss / max(total_tokens, 1):.4f}  "
                f"test acc: {acc:.4f}  (best: {best_acc:.4f})  "
                f"×chance: {ratio:.1f}×"
            )

        print()
        print("=== Results ===")
        print(
            f"Next-token accuracy:  {best_acc:.4f}  ({best_acc / chance:.1f}× random)"
        )

        # --- Per-position accuracy analysis ---
        self._report_per_position_accuracy(probe, test_loader)

        if best_acc > chance * 100:
            print("✓ Strong sequential structure — AR generator should succeed")
        elif best_acc > chance * 10:
            print("⚠  Moderate sequential structure — AR generator may struggle")
        else:
            print("✗ Weak sequential structure — AR generator will likely fail")

        return best_acc

    def _report_per_position_accuracy(
        self,
        probe: NextTokenProbe,
        test_loader: DataLoader,
    ) -> None:
        """Log next-token accuracy broken down by position in the sequence.

        The token grid is flattened in raster-scan order.  Positions are
        bucketed by row (every ``grid_width`` tokens) to reveal whether
        within-row transitions are more predictable than row boundaries.
        """
        grid_w = self.gtok.token_grid_width
        bucket_size = self.config.per_position_buckets[0]

        probe.eval()
        correct_by_pos: dict[int, int] = {}
        total_by_pos: dict[int, int] = {}

        with torch.no_grad():
            for batch in test_loader:
                token_indices = batch.to(self.device)
                B, N = token_indices.shape
                logits = probe(token_indices)  # (B, N-1, vocab_size)
                targets = token_indices[:, 1:]  # (B, N-1)
                preds = torch.argmax(logits, dim=-1)  # (B, N-1)

                for pos in range(N - 1):
                    bucket = (pos // bucket_size) * bucket_size
                    correct_by_pos.setdefault(bucket, 0)
                    total_by_pos.setdefault(bucket, 0)
                    correct_by_pos[bucket] += (
                        (preds[:, pos] == targets[:, pos]).sum().item()
                    )
                    total_by_pos[bucket] += B

        print()
        print(f"=== Per-Position Accuracy  (bucket size = {bucket_size}) ===")
        print(f"{'Bucket':>8s}  {'Positions':>12s}  {'Accuracy':>10s}  {'×Chance':>8s}")
        print("-" * 48)
        chance = 1.0 / self.vocab_size
        for bucket in sorted(correct_by_pos.keys()):
            acc = correct_by_pos[bucket] / total_by_pos[bucket]
            start = bucket
            end = min(bucket + bucket_size - 1, self.gtok.sequence_length - 2)
            # Flag row boundaries: positions where index % grid_w == grid_w - 1
            marker = " ← row start" if start % grid_w == 0 else ""
            print(
                f"{bucket:>4d}-{end:<4d}  "
                f"{total_by_pos[bucket]:>8d}       "
                f"{acc:>8.2%}     "
                f"{acc / chance:>6.1f}×"
                f"{marker}"
            )

        # Summary: within-row vs cross-row-boundary accuracy.
        within_correct = 0
        within_total = 0
        cross_correct = 0
        cross_total = 0
        with torch.no_grad():
            for batch in test_loader:
                token_indices = batch.to(self.device)
                B, N = token_indices.shape
                logits = probe(token_indices)
                targets = token_indices[:, 1:]
                preds = torch.argmax(logits, dim=-1)

                for pos in range(N - 1):
                    is_cross = (
                        pos + 1
                    ) % grid_w == 0  # predicting first token of next row
                    if is_cross:
                        cross_correct += (preds[:, pos] == targets[:, pos]).sum().item()
                        cross_total += B
                    else:
                        within_correct += (
                            (preds[:, pos] == targets[:, pos]).sum().item()
                        )
                        within_total += B

        within_acc = within_correct / max(within_total, 1)
        cross_acc = cross_correct / max(cross_total, 1)
        print()
        print(
            f"Within-row accuracy:     {within_acc:.2%}  ({within_acc / chance:.1f}×)"
        )
        print(f"Cross-row-boundary acc:  {cross_acc:.2%}  ({cross_acc / chance:.1f}×)")
        if within_acc > 0 and cross_acc > 0:
            ratio = within_acc / cross_acc
            print(f"Within/cross ratio:      {ratio:.1f}×")
            if ratio > 3:
                print(
                    "⚠  Strong positional dependence: row boundaries are much harder.\n"
                    "   The tokenizer may have good local structure but weak global structure.\n"
                    "   Visual conditioning in the full AR model should supply the global signal."
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Token autocorrelation probe for G-Tok"
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
        "--epochs", type=int, default=5, help="Training epochs for the probe"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="AdamW learning rate"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dim of the probe transformer",
    )
    parser.add_argument(
        "--max-samples", type=int, default=50_000, help="Cap on total samples"
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

    config = AutocorrConfig(
        gtok_model_path=args.gtok_model_path,
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    probe = TokenAutocorrelation(config)
    probe.run()


if __name__ == "__main__":
    main()
