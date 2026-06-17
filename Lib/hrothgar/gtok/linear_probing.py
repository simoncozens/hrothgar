"""Linear probing for G-Tok tokenizer quality assessment.

Evaluates whether the frozen G-Tok encoder produces representations that
linearly separate character identity (a-zA-Z) and font family.  High accuracy
on both tasks indicates a well-organised latent space suitable for downstream
autoregressive modelling.

Usage as a module::

    from hrothgar.gtok.linear_probing import GtokLinearProbe, ProbeConfig
    config = ProbeConfig(...)
    probe = GtokLinearProbe(config)
    char_acc, font_acc = probe.run()

CLI::

    python -m hrothgar.gtok.linear_probing \\
        --gtok-model-path models/gtok_model.pth \\
        --dataset-path $GOOGLE_FONTS_REPO \\
        --epochs 10
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from hrothgar.googlefonts import GoogleFonts
from hrothgar.gtok.model import GtokConfig, GtokModel, load_model
from hrothgar.utils import torch_setup

# ---------------------------------------------------------------------------
# Character probe alphabet: a-z and A-Z, mapped to 0..51
# ---------------------------------------------------------------------------

_PROBE_CHARS = list(range(ord("A"), ord("Z") + 1)) + list(range(ord("a"), ord("z") + 1))
_CHAR_TO_INDEX: Dict[int, int] = {cp: i for i, cp in enumerate(_PROBE_CHARS)}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
@dataclass
class ProbeConfig:
    """Configuration for G-Tok linear probing.

    Attributes:
        gtok_model_path: Path to a trained G-Tok ``.pth`` weights file.
            A ``.conf.json`` sidecar must exist alongside it.
        dataset_path: Path to the Google Fonts repository.
        epochs: Number of training epochs for each probe head.
        batch_size: Batch size for training and evaluation.
        learning_rate: Adam learning rate.
        weight_decay: L2 regularisation strength.
        probe_font_count: Maximum number of font-family classes (takes the
            most frequent *probe_font_count* families).  Set to 0 for no limit.
        probe_font_min_samples: Only include font families with at least this
            many samples in the *train* split.
        train_frac: Fraction of probe-eligible data used for the probe
            training set (rest is held out for probe validation).
        max_samples: Cap on total samples across all probe classes.  Set to 0
            for no limit.
        seed: RNG seed for reproducibility.
    """

    gtok_model_path: str = "models/gtok_model.pth"
    dataset_path: str = os.environ.get("GOOGLE_FONTS_REPO", "")
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    probe_font_count: int = 100
    probe_font_min_samples: int = 20
    train_frac: float = 0.8
    max_samples: int = 50_000
    seed: int = 42


# ---------------------------------------------------------------------------
# Probe model: mean-pool frozen features → linear classifier
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Single linear layer on top of mean-pooled frozen G-Tok features.

    The probe operates on the *pre-quantization* projected features from the
    G-Tok encoder (the output of ``vit_encoder_to_quantizer``).  These are
    the same continuous representations the AR generator's content encoder is
    designed to condition on, so probing them directly answers whether the
    latent space separates content and style.

    The full token sequence is flattened (following the paper: "Features are
    extracted and flattened from the frozen tokenizer encoder"), giving
    ``N * code_dim`` features.
    """

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Flatten token sequence, then classify.

        Args:
            features: ``(B, N, D)`` frozen G-Tok pre-quantization features.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        return self.classifier(features.reshape(features.shape[0], -1))


# ---------------------------------------------------------------------------
# Feature extractor using a frozen G-Tok encoder
# ---------------------------------------------------------------------------


class FrozenGtokFeatureExtractor:
    """Extract pre-quantization features from a frozen G-Tok model."""

    def __init__(self, model: GtokModel, config: GtokConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.downsampling_factor = 2 ** (len(config.cnn_channel_multipliers or []) - 1)
        self.grid_h = config.image_size // self.downsampling_factor
        self.grid_w = config.image_size // self.downsampling_factor
        print("Downsampling factor:", self.downsampling_factor)
        print("Grid size:", self.grid_h, "x", self.grid_w)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Return pre-quantization features for a batch of images.

        Args:
            images: ``(B, 3, H, W)`` float tensor in [0, 1].

        Returns:
            Features of shape ``(B, N, code_dim)`` where N is the flattened
            token grid size.
        """
        cnn_out = self.model.cnn_encoder(images)
        # Patch projection (Conv2d) then flatten — matches upstream GAR-Font layout.
        tokens = (
            self.model.proj_patch(cnn_out).flatten(2).transpose(1, 2)
        )  # (B, N, vit_hidden_dim)
        # Upstream ViTEncoder: no class token, same-dim in/out.
        vit_out = self.model.vit_encoder(tokens)  # (B, N, vit_hidden_dim)
        features = self.model.vit_encoder_to_quantizer(vit_out)  # (B, N, code_dim)
        return features


# ---------------------------------------------------------------------------
# Dataset builder for probing
# ---------------------------------------------------------------------------


class ProbeDataset(torch.utils.data.Dataset):
    """Dataset that renders probe-alphabet glyphs from a list of samples.

    Each sample is a ``(font, codepoint, char_label, family_label)`` tuple.
    """

    def __init__(self, samples: List[Tuple], image_size: int) -> None:
        self.samples = samples
        self._image_size = image_size

    @classmethod
    def from_font_list(
        cls,
        fonts: List,
        char_codepoints: List[int],
        image_size: int,
        char_to_label: Dict[int, int],
        family_to_label: Dict[str, int],
    ) -> ProbeDataset:
        """Build a dataset by enumerating all font × character combinations."""
        samples: List[Tuple] = []
        for font in fonts:
            family_label = family_to_label.get(font.family)
            if family_label is None:
                continue
            for cp in char_codepoints:
                char_label = char_to_label.get(cp)
                if char_label is None:
                    continue
                samples.append((font, cp, char_label, family_label))
        return cls(samples, image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        font, cp, char_label, family_label = self.samples[idx]
        return {
            "font": font,
            "char": cp,
            "char_label": char_label,
            "family_label": family_label,
        }


def _collate_probe_batch(batch, image_size: int) -> Dict[str, torch.Tensor]:
    """Render a batch of probe samples into tensors."""
    images = torch.stack(
        [
            torch.tensor(item["font"].render(item["char"], size=image_size))
            for item in batch
        ]
    )
    char_labels = torch.tensor([item["char_label"] for item in batch], dtype=torch.long)
    family_labels = torch.tensor(
        [item["family_label"] for item in batch], dtype=torch.long
    )
    return {
        "images": images,
        "char_label": char_labels,
        "family_label": family_labels,
    }


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


class GtokLinearProbe:
    """Run linear probing on a frozen G-Tok encoder.

    Builds two independent linear probes — one for character identity
    (a-zA-Z) and one for font family — trains each for a fixed number of
    epochs, and reports validation accuracy.
    """

    def __init__(self, config: ProbeConfig) -> None:
        self.config = config
        self.device = torch_setup()

        # 1. Load G-Tok.
        gtok, gtok_config = load_model(Path(config.gtok_model_path), device=self.device)
        self.gtok = gtok
        self.gtok_config = gtok_config
        # Render at the model's native resolution (read from sidecar).
        self.image_size: int = gtok_config.image_size
        self.extractor = FrozenGtokFeatureExtractor(gtok, gtok_config, self.device)
        # Flattened token sequence: N tokens × code_dim (matches the paper).
        self.feature_dim = (
            gtok_config.quantizer_code_dim
            * self.extractor.grid_h
            * self.extractor.grid_w
        )

        # 2. Build probe datasets.
        self._build_datasets()

    def _build_datasets(self) -> None:
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

        # Load all fonts and group them by family.
        gf = GoogleFonts(cfg.dataset_path)
        # Use the same font filter as GTok
        gf.fonts = [font for font in gf.fonts if font.display_score() < 60.0]

        family_to_fonts: Dict[str, List] = {}
        for font in gf.fonts:
            family_to_fonts.setdefault(font.family, []).append(font)

        # Eligible families: those with at least probe_font_min_samples font
        # *files* (each renders 52 characters, so effective sample count is
        # len(fonts) × 52 — well above 20 for even a single font file).
        eligible_families = [
            fam
            for fam, fonts in family_to_fonts.items()
            if len(fonts) >= cfg.probe_font_min_samples
        ]
        eligible_families.sort()

        if cfg.probe_font_count > 0:
            eligible_families = eligible_families[: cfg.probe_font_count]

        self.family_to_label = {fam: i for i, fam in enumerate(eligible_families)}
        self.num_font_classes = len(self.family_to_label)
        self.num_char_classes = len(_CHAR_TO_INDEX)

        # Build per-family train/test sample lists by splitting font *files*
        # within each family.  This avoids the empty test-set problem caused
        # by a family-level train/test split.
        train_samples: List[Tuple] = []
        test_samples: List[Tuple] = []

        for family in eligible_families:
            fonts = family_to_fonts[family]
            rng.shuffle(fonts)

            family_label = self.family_to_label[family]
            n_fonts = len(fonts)

            if n_fonts == 1:
                # Single-font family: probabilistically assign to train or test.
                if rng.random() < cfg.train_frac:
                    train_fonts_for_family = fonts
                    test_fonts_for_family = []
                else:
                    train_fonts_for_family = []
                    test_fonts_for_family = fonts
            else:
                # Multi-font family: split with at least one in each.
                n_train = max(1, int(round(n_fonts * cfg.train_frac)))
                n_train = min(n_train, n_fonts - 1)  # leave ≥1 for test
                train_fonts_for_family = fonts[:n_train]
                test_fonts_for_family = fonts[n_train:]

            for font in train_fonts_for_family:
                for cp in _PROBE_CHARS:
                    char_label = _CHAR_TO_INDEX[cp]
                    train_samples.append((font, cp, char_label, family_label))

            for font in test_fonts_for_family:
                for cp in _PROBE_CHARS:
                    char_label = _CHAR_TO_INDEX[cp]
                    test_samples.append((font, cp, char_label, family_label))

        # Cap total training samples if requested.
        if cfg.max_samples > 0 and len(train_samples) > cfg.max_samples:
            rng.shuffle(train_samples)
            train_samples = train_samples[: cfg.max_samples]

        self.train_dataset = ProbeDataset(train_samples, self.image_size)
        self.test_dataset = ProbeDataset(test_samples, self.image_size)

        print(f"Probe character classes: {self.num_char_classes}  (a-zA-Z)")
        print(f"Probe font-family classes: {self.num_font_classes}")
        print(
            f"Train fonts:               {len(train_samples) // self.num_char_classes}"
        )
        print(
            f"Test fonts:                {len(test_samples) // self.num_char_classes}"
        )
        print(f"Train samples:             {len(self.train_dataset)}")
        print(f"Test samples:              {len(self.test_dataset)}")

    def _make_loader(self, dataset: ProbeDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=lambda batch: _collate_probe_batch(batch, self.image_size),
            num_workers=4,
            pin_memory=True,
        )

    def _train_one_probe(
        self,
        probe: LinearProbe,
        train_loader: DataLoader,
        test_loader: DataLoader,
        label_key: str,
        *,
        desc: str,
    ) -> float:
        """Train a single linear probe and return best test accuracy."""
        cfg = self.config
        probe.to(self.device)
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0.0

        for epoch in range(cfg.epochs):
            probe.train()
            total_loss = 0.0
            total_samples = 0

            for batch in tqdm.tqdm(train_loader, desc=f"{desc} epoch {epoch+1}"):
                images = batch["images"].to(self.device)
                labels = batch[label_key].to(self.device)

                features = self.extractor.extract(images)
                logits = probe(features)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.shape[0]
                total_samples += images.shape[0]

            # --- Validation ---
            probe.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    images = batch["images"].to(self.device)
                    labels = batch[label_key].to(self.device)
                    features = self.extractor.extract(images)
                    logits = probe(features)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.shape[0]

            acc = correct / total if total > 0 else 0.0
            best_acc = max(best_acc, acc)
            print(
                f"  {desc}  epoch {epoch+1:2d}  "
                f"train loss: {total_loss / max(total_samples, 1):.4f}  "
                f"test acc: {acc:.4f}  (best: {best_acc:.4f})"
            )

        return best_acc

    def run(self) -> Tuple[float, float]:
        """Run both probes and return (char_accuracy, font_accuracy)."""
        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        test_loader = self._make_loader(self.test_dataset, shuffle=False)

        print("\n=== Character probe (a-zA-Z) ===")
        char_probe = LinearProbe(self.feature_dim, self.num_char_classes)
        char_acc = self._train_one_probe(
            char_probe,
            train_loader,
            test_loader,
            label_key="char_label",
            desc="Char",
        )

        print(f"\n=== Font-family probe ({self.num_font_classes} families) ===")
        font_probe = LinearProbe(self.feature_dim, self.num_font_classes)
        font_acc = self._train_one_probe(
            font_probe,
            train_loader,
            test_loader,
            label_key="family_label",
            desc="Font",
        )

        print(f"\n=== Results ===")
        print(
            f"Character accuracy:  {char_acc:.4f}  {'✓' if char_acc >= 0.85 else '✗ (target ≥ 0.85)'}"
        )
        print(
            f"Font-family accuracy: {font_acc:.4f}  {'✓' if font_acc >= 0.70 else '✗ (target ≥ 0.70)'}"
        )

        return char_acc, font_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Linear probing for G-Tok tokenizer quality"
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok_model.pth",
        help="Path to trained G-Tok weights (.pth); .conf.json must exist beside it",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO", ""),
        help="Path to the Google Fonts repository",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs per probe head",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularisation strength",
    )
    parser.add_argument(
        "--probe-font-count",
        type=int,
        default=100,
        help="Maximum number of font-family classes",
    )
    parser.add_argument(
        "--probe-font-min-samples",
        type=int,
        default=20,
        help="Minimum samples per font family in the train split",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50_000,
        help="Cap on total training samples (0 for no limit)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for dataset shuffling",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.dataset_path:
        parser.error(
            "--dataset-path is required (or set GOOGLE_FONTS_REPO environment variable)"
        )

    config = ProbeConfig(
        gtok_model_path=args.gtok_model_path,
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        probe_font_count=args.probe_font_count,
        probe_font_min_samples=args.probe_font_min_samples,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    probe = GtokLinearProbe(config)
    probe.run()


if __name__ == "__main__":
    main()
