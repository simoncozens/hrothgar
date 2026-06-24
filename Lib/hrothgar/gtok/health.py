"""G-Tok health checks: integrated diagnostics for the training loop.

Surfaces linear probing, autocorrelation, and oracle AR results as
TensorBoard scalars at configured intervals during tokenizer training.

All probes accept a live ``GtokModel`` instance (no disk round-trip) so
they can be called inline during training without extra serialisation.

Usage as module::

    from hrothgar.gtok.health import GtokHealthCheck, HealthCheckConfig

    config = HealthCheckConfig(
        dataset_path="/path/to/google/fonts",
        linear_probe_every=10_000,
        autocorr_every=5_000,
        oracle_ar_every=2_000,
    )
    health = GtokHealthCheck(config)
    # Inside the training loop:
    results = health.maybe_run(
        gtok=model,
        image_size=128,
        global_step=1000,
        writer=tensorboard_writer,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from hrothgar.gtok.model import GtokConfig, GtokModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HealthCheckConfig:
    """Schedule and parameters for inline health checks during G-Tok training.

    All probes accept a live ``GtokModel`` — no serialisation step needed.
    Each check runs only when ``global_step % every_N == 0``.  Set an
    interval to 0 to disable that check entirely.
    """

    dataset_path: str = ""

    # ---- Autocorrelation (next-token prediction with 1-layer probe) ----
    autocorr_every: int = 5_000
    autocorr_epochs: int = 5
    autocorr_max_samples: int = 1_000
    autocorr_batch_size: int = 64
    autocorr_lr: float = 1e-3
    autocorr_hidden_dim: int = 128
    autocorr_seed: int = 42

    # ---- Oracle AR (single-font, conditionless GPT) ----
    oracle_ar_every: int = 2_000
    oracle_ar_steps: int = 1_000
    oracle_ar_batch_size: int = 32
    oracle_ar_lr: float = 1e-4
    oracle_ar_font_index: int = 0
    oracle_ar_seed: int = 42
    oracle_ar_dim: int = 128
    oracle_ar_layers: int = 4
    oracle_ar_heads: int = 4

    # ---- Linear probing (character + font-family probes) ----
    linear_probe_every: int = 10_000
    linear_probe_epochs: int = 10
    linear_probe_batch_size: int = 64
    linear_probe_lr: float = 1e-3
    linear_probe_weight_decay: float = 1e-4
    linear_probe_font_count: int = 20
    linear_probe_font_min_samples: int = 1
    linear_probe_max_samples: int = 50_000
    linear_probe_seed: int = 42

    # ---- Internal bookkeeping (not user-facing) ----
    _last_autocorr_step: int = field(default=-1, init=False)
    _last_oracle_ar_step: int = field(default=-1, init=False)
    _last_linear_probe_step: int = field(default=-1, init=False)


@dataclass
class HealthCheckResults:
    """Aggregated results from one round of health checks.

    All fields are ``None`` when the corresponding check did not run.
    """

    # Autocorrelation
    autocorr_accuracy: Optional[float] = None
    autocorr_x_chance: Optional[float] = None
    autocorr_within_row_acc: Optional[float] = None
    autocorr_cross_row_acc: Optional[float] = None
    autocorr_within_cross_ratio: Optional[float] = None

    # Oracle AR
    oracle_ar_accuracy: Optional[float] = None
    oracle_ar_x_chance: Optional[float] = None

    # Linear probing
    linear_probe_char_acc: Optional[float] = None
    linear_probe_font_acc: Optional[float] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class GtokHealthCheck:
    """Run a set of tokenizer-quality diagnostics at scheduled intervals.

    Intended to be called from the G-Tok training loop's validation step.
    Each probe builds fresh datasets on first call (with a fixed seed) and
    reuses cached loaders across steps to amortise rendering costs.

    Typical usage inside ``GtokTrainingLoop.post_train_step``::

        results = self.health.maybe_run(
            gtok=self.model,
            image_size=config.image_size,
            global_step=self.global_step,
            writer=self.writer,
        )
    """

    def __init__(self, config: HealthCheckConfig) -> None:
        self.config = config
        self._device: Optional[torch.device] = None

        # Cached datasets / probes — built once, reused across calls.
        self._autocorr_cache: Optional[dict] = None
        self._oracle_ar_cache: Optional[dict] = None
        self._linear_probe_cache: Optional[dict] = None

    def maybe_run(
        self,
        gtok: GtokModel,
        image_size: int,
        global_step: int,
        writer: SummaryWriter,
    ) -> HealthCheckResults:
        """Run any health checks that are due at this training step.

        Args:
            gtok: The live G-Tok model (may be in train mode; probes will
                switch it to eval temporarily).
            image_size: Current training image size (read from config).
            global_step: Current training step counter.
            writer: TensorBoard ``SummaryWriter`` for scalar logging.

        Returns:
            ``HealthCheckResults`` with fields for checks that actually ran.
        """
        results = HealthCheckResults()

        # --- Autocorrelation ---
        if (
            self.config.autocorr_every > 0
            and global_step % self.config.autocorr_every == 0
        ):
            if self._autocorr_cache is None:
                self._autocorr_cache = self._build_autocorr(gtok, image_size)
            acc, xchance, within, cross, ratio = self._run_autocorr(
                gtok, image_size, writer, global_step
            )
            results.autocorr_accuracy = acc
            results.autocorr_x_chance = xchance
            results.autocorr_within_row_acc = within
            results.autocorr_cross_row_acc = cross
            results.autocorr_within_cross_ratio = ratio

        # --- Oracle AR ---
        if (
            self.config.oracle_ar_every > 0
            and global_step % self.config.oracle_ar_every == 0
        ):
            if self._oracle_ar_cache is None:
                self._oracle_ar_cache = self._build_oracle_ar(gtok, image_size)
            acc, xchance = self._run_oracle_ar(gtok, image_size, writer, global_step)
            results.oracle_ar_accuracy = acc
            results.oracle_ar_x_chance = xchance

        # --- Linear probing ---
        if (
            self.config.linear_probe_every > 0
            and global_step % self.config.linear_probe_every == 0
        ):
            if self._linear_probe_cache is None:
                self._linear_probe_cache = self._build_linear_probe(gtok, image_size)
            char_acc, font_acc = self._run_linear_probe(
                gtok, image_size, writer, global_step
            )
            results.linear_probe_char_acc = char_acc
            results.linear_probe_font_acc = font_acc

        return results

    # ------------------------------------------------------------------
    # Autocorrelation
    # ------------------------------------------------------------------

    def _build_autocorr(self, gtok: GtokModel, image_size: int) -> dict:
        """Pre-build datasets and probe for autocorrelation checks."""
        import numpy as np
        from torch.utils.data import DataLoader

        from hrothgar.googlefonts import GoogleFonts
        from hrothgar.gtok.autocorrelation import (
            NextTokenProbe,
            TokenSequenceDataset,
            _collate_indices,
        )

        cfg = self.config
        rng = np.random.RandomState(cfg.autocorr_seed)

        gf = GoogleFonts(cfg.dataset_path)
        all_fonts = list(gf.fonts)
        rng.shuffle(all_fonts)

        probe_chars = list(range(ord("A"), ord("Z") + 1))

        samples = []
        for font in all_fonts:
            for cp in probe_chars:
                samples.append((font, cp))

        if cfg.autocorr_max_samples > 0 and len(samples) > cfg.autocorr_max_samples:
            indices = rng.choice(len(samples), cfg.autocorr_max_samples, replace=False)
            samples = [samples[i] for i in indices]

        split = int(len(samples) * 0.8)
        train_samples = samples[:split]
        test_samples = samples[split:]

        device = self._resolve_device(gtok)
        train_dataset = TokenSequenceDataset(train_samples, gtok, image_size, device)
        test_dataset = TokenSequenceDataset(test_samples, gtok, image_size, device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.autocorr_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=_collate_indices,
            num_workers=0,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.autocorr_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate_indices,
            num_workers=0,
            pin_memory=False,
        )

        probe = NextTokenProbe(
            vocab_size=gtok.config.quantizer_codebook_size,
            hidden_dim=cfg.autocorr_hidden_dim,
        ).to(device)

        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "probe": probe,
            "grid_width": gtok.token_grid_width,
            "vocab_size": gtok.config.quantizer_codebook_size,
        }

    def _run_autocorr(
        self,
        gtok: GtokModel,
        image_size: int,
        writer: SummaryWriter,
        global_step: int,
    ) -> Tuple[float, float, float, float, float]:
        """Train the autocorrelation probe and return summary metrics."""
        import tqdm

        cache = self._autocorr_cache
        assert cache is not None

        probe = cache["probe"]
        train_loader = cache["train_loader"]
        test_loader = cache["test_loader"]
        vocab_size = cache["vocab_size"]
        grid_width = cache["grid_width"]
        device = self._resolve_device(gtok)

        chance = 1.0 / vocab_size

        was_training = gtok.training
        gtok.eval()

        optimizer = torch.optim.AdamW(probe.parameters(), lr=self.config.autocorr_lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        try:
            best_acc = 0.0
            for epoch in range(self.config.autocorr_epochs):
                probe.train()
                for batch in train_loader:
                    token_indices = batch.to(device)
                    B, N = token_indices.shape
                    logits = probe(token_indices)
                    targets = token_indices[:, 1:]
                    loss = loss_fn(
                        logits.reshape(B * (N - 1), -1),
                        targets.reshape(B * (N - 1)),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                probe.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in test_loader:
                        token_indices = batch.to(device)
                        B, N = token_indices.shape
                        logits = probe(token_indices)
                        targets = token_indices[:, 1:]
                        preds = torch.argmax(logits, dim=-1)
                        correct += (preds == targets).sum().item()
                        total += B * (N - 1)

                acc = correct / total if total > 0 else 0.0
                best_acc = max(best_acc, acc)

            # Per-position: within-row vs cross-row
            within_correct = 0
            within_total = 0
            cross_correct = 0
            cross_total = 0
            probe.eval()
            with torch.no_grad():
                for batch in test_loader:
                    token_indices = batch.to(device)
                    B, N = token_indices.shape
                    logits = probe(token_indices)
                    targets = token_indices[:, 1:]
                    preds = torch.argmax(logits, dim=-1)
                    for pos in range(N - 1):
                        is_cross = (pos + 1) % grid_width == 0
                        if is_cross:
                            cross_correct += (
                                (preds[:, pos] == targets[:, pos]).sum().item()
                            )
                            cross_total += B
                        else:
                            within_correct += (
                                (preds[:, pos] == targets[:, pos]).sum().item()
                            )
                            within_total += B

            within_acc = within_correct / max(within_total, 1)
            cross_acc = cross_correct / max(cross_total, 1)
            ratio = within_acc / cross_acc if cross_acc > 0 else 0.0

            x_chance = best_acc / chance if chance > 0 else 0.0

            writer.add_scalar("Health/Autocorr/Accuracy", best_acc, global_step)
            writer.add_scalar("Health/Autocorr/xChance", x_chance, global_step)
            writer.add_scalar("Health/Autocorr/WithinRow", within_acc, global_step)
            writer.add_scalar("Health/Autocorr/CrossRow", cross_acc, global_step)
            writer.add_scalar("Health/Autocorr/WithinCrossRatio", ratio, global_step)

            return best_acc, x_chance, within_acc, cross_acc, ratio
        finally:
            if was_training:
                gtok.train()

    # ------------------------------------------------------------------
    # Oracle AR
    # ------------------------------------------------------------------

    def _build_oracle_ar(self, gtok: GtokModel, image_size: int) -> dict:
        """Pre-build dataset and model for oracle AR checks."""
        from torch.utils.data import DataLoader

        from hrothgar.googlefonts import GoogleFonts
        from hrothgar.gtok.oracle_ar import (
            CausalDecoder,
            SingleFontTokenDataset,
            _collate_oracle,
        )

        cfg = self.config
        device = self._resolve_device(gtok)

        gf = GoogleFonts(cfg.dataset_path)
        all_fonts = sorted(gf.fonts, key=lambda f: f.family)
        font = all_fonts[cfg.oracle_ar_font_index % len(all_fonts)]
        font_name = font.family

        vocab_size = gtok.config.quantizer_codebook_size

        dataset = SingleFontTokenDataset(font, gtok, image_size, device)
        seq_len = dataset.token_sequences[0].shape[0]

        loader = DataLoader(
            dataset,
            batch_size=cfg.oracle_ar_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=_collate_oracle,
        )

        model = CausalDecoder(
            vocab_size=vocab_size,
            dim=cfg.oracle_ar_dim,
            n_layers=cfg.oracle_ar_layers,
            n_heads=cfg.oracle_ar_heads,
            max_seq_len=seq_len + 1,
        ).to(device)

        return {
            "loader": loader,
            "model": model,
            "vocab_size": vocab_size,
            "font_name": font_name,
            "num_glyphs": len(dataset),
        }

    def _run_oracle_ar(
        self,
        gtok: GtokModel,
        image_size: int,
        writer: SummaryWriter,
        global_step: int,
    ) -> Tuple[float, float]:
        """Run the oracle AR probe and return (best_accuracy, x_chance)."""
        cache = self._oracle_ar_cache
        assert cache is not None

        cfg = self.config
        device = self._resolve_device(gtok)
        model = cache["model"]
        loader = cache["loader"]
        vocab_size = cache["vocab_size"]
        chance = 1.0 / vocab_size

        was_training = gtok.training
        gtok.eval()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.oracle_ar_lr, betas=(0.9, 0.95)
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        data_iter = iter(loader)

        try:
            running_correct = 0
            running_total = 0
            best_acc = 0.0

            model.train()
            for _ in range(cfg.oracle_ar_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                token_indices = batch.to(device)
                input_ids = token_indices[:, :-1]
                targets = token_indices[:, 1:]

                logits = model(input_ids)

                loss = loss_fn(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                preds = torch.argmax(logits, dim=-1)
                running_correct += (preds == targets).sum().item()
                running_total += targets.numel()

            acc = running_correct / max(running_total, 1)
            best_acc = max(best_acc, acc)
            x_chance = best_acc / chance if chance > 0 else 0.0

            writer.add_scalar("Health/OracleAR/Accuracy", best_acc, global_step)
            writer.add_scalar("Health/OracleAR/xChance", x_chance, global_step)

            return best_acc, x_chance
        finally:
            if was_training:
                gtok.train()

    # ------------------------------------------------------------------
    # Linear probing
    # ------------------------------------------------------------------

    def _build_linear_probe(self, gtok: GtokModel, image_size: int) -> dict:
        """Pre-build probe datasets for linear probing."""
        import numpy as np
        from torch.utils.data import DataLoader

        from hrothgar.googlefonts import GoogleFonts
        from hrothgar.gtok.linear_probing import (
            _CHAR_TO_INDEX,
            _PROBE_CHARS,
            FrozenGtokFeatureExtractor,
            LinearProbe,
            ProbeDataset,
            _collate_probe_batch,
        )

        cfg = self.config
        device = self._resolve_device(gtok)
        rng = np.random.RandomState(cfg.linear_probe_seed)

        gf = GoogleFonts(cfg.dataset_path)
        # Use the same filter as linear probing CLI: display_score < 60.
        gf.fonts = [font for font in gf.fonts if font.display_score() < 60.0]

        family_to_fonts = {}
        for font in gf.fonts:
            family_to_fonts.setdefault(font.family, []).append(font)

        eligible_families = [
            fam
            for fam, fonts in family_to_fonts.items()
            if len(fonts) >= cfg.linear_probe_font_min_samples
        ]
        eligible_families.sort()
        if cfg.linear_probe_font_count > 0:
            eligible_families = eligible_families[: cfg.linear_probe_font_count]

        family_to_label = {fam: i for i, fam in enumerate(eligible_families)}
        num_font_classes = len(family_to_label)
        num_char_classes = len(_CHAR_TO_INDEX)

        train_samples = []
        test_samples = []
        train_frac = 0.8

        for family in eligible_families:
            fonts = family_to_fonts[family]
            rng.shuffle(fonts)
            family_label = family_to_label[family]
            n_fonts = len(fonts)

            if n_fonts == 1:
                if rng.random() < train_frac:
                    train_fonts_for_family = fonts
                    test_fonts_for_family = []
                else:
                    train_fonts_for_family = []
                    test_fonts_for_family = fonts
            else:
                n_train = max(1, int(round(n_fonts * train_frac)))
                n_train = min(n_train, n_fonts - 1)
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

        if (
            cfg.linear_probe_max_samples > 0
            and len(train_samples) > cfg.linear_probe_max_samples
        ):
            rng.shuffle(train_samples)
            train_samples = train_samples[: cfg.linear_probe_max_samples]

        # Compute feature dimension from the model config.
        downsampling_factor = 2 ** (len(gtok.config.cnn_channel_multipliers or []) - 1)
        grid_h = image_size // downsampling_factor
        grid_w = image_size // downsampling_factor
        feature_dim = gtok.config.quantizer_code_dim * grid_h * grid_w

        train_dataset = ProbeDataset(train_samples, image_size)
        test_dataset = ProbeDataset(test_samples, image_size)

        def collate_fn(batch):
            return _collate_probe_batch(batch, image_size)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.linear_probe_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.linear_probe_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        extractor = FrozenGtokFeatureExtractor(gtok, gtok.config, device)

        char_probe = LinearProbe(feature_dim, num_char_classes).to(device)
        font_probe = LinearProbe(feature_dim, num_font_classes).to(device)

        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "extractor": extractor,
            "char_probe": char_probe,
            "font_probe": font_probe,
            "num_char_classes": num_char_classes,
            "num_font_classes": num_font_classes,
        }

    def _run_linear_probe(
        self,
        gtok: GtokModel,
        image_size: int,
        writer: SummaryWriter,
        global_step: int,
    ) -> Tuple[float, float]:
        """Run both linear probes; return (char_acc, font_acc)."""
        cache = self._linear_probe_cache
        assert cache is not None

        cfg = self.config
        device = self._resolve_device(gtok)

        train_loader = cache["train_loader"]
        test_loader = cache["test_loader"]
        extractor = cache["extractor"]
        char_probe = cache["char_probe"]
        font_probe = cache["font_probe"]

        was_training = gtok.training
        gtok.eval()

        try:
            char_acc = self._train_one_probe(
                probe=char_probe,
                extractor=extractor,
                train_loader=train_loader,
                test_loader=test_loader,
                label_key="char_label",
                epochs=cfg.linear_probe_epochs,
                lr=cfg.linear_probe_lr,
                weight_decay=cfg.linear_probe_weight_decay,
                device=device,
            )
            font_acc = self._train_one_probe(
                probe=font_probe,
                extractor=extractor,
                train_loader=train_loader,
                test_loader=test_loader,
                label_key="family_label",
                epochs=cfg.linear_probe_epochs,
                lr=cfg.linear_probe_lr,
                weight_decay=cfg.linear_probe_weight_decay,
                device=device,
            )

            writer.add_scalar("Health/LinearProbe/CharAccuracy", char_acc, global_step)
            writer.add_scalar("Health/LinearProbe/FontAccuracy", font_acc, global_step)

            return char_acc, font_acc
        finally:
            if was_training:
                gtok.train()

    @staticmethod
    def _train_one_probe(
        probe,
        extractor,
        train_loader,
        test_loader,
        label_key: str,
        epochs: int,
        lr: float,
        weight_decay: float,
        device: torch.device,
    ) -> float:
        """Train a single linear probe and return best test accuracy."""
        import torch.nn as nn

        optimizer = torch.optim.AdamW(
            probe.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0.0

        for _epoch in range(epochs):
            probe.train()
            for batch in train_loader:
                images = batch["images"].to(device)
                labels = batch[label_key].to(device)

                features = extractor.extract(images)
                logits = probe(features)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            probe.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    images = batch["images"].to(device)
                    labels = batch[label_key].to(device)
                    features = extractor.extract(images)
                    logits = probe(features)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.shape[0]

            acc = correct / total if total > 0 else 0.0
            best_acc = max(best_acc, acc)

        return best_acc

    def _resolve_device(self, gtok: GtokModel) -> torch.device:
        """Return the device the model is on."""
        if self._device is None:
            self._device = next(gtok.parameters()).device
        return self._device
