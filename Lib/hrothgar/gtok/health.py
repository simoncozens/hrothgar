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
import torch.nn.functional as F
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
    _last_codebook_sim_step: int = field(default=-1, init=False)
    _last_code_entropy_step: int = field(default=-1, init=False)

    # ---- Codebook diagnostics ----
    codebook_sim_every: int = 2_000
    grad_norm_every: int = 1_000

    # ---- Code entropy (per-code font entropy) ----
    code_entropy_every: int = 2_000
    code_entropy_num_fonts: int = 100
    code_entropy_max_samples: int = 10_000
    code_entropy_batch_size: int = 64
    code_entropy_seed: int = 42

    # ---- Patch classification (white / black / mix) ----
    # Margin in [0, 1] used to classify each token-grid patch.  A patch is
    # "white" if every pixel is within this margin of 1.0, "black" if every
    # pixel is within this margin of 0.0, otherwise "mix".  Small margins are
    # more precise -- the faint anti-aliased serif tips and hairline strokes
    # that carry fine detail sit just inside "mix".
    patch_margin: float = 5 / 255


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

    # Codebook diagnostics
    codebook_mean_similarity: Optional[float] = None

    # Code entropy
    code_entropy_mean: Optional[float] = None
    code_entropy_median: Optional[float] = None
    code_entropy_active_codes: Optional[int] = None
    code_entropy_high_frac: Optional[float] = None
    code_entropy_low_frac: Optional[float] = None

    # Codebook usage by patch type (white / black / mix).
    patch_usage: Optional["PatchCategoryUsage"] = None

    # Per-code patch-type purity (how content-aligned the discrete codes are).
    purity: Optional["CodePurity"] = None


@dataclass
class PatchCategoryUsage:
    """Codebook usage broken down by patch type.

    A token-grid patch is classified as ``white`` (blank background), ``black``
    (solid ink interior), or ``mix`` (edge / transition) at encode time.

    ``*_positions`` are token counts; ``*_codes`` are distinct codebook entries
    used on that patch type; ``*_exclusive`` are entries used *only* on that
    patch type -- the strongest signal that capacity is being spent on
    blank/solid-ink patches that carry no stylistic information.
    """

    white_positions: int = 0
    black_positions: int = 0
    mix_positions: int = 0
    white_codes: int = 0
    black_codes: int = 0
    mix_codes: int = 0
    white_exclusive: int = 0
    black_exclusive: int = 0
    mix_exclusive: int = 0


@dataclass
class CodePurity:
    """Per-code patch-type purity of the codebook.

    For each active code we compute the fraction of its assignments that fall
    in its single most common patch type (white / black / mix).  Purity ≈ 1.0
    means the code is used almost entirely on one patch type (content-aligned);
    purity ≈ 1/3 means it is spread roughly uniformly across all three
    (scrambled — the code does not correspond to a single kind of content).
    """

    mean_purity: float = 0.0
    median_purity: float = 0.0
    high_purity_frac: float = 0.0  # active codes with purity >= 0.9
    scrambled_frac: float = 0.0    # active codes with purity <= 0.4


def _classify_patches(
    images: torch.Tensor,
    grid_h: int,
    grid_w: int,
    margin: float,
) -> torch.Tensor:
    """Classify each token-grid patch as 0=white, 1=black, 2=mix.

    A patch is ``white`` if every pixel is within ``margin`` of 1.0, ``black``
    if every pixel is within ``margin`` of 0.0, otherwise ``mix``.  Returns a
    ``(B, grid_h * grid_w)`` long tensor in row-major order, matching the
    token sequence produced by G-Tok's encoder.
    """
    batch, channels, height, width = images.shape
    patch_h = height // grid_h
    patch_w = width // grid_w
    # (B, C, H, W) -> (B, grid_h, grid_w, C, patch_h, patch_w)
    patches = images.reshape(
        batch, channels, grid_h, patch_h, grid_w, patch_w
    ).permute(0, 2, 4, 1, 3, 5)
    patch_min = patches.amin(dim=(3, 4, 5))  # (B, grid_h, grid_w)
    patch_max = patches.amax(dim=(3, 4, 5))

    white = patch_min >= (1.0 - margin)
    black = patch_max <= margin
    category = torch.where(white, 0, torch.where(black, 1, 2))
    return category.reshape(batch, grid_h * grid_w)


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
        self._code_entropy_cache: Optional[dict] = None

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

        # --- Codebook similarity ---
        if (
            self.config.codebook_sim_every > 0
            and global_step % self.config.codebook_sim_every == 0
        ):
            sim = self._run_codebook_similarity(gtok, writer, global_step)
            results.codebook_mean_similarity = sim

        # --- Code entropy ---
        if (
            self.config.code_entropy_every > 0
            and global_step % self.config.code_entropy_every == 0
        ):
            if self._code_entropy_cache is None:
                self._code_entropy_cache = self._build_code_entropy(gtok, image_size)
            ent_results = self._run_code_entropy(gtok, writer, global_step)
            (
                results.code_entropy_mean,
                results.code_entropy_median,
                results.code_entropy_active_codes,
                results.code_entropy_high_frac,
                results.code_entropy_low_frac,
                results.patch_usage,
                results.purity,
            ) = ent_results

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

    # ------------------------------------------------------------------
    # Codebook diagnostics
    # ------------------------------------------------------------------

    def _run_codebook_similarity(
        self,
        gtok: GtokModel,
        writer: SummaryWriter,
        global_step: int,
    ) -> float:
        """Compute mean pairwise cosine similarity of codebook entries.

        High similarity (>0.7) means entries are redundant — the codebook
        isn't using its capacity effectively.  Low similarity (<0.3) means
        entries are diverse.
        """
        weight = gtok.quantizer.embedding.weight  # (N, D)
        weight_norm = F.normalize(weight, dim=-1)
        # Pairwise cosine similarity matrix: (N, N)
        sim_matrix = weight_norm @ weight_norm.T
        # Exclude self-similarity (diagonal = 1.0)
        n = sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
        mean_sim = sim_matrix[mask].mean().item()

        writer.add_scalar(
            "Health/Codebook/MeanPairwiseSimilarity", mean_sim, global_step
        )
        return mean_sim

    # ------------------------------------------------------------------
    # Code entropy (per-code font entropy)
    # ------------------------------------------------------------------

    def _build_code_entropy(self, gtok: GtokModel, image_size: int) -> dict:
        """Pre-build a dataset for per-code font entropy computation.

        Samples glyphs from many font families and builds a DataLoader that
        yields ``(image_batch, font_index_batch)`` pairs.  Font identities
        are tracked so we can count which fonts use each codebook entry.
        """
        import numpy as np
        from torch.utils.data import DataLoader, Dataset

        from hrothgar.glyph_rendering import crop_to_ink
        from hrothgar.googlefonts import GoogleFonts

        cfg = self.config
        rng = np.random.RandomState(cfg.code_entropy_seed)

        gf = GoogleFonts(cfg.dataset_path)
        all_fonts = list(gf.fonts)
        rng.shuffle(all_fonts)

        # Group by family, then select the first N families.
        family_to_fonts: dict[str, list] = {}
        for font in all_fonts:
            family_to_fonts.setdefault(font.family, []).append(font)

        families = sorted(family_to_fonts.keys())
        if cfg.code_entropy_num_fonts > 0:
            families = families[: cfg.code_entropy_num_fonts]

        font_to_index = {fam: i for i, fam in enumerate(families)}

        # Build sample list: (font, cp, font_index) for each glyph.
        probe_chars = [
            ord(c)
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        ]
        samples: list[tuple] = []
        for fam in families:
            f_idx = font_to_index[fam]
            for font in family_to_fonts[fam]:
                available = probe_chars  # simplified: try all, let render handle misses
                for cp in available:
                    if font.has_codepoint(cp):
                        samples.append((font, cp, f_idx))

        if (
            cfg.code_entropy_max_samples > 0
            and len(samples) > cfg.code_entropy_max_samples
        ):
            indices = rng.choice(
                len(samples), cfg.code_entropy_max_samples, replace=False
            )
            samples = [samples[i] for i in indices]

        print(
            f"[CodeEntropy] Built dataset: {len(samples)} samples "
            f"across {len(families)} font families"
        )

        device = self._resolve_device(gtok)

        class _EntropyDataset(Dataset):
            """Dataset that renders glyphs (crop-to-ink) and returns
            ``(image, font_index)``, matching the G-Tok input policy."""

            def __init__(self, samples, image_size):
                self.samples = samples
                self.image_size = image_size

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                font, cp, f_idx = self.samples[idx]
                image = torch.tensor(
                    font.render(cp, size=self.image_size), dtype=torch.float32
                )
                return crop_to_ink(image, self.image_size), f_idx

        def _collate(batch):
            images = torch.stack([item[0] for item in batch])
            font_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
            return {"image": images, "font_idx": font_indices}

        dataset = _EntropyDataset(samples, image_size)
        loader = DataLoader(
            dataset,
            batch_size=cfg.code_entropy_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=_collate,
            num_workers=4,
            pin_memory=True,
        )

        return {
            "loader": loader,
            "num_fonts": len(families),
            "num_samples": len(samples),
        }

    def _run_code_entropy(
        self,
        gtok: GtokModel,
        writer: SummaryWriter,
        global_step: int,
    ) -> Tuple[float, float, int, float, float, PatchCategoryUsage, CodePurity]:
        """Compute per-code font entropy, patch-type usage, and code purity.

        For each code in the codebook, we compute the entropy of the
        distribution of fonts that use it:

            H(code) = -Σ p(font | code) * log₂(p(font | code))

        where p(font | code) is estimated from code usage counts across a
        diverse validation set.  High entropy means the code is shared across
        many fonts (good for generalization); low entropy means the code is
        specific to one or a few fonts (bad — the tokenizer is overfitting to
        font identity).

        We also classify each token-grid patch as white (blank background),
        black (solid ink interior), or mix (edge / transition) using
        ``config.patch_margin``, count how the active codebook is split across
        those patch types, and report per-code patch-type purity (how much each
        code is confined to a single patch type).

        Returns:
            Tuple of (mean_entropy_normalised, median_entropy_normalised,
                      active_codes, high_entropy_fraction, low_entropy_fraction,
                      patch_usage, purity).
        """
        import math

        cache = self._code_entropy_cache
        assert cache is not None

        device = self._resolve_device(gtok)
        loader = cache["loader"]
        num_fonts = cache["num_fonts"]
        codebook_size = gtok.config.quantizer_codebook_size
        seq_len = gtok.sequence_length

        # Accumulator: count[code, font] — how many times each code was
        # assigned to a glyph from each font.
        count = torch.zeros(codebook_size, num_fonts, dtype=torch.long, device=device)
        # Accumulator: cat_count[code, kind] — how many times each code was
        # assigned to a patch of each kind (0=white, 1=black, 2=mix).
        cat_count = torch.zeros(codebook_size, 3, dtype=torch.long, device=device)
        margin = self.config.patch_margin

        was_training = gtok.training
        gtok.eval()

        try:
            with torch.no_grad():
                for batch in loader:
                    images = batch["image"].to(device)  # (B, 3, H, W)
                    font_indices = batch["font_idx"].to(device)  # (B,)
                    B = images.shape[0]

                    # Run the encoder manually, following the same pattern as
                    # ``TokenSequenceDataset`` in ``autocorrelation.py``, to
                    # extract discrete code indices.
                    cnn_out = gtok.cnn_encoder(images)
                    tokens = gtok.proj_patch(cnn_out).flatten(2).transpose(1, 2)
                    vit_out = gtok.vit_encoder(tokens)
                    pre_quant = gtok.vit_encoder_to_quantizer(vit_out)
                    _, _channels, _h, _w = cnn_out.shape
                    pre_quant_4d = pre_quant.reshape(B, _h, _w, -1).permute(0, 3, 1, 2)
                    _, _, indices_info = gtok.quantizer(pre_quant_4d)
                    code_indices = indices_info[2].view(B, seq_len)  # (B, N)

                    # Accumulate per-code font counts.
                    for b in range(B):
                        for n in range(seq_len):
                            count[code_indices[b, n], font_indices[b]] += 1

                    # Classify patches and accumulate per-code patch-kind counts.
                    category = _classify_patches(
                        images, gtok.token_grid_height, gtok.token_grid_width, margin
                    )  # (B, N) in {0, 1, 2}
                    for kind in range(3):
                        codes = code_indices[category == kind]
                        if codes.numel() > 0:
                            cat_count[:, kind] += torch.bincount(
                                codes, minlength=codebook_size
                            )

            # --- Compute per-code entropy ---
            code_totals = count.sum(dim=1)  # (codebook_size,)
            active_mask = code_totals > 0
            num_active = int(active_mask.sum().item())

            if num_active == 0:
                writer.add_scalar("Health/CodeEntropy/Mean", 0.0, global_step)
                writer.add_scalar("Health/CodeEntropy/ActiveCodes", 0, global_step)
                return 0.0, 0.0, 0, 0.0, 0.0, PatchCategoryUsage(), CodePurity()

            active_count = count[active_mask].float()  # (A, F)
            active_totals = code_totals[active_mask].float()  # (A,)

            # p(font | code) = count[code, font] / total[code]
            p = active_count / active_totals.unsqueeze(1)  # (A, F)

            # Entropy in bits: H = -Σ p log₂ p
            # Add tiny epsilon to avoid log₂(0) = -inf (happens when a code
            # was never used for a particular font).
            entropy = -torch.sum(p * torch.log2(p + 1e-12), dim=1)  # (A,)

            # Normalize by maximum possible entropy (log₂(num_fonts)) so the
            # value is in [0, 1].  1.0 = perfect uniformity across all fonts.
            max_entropy = math.log2(max(num_fonts, 1))
            if max_entropy > 0:
                norm_entropy = entropy / max_entropy
            else:
                norm_entropy = entropy

            mean_ent = float(norm_entropy.mean().item())
            median_ent = float(norm_entropy.median().item())

            # Fraction of active codes that are "general" (entropy > 0.5 of max)
            # and "font-specific" (entropy < 0.1 of max).
            high_frac = float((norm_entropy > 0.5).float().mean().item())
            low_frac = float((norm_entropy < 0.1).float().mean().item())

            # --- Patch-type codebook usage ---
            kind_positions = cat_count.sum(dim=0)  # (3,)
            kind_codes = (cat_count > 0).sum(dim=0)  # (3,)
            # Codes used in exactly one patch kind.
            n_kinds = (cat_count > 0).sum(dim=1)  # (V,)
            kind_exclusive = torch.tensor(
                [
                    int(((n_kinds == 1) & (cat_count[:, k] > 0)).sum().item())
                    for k in range(3)
                ],
                device=device,
            )
            patch_usage = PatchCategoryUsage(
                white_positions=int(kind_positions[0].item()),
                black_positions=int(kind_positions[1].item()),
                mix_positions=int(kind_positions[2].item()),
                white_codes=int(kind_codes[0].item()),
                black_codes=int(kind_codes[1].item()),
                mix_codes=int(kind_codes[2].item()),
                white_exclusive=int(kind_exclusive[0].item()),
                black_exclusive=int(kind_exclusive[1].item()),
                mix_exclusive=int(kind_exclusive[2].item()),
            )

            # --- Per-code patch-type purity ---
            # For each active code, the fraction of its assignments in its
            # single most common patch type.
            usage = cat_count[active_mask].float()  # (A, 3)
            usage_totals = usage.sum(dim=1)  # (A,)
            dominant = usage.max(dim=1).values  # (A,)
            purity = dominant / usage_totals.clamp(min=1.0)  # (A,)
            purity_stats = CodePurity(
                mean_purity=float(purity.mean().item()),
                median_purity=float(purity.median().item()),
                high_purity_frac=float((purity >= 0.9).float().mean().item()),
                scrambled_frac=float((purity <= 0.4).float().mean().item()),
            )

            # --- Log to TensorBoard ---
            writer.add_scalar("Health/CodeEntropy/Mean", mean_ent, global_step)
            writer.add_scalar("Health/CodeEntropy/Median", median_ent, global_step)
            writer.add_scalar("Health/CodeEntropy/ActiveCodes", num_active, global_step)
            writer.add_scalar(
                "Health/CodeEntropy/FractionHighEntropy",
                high_frac,
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/FractionLowEntropy",
                low_frac,
                global_step,
            )
            writer.add_histogram(
                "Health/CodeEntropy/Distribution",
                norm_entropy.cpu(),
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/PatchWhiteCodes",
                patch_usage.white_codes,
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/PatchBlackCodes",
                patch_usage.black_codes,
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/PatchMixCodes",
                patch_usage.mix_codes,
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/PatchWhiteExclusive",
                patch_usage.white_exclusive,
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/PatchBlackExclusive",
                patch_usage.black_exclusive,
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/PatchMixExclusive",
                patch_usage.mix_exclusive,
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/PurityMean",
                purity_stats.mean_purity,
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/PurityMedian",
                purity_stats.median_purity,
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/PurityHighFraction",
                purity_stats.high_purity_frac,
                global_step,
            )
            writer.add_scalar(
                "Health/CodeEntropy/PurityScrambledFraction",
                purity_stats.scrambled_frac,
                global_step,
            )

            return (
                mean_ent,
                median_ent,
                num_active,
                high_frac,
                low_frac,
                patch_usage,
                purity_stats,
            )

        finally:
            if was_training:
                gtok.train()

    @staticmethod
    def log_gradient_norms(
        gtok: GtokModel,
        writer: SummaryWriter,
        global_step: int,
    ) -> None:
        """Log L2 gradient norms of encoder vs codebook parameters.

        Called from the training loop after ``loss.backward()`` but before
        ``optimizer.zero_grad()``.  A collapse in encoder gradient norm
        relative to codebook gradient norm indicates the straight-through
        estimator is stalling — the quantizer is rejecting encoder updates.
        """
        encoder_norm_sq = 0.0
        vit_norm_sq = 0.0

        for name, param in gtok.named_parameters():
            if param.grad is None:
                continue
            gn = param.grad.data.norm(2).item() ** 2
            if "quantizer.embedding" in name:
                codebook_norm_sq += gn
            elif "vit_encoder" in name or "vit_encoder_to_quantizer" in name:
                vit_norm_sq += gn
            elif "cnn_encoder" in name:
                encoder_norm_sq += gn

        cnn_grad = encoder_norm_sq**0.5
        vit_grad = vit_norm_sq**0.5

        writer.add_scalar("Health/GradNorm/CNN_Encoder", cnn_grad, global_step)
        writer.add_scalar("Health/GradNorm/ViT_Encoder", vit_grad, global_step)
        writer.add_scalar(
            "Health/GradNorm/Encoder_Total",
            (encoder_norm_sq + vit_norm_sq) ** 0.5,
            global_step,
        )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


class _NullWriter:
    """Drop-in ``SummaryWriter`` stand-in that discards every call.

    ``GtokHealthCheck.maybe_run`` logs every check to a ``SummaryWriter``.
    For a standalone console run we don't want TensorBoard event files, so
    we hand it this no-op writer and print ``HealthCheckResults`` ourselves.
    """

    def add_scalar(self, *args, **kwargs) -> None:
        del args, kwargs

    def add_histogram(self, *args, **kwargs) -> None:
        del args, kwargs

    def add_image(self, *args, **kwargs) -> None:
        del args, kwargs

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _print_results(results: HealthCheckResults) -> None:
    """Print every non-``None`` health-check result to the console."""
    sections: list[str] = []

    if results.autocorr_accuracy is not None:
        sections.append(
            "Autocorrelation (next-token prediction, 1-layer probe)\n"
            f"  accuracy           : {results.autocorr_accuracy:.4f}\n"
            f"  x chance           : {results.autocorr_x_chance:.2f}x\n"
            f"  within-row acc     : {results.autocorr_within_row_acc:.4f}\n"
            f"  cross-row acc      : {results.autocorr_cross_row_acc:.4f}\n"
            f"  within/cross ratio : {results.autocorr_within_cross_ratio:.2f}"
        )

    if results.oracle_ar_accuracy is not None:
        sections.append(
            "Oracle AR (single-font, conditionless GPT)\n"
            f"  accuracy : {results.oracle_ar_accuracy:.4f}\n"
            f"  x chance : {results.oracle_ar_x_chance:.2f}x"
        )

    if results.linear_probe_char_acc is not None:
        sections.append(
            "Linear probe (frozen features)\n"
            f"  character acc : {results.linear_probe_char_acc:.4f}\n"
            f"  font acc      : {results.linear_probe_font_acc:.4f}"
        )

    if results.codebook_mean_similarity is not None:
        sections.append(
            "Codebook similarity (mean pairwise cosine)\n"
            f"  mean similarity : {results.codebook_mean_similarity:.4f}  "
            "(>0.7 redundant, <0.3 diverse)"
        )

    if results.code_entropy_mean is not None:
        sections.append(
            "Code entropy (per-code font entropy, normalised)\n"
            f"  mean          : {results.code_entropy_mean:.4f}\n"
            f"  median        : {results.code_entropy_median:.4f}\n"
            f"  active codes  : {results.code_entropy_active_codes}\n"
            f"  high-entropy  : {results.code_entropy_high_frac:.4f}  (shared across fonts)\n"
            f"  low-entropy   : {results.code_entropy_low_frac:.4f}  (font-specific)"
        )

    if results.patch_usage is not None:
        pu = results.patch_usage
        total = pu.white_positions + pu.black_positions + pu.mix_positions
        denom = max(total, 1)

        def _pct(x: int) -> float:
            return 100.0 * x / denom

        sections.append(
            "Codebook usage by patch type\n"
            f"  white (blank) : {_pct(pu.white_positions):5.1f}% of positions  "
            f"{pu.white_codes} codes ({pu.white_exclusive} exclusive)\n"
            f"  black (ink)   : {_pct(pu.black_positions):5.1f}% of positions  "
            f"{pu.black_codes} codes ({pu.black_exclusive} exclusive)\n"
            f"  mix (edge)    : {_pct(pu.mix_positions):5.1f}% of positions  "
            f"{pu.mix_codes} codes ({pu.mix_exclusive} exclusive)"
        )

    if results.purity is not None:
        pur = results.purity
        sections.append(
            "Code purity (per-code patch-type concentration)\n"
            f"  mean          : {pur.mean_purity:.4f}\n"
            f"  median        : {pur.median_purity:.4f}\n"
            f"  high (>=0.9)  : {pur.high_purity_frac:.4f}  (content-aligned codes)\n"
            f"  scrambled     : {pur.scrambled_frac:.4f}  (<=0.4, uniform across types)"
        )

    if not sections:
        print("No health checks ran (all disabled).")
        return

    print()
    print("\n".join(sections))


def main(argv: Optional[list[str]] = None) -> None:
    """Run G-Tok health checks on a trained tokenizer and print the results.

    Loads the model from ``--gtok-model-path`` (plus its sidecar config),
    runs every scheduled health check once, and prints a readable summary
    to the console instead of writing TensorBoard event files.
    """
    import argparse
    import os
    from pathlib import Path

    from hrothgar.gtok.model import load_model
    from hrothgar.utils import pick_device

    parser = argparse.ArgumentParser(
        description="Run G-Tok health checks on a trained tokenizer."
    )
    parser.add_argument(
        "--gtok-model-path",
        type=Path,
        default=Path("models/gtok.pth"),
        help="Path to trained G-Tok weights (.pth); sidecar .conf.json must exist beside it",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(os.environ.get("GOOGLE_FONTS_REPO", "")),
        help="Path to the Google Fonts repository (or set GOOGLE_FONTS_REPO)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use (e.g. cuda, mps, cpu). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        choices=["autocorr", "oracle-ar", "linear-probe", "codebook", "entropy"],
        default=[],
        help="Skip a specific check (repeatable).",
    )
    parser.add_argument(
        "--patch-margin",
        type=float,
        default=5.0,
        help=(
            "Pixel-value margin (8-bit units, 0-255) used to classify each "
            "token patch as white/black/mix. A patch is 'white' if every pixel "
            "is within this many levels of 255, 'black' if within it of 0. "
            "Smaller margins are more precise; run with several values to see "
            "where the fine detail sits. (default: 5)"
        ),
    )
    args = parser.parse_args(argv)

    if not args.gtok_model_path.exists():
        parser.error(f"G-Tok model not found: {args.gtok_model_path}")
    if not args.dataset_path or not args.dataset_path.exists():
        parser.error(
            "--dataset-path is required (or set the GOOGLE_FONTS_REPO env var)"
        )

    device = torch.device(args.device) if args.device else pick_device()
    print(f"Loading G-Tok model from {args.gtok_model_path} (device={device})")

    gtok, gtok_config = load_model(args.gtok_model_path, device)
    image_size = gtok_config.image_size
    gtok.eval()
    for param in gtok.parameters():
        param.requires_grad = False

    print(
        f"Loaded G-Tok model (image_size={image_size}, "
        f"codebook_size={gtok_config.quantizer_codebook_size})"
    )

    skip = set(args.skip)
    config = HealthCheckConfig(
        dataset_path=str(args.dataset_path),
        patch_margin=args.patch_margin / 255.0,
    )
    if "autocorr" in skip:
        config.autocorr_every = 0
    if "oracle-ar" in skip:
        config.oracle_ar_every = 0
    if "linear-probe" in skip:
        config.linear_probe_every = 0
    if "codebook" in skip:
        config.codebook_sim_every = 0
    if "entropy" in skip:
        config.code_entropy_every = 0

    health = GtokHealthCheck(config)
    print("Running health checks...")
    results = health.maybe_run(
        gtok=gtok,
        image_size=image_size,
        global_step=0,
        writer=_NullWriter(),  # type: ignore[arg-type]
    )
    _print_results(results)


if __name__ == "__main__":
    main()
