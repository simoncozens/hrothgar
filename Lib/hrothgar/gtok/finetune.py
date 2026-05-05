"""Utilities for light per-font G-Tok adaptation.

This module intentionally keeps the tokenizer's encoder-side token semantics
stable while allowing the decoder path to adapt to a specific font's raster
style. The intended use is single-font adaptation before AR fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hrothgar.dataset import Dataset, LATIN_CORE
from hrothgar.gtok.losses import GtokLossWeights, compute_gtok_loss
from hrothgar.gtok.model import GtokModel


def _latin_core_filter(font_codepoints: set[int]) -> set[int]:
    return set(font_codepoints) & set(LATIN_CORE)


def _collate_gtok_batch(batch: list[dict], image_size: int) -> dict[str, torch.Tensor]:
    chars = torch.tensor([item["char"] for item in batch], dtype=torch.long)
    renderings = torch.stack(
        [
            torch.tensor(item["font"].render(item["char"], size=image_size))
            for item in batch
        ]
    )
    return {"char": chars, "rendering": renderings}


@dataclass(frozen=True)
class GtokFineTuneConfig:
    """Configuration for light decoder-only G-Tok adaptation."""

    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-5
    loss_weights: GtokLossWeights = GtokLossWeights()


def configure_decoder_only_finetuning(model: GtokModel) -> list[str]:
    """Freeze encoder/tokenizer semantics and leave only the decoder path trainable.

    Trainable modules:
    - ``quantizer_to_vit_decoder``
    - ``vit_decoder``
    - ``cnn_decoder``

    Returns:
        A sorted list of parameter names that remain trainable.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_prefixes = (
        "quantizer_to_vit_decoder",
        "vit_decoder",
        "cnn_decoder",
    )
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith(trainable_prefixes):
            parameter.requires_grad = True
            trainable_names.append(name)
    return sorted(trainable_names)


def fine_tune_gtok_decoder_only(
    *,
    model: GtokModel,
    font,
    image_size: int,
    config: GtokFineTuneConfig,
    device: torch.device,
    progress: Callable[[str], None] = print,
) -> None:
    """Adapt only the G-Tok decoder path on Latin Core glyphs from one font."""
    dataset = Dataset([font], codepoint_filter_fn=_latin_core_filter)
    if len(dataset) == 0:
        raise ValueError("Latin Core dataset is empty; cannot fine-tune G-Tok.")

    trainable_names = configure_decoder_only_finetuning(model)
    if not trainable_names:
        raise ValueError("No trainable G-Tok parameters selected for fine-tuning.")

    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
        collate_fn=lambda batch: _collate_gtok_batch(batch, image_size=image_size),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )

    progress(
        "GTok decoder-only fine-tuning on "
        f"{len(dataset)} Latin Core glyphs for {config.epochs} epochs"
    )
    for epoch in range(config.epochs):
        model.train()
        # Keep tokenization stable while adapting only the rendering path.
        model.cnn_encoder.eval()
        model.vit_encoder.eval()
        model.vit_encoder_to_quantizer.eval()
        model.quantizer.eval()
        running_loss = 0.0
        for batch in tqdm(loader, desc=f"GTok epoch {epoch + 1}/{config.epochs}"):
            images = batch["rendering"].to(device)

            optimizer.zero_grad(set_to_none=True)
            reconstructed_images, vq_loss_info = model(images)
            loss, _ = compute_gtok_loss(
                reconstructed_images,
                images,
                vq_loss_info,
                perceptual_loss_fn=None,
                weights=config.loss_weights,
            )
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())

        avg_loss = running_loss / max(len(loader), 1)
        progress(f"  GTok epoch {epoch + 1}: avg loss={avg_loss:.5f}")