"""Dataset utilities for AR phase-1 visual pretraining.

This module reuses the shared font split logic from ``hrothgar.dataset`` and
provides a collation function tailored to the AR generator inputs.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence, Set

import torch
from hrothgar.dataset import DatasetMaker


def _sample_style_codepoints(
    *,
    font,
    target_char: int,
    style_glyph_count: int,
    common_style_codepoints: Optional[Sequence[int]],
) -> list[int]:
    """Select style codepoints for one target item.

    By default this samples per-item codepoints from the same font, excluding
    the target codepoint. If a common list is provided, it is used first and
    then padded with per-font sampling when needed.
    """

    if style_glyph_count <= 0:
        raise ValueError(f"style_glyph_count must be positive, got {style_glyph_count}")

    available = [cp for cp in font.codepoints if cp != target_char]
    if not available:
        return [target_char] * style_glyph_count

    selected: list[int] = []
    if common_style_codepoints is not None:
        selected = [
            cp
            for cp in common_style_codepoints
            if cp != target_char and cp in font.codepoints
        ]

    if len(selected) >= style_glyph_count:
        return selected[:style_glyph_count]

    remaining_count = style_glyph_count - len(selected)
    remaining_candidates = [cp for cp in available if cp not in selected]
    if len(remaining_candidates) >= remaining_count:
        selected.extend(random.sample(remaining_candidates, remaining_count))
    else:
        selected.extend(remaining_candidates)
        if selected:
            selected.extend(
                random.choices(selected, k=style_glyph_count - len(selected))
            )
        else:
            selected.extend([target_char] * (style_glyph_count - len(selected)))

    return selected


class ARPhase1DatasetMaker(DatasetMaker):
    """Dataset maker for AR phase-1 visual pretraining.

    This class reuses the same train/test split and item ordering logic as the
    GTok dataset maker, but emits AR-specific batches during collation.
    """

    def __init__(
        self,
        repo_url: str,
        batch_size: int,
        having: Optional[Set[int]] = None,
        canary_size: Optional[int] = None,
        image_size: int = 128,
        style_glyph_count: int = 8,
        common_style_codepoints: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(
            repo_url=repo_url,
            batch_size=batch_size,
            having=having,
            canary_size=canary_size,
            image_size=image_size,
        )
        if style_glyph_count <= 0:
            raise ValueError(
                f"style_glyph_count must be positive, got {style_glyph_count}"
            )
        self.style_glyph_count = style_glyph_count
        self.common_style_codepoints = common_style_codepoints

    def collate_fn(self, batch):
        """Collate samples for AR visual pretraining.

        Returns:
        - ``target_rendering``: target font glyph image (ground truth)
        - ``content_rendering``: same codepoint rendered by the reference font
        - ``style_renderings``: style support set from the target font
        Plus metadata useful for debugging and future adaptation wiring.
        """

        chars = torch.tensor([item["char"] for item in batch], dtype=torch.long)
        target_renderings = torch.stack(
            [
                torch.tensor(item["font"].render(item["char"], size=self.image_size))
                for item in batch
            ]
        )

        content_renderings = []
        style_renderings = []
        style_chars = []
        descriptions = []

        for item in batch:
            font = item["font"]
            char = item["char"]
            reference_font = font.reference_font() or font

            content_renderings.append(
                torch.tensor(reference_font.render(char, size=self.image_size))
            )

            sampled_style_chars = _sample_style_codepoints(
                font=font,
                target_char=char,
                style_glyph_count=self.style_glyph_count,
                common_style_codepoints=self.common_style_codepoints,
            )
            style_chars.append(sampled_style_chars)
            style_renderings.append(
                torch.stack(
                    [
                        torch.tensor(font.render(cp, size=self.image_size))
                        for cp in sampled_style_chars
                    ]
                )
            )
            descriptions.append(font.description_with_tags())

        return {
            "char": chars,
            "target_rendering": target_renderings,
            "content_rendering": torch.stack(content_renderings),
            "style_renderings": torch.stack(style_renderings),
            "style_chars": torch.tensor(style_chars, dtype=torch.long),
            "description": descriptions,
        }


__all__ = ["ARPhase1DatasetMaker"]
