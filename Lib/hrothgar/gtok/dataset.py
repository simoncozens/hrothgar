"""A dataset maker for G-Tok.

Loads the Google Fonts repository and produces batches of
(char, rendering, description) tuples for training.
"""

import torch

from hrothgar.dataset import Dataset, DatasetMaker, LATIN_CORE


class GTokDatasetMaker(DatasetMaker):
    def __init__(self, repo_url: str, batch_size: int, **kwargs):
        target = kwargs.pop("target_codepoints", None)
        if target is None:
            target = set(LATIN_CORE)
        super().__init__(
            repo_url=repo_url,
            batch_size=batch_size,
            having=kwargs.pop("having", None),
            target_codepoints=target,
            canary_size=kwargs.pop("canary_size", None),
            image_size=kwargs.pop("image_size", 128),
            **kwargs,
        )

    def train_set(self):
        return Dataset(self.train_fonts, codepoint_filter_fn=self.train_codepoint_filter)

    def test_set(self):
        return Dataset(self.test_fonts, codepoint_filter_fn=self.test_codepoint_filter)

    def collate_fn(self, batch):
        chars = torch.tensor([item["char"] for item in batch])
        renderings = torch.stack(
            [
                torch.tensor(item["font"].render(item["char"], size=self.image_size))
                for item in batch
            ]
        )
        descriptions = [item["font"].description_with_tags() for item in batch]

        return {
            "char": chars,
            "rendering": renderings,
            "description": descriptions,
        }
