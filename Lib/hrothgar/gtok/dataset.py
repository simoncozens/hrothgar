"""
A dataset maker for G-Tok.

Loads the Google Fonts repository and produces batches of (character, rendering, description) tuples for training.
"""

from hrothgar.dataset import DatasetMaker
import torch


class GTokDatasetMaker(DatasetMaker):
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
