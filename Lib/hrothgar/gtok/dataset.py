"""
A dataset maker for G-Tok.

Loads the Google Fonts repository and produces batches of
(gid, rendering, description) tuples for training.
"""

from hrothgar.dataset import AllGidsDataset, DatasetMaker
import torch


class GTokDatasetMaker(DatasetMaker):
    def train_set(self):
        return AllGidsDataset(self.train_fonts)

    def test_set(self):
        return AllGidsDataset(self.test_fonts)

    def collate_fn(self, batch):
        gids = torch.tensor([item["gid"] for item in batch])
        renderings = torch.stack(
            [
                torch.tensor(item["font"].render_gid(item["gid"], size=self.image_size))
                for item in batch
            ]
        )
        descriptions = [item["font"].description_with_tags() for item in batch]

        return {
            "gid": gids,
            "rendering": renderings,
            "description": descriptions,
        }
