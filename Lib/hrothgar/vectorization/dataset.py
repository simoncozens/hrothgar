"""
A dataset maker for G-Tok.

Loads the Google Fonts repository and produces batches of (character, rendering, description) tuples for training.
"""

import torch
import uharfbuzz as hb

from blys.dataset import DatasetMaker
from hrothgar.vectorization.glyph import Glyph
from hrothgar.vectorization.representations.nodecommand import NodeCommand


def vectorize_glyph(font, char):
    hb_font = hb.Font(font.hb_face)  # type: ignore
    gid = hb_font.get_nominal_glyph(char)
    glyph = Glyph(glyph_id=gid, face=font.hb_face, location={})
    svgglyph = glyph.vectorize()
    nodeglyph = svgglyph.to_node_glyph()
    return nodeglyph.encode(NodeCommand)


class GTokDatasetMaker(DatasetMaker):
    def collate_fn(self, batch):
        chars = torch.tensor([item["char"] for item in batch])
        renderings = torch.stack(
            [
                torch.tensor(item["font"].render(item["char"], size=self.image_size))
                for item in batch
            ]
        )
        vectorizations = [vectorize_glyph(item["font"], item["char"]) for item in batch]

        return {
            "char": chars,
            "rendering": renderings,
            "vectorization": vectorizations,
        }
