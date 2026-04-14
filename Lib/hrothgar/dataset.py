from typing import Optional, Set

import uharfbuzz as hb
from glyphsets import GlyphSet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.googlefonts import GoogleFonts

LATIN_CORE = GlyphSet("GF_Latin_Core").get_characters()


class DatasetMaker:
    """Create train/test splits and loaders over glyph rendering items."""

    def __init__(
        self,
        repo_url: str,
        batch_size: int,
        having: Optional[Set[int]] = None,
        canary_size: Optional[int] = None,
        image_size: int = 128,
    ):
        self.googlefonts = GoogleFonts(repo_url, having=having)
        self.batch_size = batch_size
        self.image_size = image_size

        # Test chars are a random split from GF Latin Core.
        _, self.test_latincore_chars = train_test_split(LATIN_CORE)

        if canary_size is not None:
            fonts = self.googlefonts.fonts[:canary_size]
        else:
            fonts = self.googlefonts.fonts

        self.train_fonts, self.test_fonts = train_test_split(fonts)
        print("Train fonts:", len(self.train_fonts))
        print("Test fonts:", len(self.test_fonts))

    def train_set(self):
        return Dataset(self.train_fonts, self.test_latincore_chars, test=False)

    def test_set(self):
        return Dataset(self.test_fonts, self.test_latincore_chars, test=True)

    def train_loader(self):
        return DataLoader(
            self.train_set(),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda batch: self.collate_fn(batch),
        )

    def test_loader(self):
        return DataLoader(
            self.test_set(),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda batch: self.collate_fn(batch),
        )

    def collate_fn(self, batch):
        raise NotImplementedError("Base DatasetMaker does not implement collate_fn")


class Dataset(TorchDataset):
    def __init__(self, fonts, test_latincore_chars, test=False):
        self.fonts = fonts
        self.test_latincore_chars = test_latincore_chars
        self.is_test = test
        self.order = []
        for font in self.fonts:
            if self.is_test:
                chars = set(font.codepoints) & set(self.test_latincore_chars)
            else:
                chars = set(font.codepoints) - set(self.test_latincore_chars)
            for char in chars:
                # Skip empty glyphs; they can destabilize training targets.
                hb_font = hb.Font(font.hb_face)  # type: ignore
                gid = hb_font.get_nominal_glyph(char)
                extents = hb_font.get_glyph_extents(gid)
                if all(x for x in extents):
                    self.order.append((font, char))

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        font, char = self.order[idx]
        return {
            "char": char,
            "font": font,
        }
