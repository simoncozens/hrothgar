from typing import Optional, Set

import torch
import uharfbuzz as hb
from glyphsets import GlyphSet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.googlefonts import GoogleFonts

LATIN_CORE = GlyphSet("GF_Latin_Core").get_characters()


class DatasetMaker:
    """A class for creating training and test datasets for font generation. It uses the GoogleFonts class to load font data from the specified repository URL, and splits the fonts into training and test sets. It also defines a set of test characters (the Latin Core codepoints) that will be used for evaluation."""

    def __init__(
        self,
        repo_url: str,
        batch_size: int,
        having: Optional[Set[int]] = None,
        canary_size: Optional[int] = None,
        image_size: int = 128,
    ):
        """
        Class for creating training and test datasets for font generation. It uses the GoogleFonts class to load font data from the specified repository URL, and splits the fonts into training and test sets. It also defines a set of test characters (the Latin Core codepoints) that will be used for evaluation.

        Args:
            repo_url (str): The URL of the Google Fonts repository to load fonts from.
            batch_size (int): The batch size to use for the data loaders.
            having (Optional[Set[int]]): An optional set of Unicode codepoints to filter the
                fonts by. Only fonts that have at least one of these codepoints will be included.
            canary_size (Optional[int]): An optional integer specifying a smaller number of fonts to use for testing purposes. If None, all fonts will be used.
        """
        self.googlefonts = GoogleFonts(repo_url, having=having)
        self.batch_size = batch_size
        self.image_size = image_size

        # Test chars is a random 10% of the GF Latin Core codepoints
        # (because these are the codepoints which all fonts ought to have)
        _, self.test_latincore_chars = train_test_split(LATIN_CORE)
        # Train chars is everything that's not in test chars!
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
            collate_fn=lambda batch: collate_fn(batch, image_size=self.image_size),
        )

    def test_loader(self):
        return DataLoader(
            self.test_set(),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda batch: collate_fn(batch, image_size=self.image_size),
        )


def collate_fn(batch, image_size):
    chars = torch.tensor([item["char"] for item in batch])
    renderings = torch.stack(
        [
            torch.tensor(item["font"].render(item["char"], size=image_size))
            for item in batch
        ]
    )
    descriptions = [item["font"].description_with_tags() for item in batch]

    return {
        "char": chars,
        "rendering": renderings,
        "description": descriptions,
    }


class Dataset(TorchDataset):
    def __init__(self, fonts, test_latincore_chars, test=False):
        self.fonts = fonts
        self.test_latincore_chars = test_latincore_chars
        self.is_test = test
        # Calculate the length and order once here
        self.order = []
        for font in self.fonts:
            if self.is_test:
                # Only count the test Latin Core chars that this font has
                chars = set(font.codepoints) & set(self.test_latincore_chars)
            else:
                # Count all chars that are not in the test Latin Core chars
                chars = set(font.codepoints) - set(self.test_latincore_chars)
            for char in chars:
                # Make sure we have glyph extents, i.e. no empty glyphs, because those would cause problems for the model
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
            "font": font,  # Do the processing in the collate_fn
        }
