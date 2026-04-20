from functools import cached_property
from pathlib import Path
from typing import Dict, Optional, Self, Set, Union
from hrothgar.render import render_gid

import numpy as np
import uharfbuzz as hb
from gftools.util.google_fonts import Metadata


class Font:
    """A font, whether standalone or from the Google Fonts repository. This is an abstract base class that defines the interface for fonts, and provides some common functionality. The concrete implementations are GoogleFont and StandaloneFont."""

    hb_face: hb.Face
    path: Path

    def render(self, char: int, size: int = 64) -> np.ndarray:
        """Render a single glyph as a (3, size, size) float32 array."""
        try:
            gid = hb.Font(self.hb_face).get_nominal_glyph(char)
            return self.render_gid(gid, size)
        except Exception:
            return np.ones((3, size, size), dtype=np.float32)

    def render_gid(self, gid: int, size: int = 64) -> np.ndarray:
        """Render a single glyph by GID as a (3, size, size) float32 array."""
        try:
            return render_gid(self.path, gid, size)
        except Exception:
            return np.ones((3, size, size), dtype=np.float32)

    @cached_property
    def codepoints(self) -> Set[int]:
        """Unicode codepoints present in this font."""
        return set(self.hb_face.unicodes)

    def has_codepoint(self, char: int) -> bool:
        """Returns whether this font has a glyph for the given character."""
        return char in self.codepoints

    def description(self) -> str:
        """Empty description — no metadata available for standalone fonts."""
        return ""

    def tags(self) -> Dict[str, float]:
        """Empty tags — no metadata available for standalone fonts."""
        return {}


class GoogleFonts:
    """A class for interacting with the Google Fonts repository.
    It loads all the fonts and their metadata, and provides methods for accessing them.
    This is the bottom of the data collection stack - it does not make any assumptions about
    character set, test/train split, etc. It just provides access to the raw data.
    The higher level classes will build on top of this to provide more specific functionality.
    """

    tags = {}
    families_by_name = {}

    def __init__(self, repo: str | Path, having: Optional[Set[int]] = None):
        self.repo_path = Path(repo)
        self.fonts = []
        for font_path in self.repo_path.glob("ofl/*/*.ttf"):
            try:
                font = GoogleFont(font_path, self)
                GoogleFonts.families_by_name[font.family] = font
            except Exception as _e:
                # print(f"Error loading font {font_path}: {e}")
                continue
            if self.should_skip(font):
                continue
            if having is not None:
                if any(not font.has_codepoint(cp) for cp in having):
                    continue
            self.fonts.append(font)
        self._init_tags()

    def _init_tags(self):
        tag_csv = self.repo_path / "tags" / "all" / "families.csv"
        with tag_csv.open() as f:
            for line in f:
                family, _variation, tag, value = line.strip().split(",")
                self.tags[family] = self.tags.get(family, {})
                self.tags[family][tag] = float(value)

    def should_skip(self, font: GoogleFont) -> bool:
        if font.path.parts[-2].startswith("noto"):
            return True
        # Skip CJK; we don't want these big fonts with lots of characters
        # and a different glyph construction style to dominate
        has_cjk = [
            "chinese-hongkong",
            "chinese-simplified",
            "chinese-traditional",
            "japanese",
            "korean",
        ]
        subsets = font.metadata.subsets
        if any(subset in subsets for subset in has_cjk):
            return True
        return False


class GoogleFont(Font):
    """A single font in the Google Fonts repository."""

    def __init__(self, path: str | Path, gf: GoogleFonts | None = None):
        self.path = Path(path)
        self.metadata_pb = self.path.parent / "METADATA.pb"
        # May raise exception, caller should catch
        self.metadata = Metadata(self.metadata_pb)
        self.family = self.metadata.name
        self.hb_face = hb.Face(hb.Blob.from_file_path(self.path))
        self.gf = gf

    def tags(self) -> Dict[str, float]:
        """Returns the tags for this font, as a dictionary of tag name to value. The values are centiles from 0 to 100."""
        return self.gf.tags.get(self.family, {}) if self.gf else {}

    def description(self) -> str:
        """Returns the description of this font, as a string. This is taken from the article if it exists, otherwise from the description, otherwise it's empty."""
        if (self.path.parent / "article" / "ARTICLE.en_us.html").exists():
            with (self.path.parent / "article" / "ARTICLE.en_us.html").open() as f:
                return dehtml(f.read())
        if (self.path.parent / "DESCRIPTION.en_us.html").exists():
            with (self.path.parent / "DESCRIPTION.en_us.html").open() as f:
                return dehtml(f.read())
        return ""

    def description_with_tags(self) -> str:
        """Returns the description of this font, including the tags. This is the same as description(), but with the tags appended to the end."""
        desc = self.description()

        def stringify_tag(tag, value):
            if tag.startswith("/Expressive/"):
                tag_name = tag[len("/Expressive/") :]
                return f"{centile_to_text(value)} {tag_name.lower()}"
            elif tag.startswith("/Sans/"):
                tag_name = tag[len("/Sans/") :]
                return f"a {centile_to_text(value)} {tag_name} sans-serif font"
            elif tag.startswith("/Serif/"):
                tag_name = tag[len("/Serif/") :]
                return f"a {centile_to_text(value)} {tag_name} serif font"
            return f" {centile_to_text(value)} {tag}"

        tag_descriptions = ", ".join(
            stringify_tag(tag, value)
            for tag, value in self.tags().items()
            if not tag.startswith("/Quality/")
        )
        if tag_descriptions:
            desc += " It is considered "
            desc += tag_descriptions
        return desc

    def reference_font(self) -> Union[Self, None]:
        """Returns a reference font for this font, based on its stroke tags. This is used to provide a baseline for comparison when describing the font."""
        if self.metadata.stroke == "SANS_SERIF":
            return GoogleFonts.families_by_name.get("Noto Sans")
        elif self.metadata.stroke == "SERIF":
            return GoogleFonts.families_by_name.get("Noto Serif")
        else:
            return GoogleFonts.families_by_name.get("Noto Sans")


class StandaloneFont(Font):
    """A font loaded from a local file path without Google Fonts repository structure.

    This is used for NFA (Novel Font Adaptation) when adapting to fonts that are
    not part of the Google Fonts repository, or when the repo is unavailable.

    It exposes the same interface as ``GoogleFont`` (``codepoints``, ``render``,
    ``reference_font``, ``has_codepoint``, ``hb_face``) so it can be used
    as a drop-in replacement in dataset collation code.
    """

    def __init__(
        self,
        path: Union[str, Path],
        reference: Optional["StandaloneFont | GoogleFont"] = None,
    ) -> None:
        self.path = Path(path)
        self.hb_face = hb.Face(hb.Blob.from_file_path(str(self.path)))
        self.family = self.path.stem
        self._reference = reference

    def reference_font(self) -> Optional["StandaloneFont | GoogleFont"]:
        """Return the reference (content) font, or None if not set."""
        return self._reference


def centile_to_text(score: int) -> str:
    if score < 20:
        return "not at all"
    elif score < 40:
        return "not very"
    elif score < 60:
        return "quite"
    elif score < 80:
        return "somewhat"
    else:
        return "very"


def dehtml(html: str) -> str:
    # This is a very naive implementation, but it should be good enough for our purposes.
    # It just removes all tags and replaces multiple whitespace with a single space.
    text = ""
    in_tag = False
    for c in html:
        if c == "<":
            in_tag = True
        elif c == ">":
            in_tag = False
        elif not in_tag:
            text += c
    return " ".join(text.split())
