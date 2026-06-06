from functools import cached_property
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Self, Set, Union

from hrothgar.render import render_gid

import numpy as np
import uharfbuzz as hb
from gftools.util.google_fonts import Metadata
from fontTools.ttLib import TTFont


class Font:
    """A font, whether standalone or from the Google Fonts repository. This is an abstract base class that defines the interface for fonts, and provides some common functionality. The concrete implementations are GoogleFont and StandaloneFont."""

    hb_face: hb.Face
    path: Path

    def render(
        self, char: int, size: int = 64, axis_position: Optional[List[float]] = None
    ) -> np.ndarray:
        """Render a single glyph as a (3, size, size) float32 array."""
        try:
            gid = hb.Font(self.hb_face).get_nominal_glyph(char)
            return self.render_gid(gid, size, axis_position=axis_position)
        except Exception:
            return np.ones((3, size, size), dtype=np.float32)

    def render_gid(
        self, gid: int, size: int = 64, axis_position: Optional[List[float]] = None
    ) -> np.ndarray:
        """Render a single glyph by GID as a (3, size, size) float32 array."""
        try:
            axis_tuple = tuple(axis_position) if axis_position is not None else None
            return render_gid(self.path, gid, size, axis_position=axis_tuple)
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

    def classification(self) -> str:
        """Coarse style classification used for bucketed training metrics."""
        return "UNKNOWN"

    def sample_axis_positions(self, splits: int = 5) -> List[List[float]]:
        """Sample axis positions for this font. If the font has no variable axes, returns a list of lists, with each internal list being the user-space location on the axes ordered by their order in the fvar table."""
        if "fvar" not in self.hb_face.table_tags:
            return [[]]
        # Slow path
        ttfont = TTFont(self.path)
        axes = {
            ix: np.linspace(axis.minValue, axis.maxValue, splits).tolist()
            for ix, axis in enumerate(ttfont["fvar"].axes[0:5])
            # Use first five axes to stop things like Amstelvar dominating the dataset
        }
        # Take Cartesian product, convert each set of coordinates to list in order
        tags = axes.keys()
        instances = itertools.product(*axes.values())
        instances = [[instance[ix] for ix in tags] for instance in instances]
        return [[]] + instances


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
        for font_path in sorted(self.repo_path.glob("ofl/*/*.ttf")):
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

    def should_skip(self, font: "GoogleFont") -> bool:
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

    def display_score(self) -> float:
        """Compute the display-ness score for this font (0-100).

        Returns a numeric score indicating how display-like the font is,
        based on its tag composition. Higher values indicate more display-oriented
        characteristics (ornaments, effects, distinctive styling).
        """
        return compute_display_score(self.tags())

    def description_with_tags_and_display(self) -> str:
        """Returns the description with tags and explicit display conditioning.

        This version prepends a natural language description of the font's
        display characteristics before the other tags. Useful for conditioning
        the model to understand display vs. text fonts.
        """
        display = self.display_score()
        display_descriptor = f"This is a {centile_to_text(int(display))} display font. "
        return display_descriptor + self.description_with_tags()

    def reference_font(self) -> Union[Self, None]:
        """Returns a reference font for this font, based on its stroke tags. This is used to provide a baseline for comparison when describing the font."""
        if (
            self.metadata.stroke == "SANS_SERIF"
            or self.metadata.category == "SANS_SERIF"
        ):
            return GoogleFonts.families_by_name.get("Noto Sans")
        elif self.metadata.stroke == "SERIF" or self.metadata.category == "SERIF":
            return GoogleFonts.families_by_name.get("Noto Serif")
        else:
            return GoogleFonts.families_by_name.get("Noto Sans")

    def classification(self) -> str:
        c = self.metadata.classifications or self.metadata.category
        if isinstance(c, str):
            return c
        if c is None:
            return "UNKNOWN"
        return "/".join(c)


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


def find_google_font_by_basename(
    dataset_path: Union[str, Path],
    font_path: Union[str, Path],
) -> GoogleFont:
    """Find exactly one ``GoogleFont`` by matching font filename basename.

    Args:
        google_fonts: Loaded Google Fonts repository object.
        font_path: Font path whose basename will be matched.

    Returns:
        The uniquely matched ``GoogleFont``.

    Raises:
        ValueError: If no matches are found or multiple matches exist.
    """
    google_fonts = GoogleFonts(dataset_path)
    basename = Path(font_path).name
    matches = [font for font in google_fonts.fonts if font.path.name == basename]
    if not matches:
        raise ValueError(
            f"Could not find a Google Font whose basename matches {basename}."
        )
    if len(matches) > 1:
        families = sorted({font.family for font in matches})
        raise ValueError(
            "Multiple Google Fonts match basename "
            f"{basename}: {families}. Please provide a uniquely identifying file."
        )
    return matches[0]


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


def compute_display_score(tags: Dict[str, float]) -> float:
    """Derive a display-ness score (0-100) from a font's tag composition.

    Display fonts are characterized by distinctive visual features like ornaments,
    effects, and non-standard construction. This function identifies tags that
    correlate with display typography and combines them into a single signal.

    Args:
        tags: Dictionary of tag names to centile values (0-100).

    Returns:
        A composite display score in the range [0, 100].
    """
    # Tags strongly indicating display characteristics
    display_positive = {
        "/Theme/Stencil": 2.0,
        "/Theme/Inline": 2.0,
        "/Theme/Pixel": 2.0,
        "/Theme/Blackletter": 2.0,
        "/Serif/Fat Face": 1.8,
        "/Theme/Tuscan": 1.8,
        "/Theme/Woodtype": 1.8,
        "/Theme/Distressed": 1.4,
        "/Theme/Art Deco": 1.6,
        "/Theme/Medieval": 1.6,
        "/Theme/Shaded": 1.6,
        "/Expressive/Innovative": 1.2,
        "/Theme/Brush": 1.2,
        "/Expressive/Futuristic": 1.4,
        "/Sans/Glyphic": 1.4,
        "/Expressive/Excited": 1.0,
        "/Sans/Superellipse": 1.2,
        "/Script/Formal": 1.0,
        "/Script/Handwritten": 0.8,
        "/Slab/Clarendon": 0.6,
    }

    # Tags indicating text/body-oriented fonts (reduce display score)
    display_negative = {
        "/Purpose/Easy Reading": -2.0,
        "/Expressive/Calm": -1.2,
        "/Expressive/Competent": -0.8,
        "/Expressive/Sincere": -0.6,
        "/Sans/Humanist": -0.8,
        "/Serif/Old Style Garalde": -1.0,
        "/Serif/Transitional": -1.0,
        "/Serif/Modern": -0.8,
        "/Slab/Humanist": -0.8,
    }

    # Compute direct weighted scores
    positive_score = 0.0
    positive_count = 0
    for tag, weight in display_positive.items():
        if tag in tags:
            positive_score += (tags[tag] / 100.0) * weight
            positive_count += 1

    negative_score = 0.0
    negative_count = 0
    for tag, weight in display_negative.items():
        if tag in tags:
            negative_score += (tags[tag] / 100.0) * weight
            negative_count += 1

    # Compute average contribution per present tag
    # This avoids penalizing fonts that simply don't have many tags
    positive_avg = positive_score / positive_count if positive_count > 0 else 0.0
    negative_avg = abs(negative_score) / negative_count if negative_count > 0 else 0.0

    # Combine: positive pushes toward 100, negative pushes toward 0
    # If no tags: score = 50 (neutral)
    display_score = 50.0 + 50.0 * (positive_avg - negative_avg)

    return max(0.0, min(100.0, display_score))


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
