from functools import cached_property
import io
import subprocess
from pathlib import Path
from typing import Dict, Optional, Self, Set, Union

import numpy as np
import uharfbuzz as hb
from gftools.util.google_fonts import Metadata
from PIL import Image, ImageChops


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
            # Skip noto
            if font_path.parts[-2].startswith("noto"):
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


class GoogleFont:
    """A single font in the Google Fonts repository."""

    def __init__(self, path: str | Path, gf: GoogleFonts | None = None):
        self.path = Path(path)
        self.metadata_pb = self.path.parent / "METADATA.pb"
        # May raise exception, caller should catch
        self.metadata = Metadata(self.metadata_pb)
        self.family = self.metadata.name
        self.hb_face = hb.Face(hb.Blob.from_file_path(self.path))
        self.gf = gf

    @cached_property
    def codepoints(self):
        return set(self.hb_face.unicodes)

    def tags(self) -> Dict[str, float]:
        """Returns the tags for this font, as a dictionary of tag name to value. The values are centiles from 0 to 100."""
        return self.gf.tags.get(self.family, {}) if self.gf else {}

    def render(self, char: int, size: int = 64) -> np.ndarray:
        """Renders the given character as a square image of the requested size. The character should be given as a Unicode code point (i.e. ord("a") for "a")."""
        try:
            return render(self.path, chr(char), size, self.hb_face, do_trim=False)
        except Exception as e:
            return np.zeros((3, size, size), dtype=np.float32)

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

    def has_codepoint(self, char: int) -> bool:
        """Returns whether this font has a glyph for the given character."""
        return char in self.codepoints


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


# Glyph rendering utils
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im


def _render(vars_text, font, text):
    return subprocess.run(
        [
            "hb-view",
            "-o",
            "-",
            "-O",
            "png",
            vars_text,
            "--font-size=1024",
            font,
            text,
        ],
        check=True,
        capture_output=True,
    ).stdout


def render(
    font,
    text: str,
    size: int,
    hb_face: hb.Face,
    variation: Dict[str, float] = {},
    do_trim=False,
):
    if not variation:
        vars_text = "--variations=wght=400"
    else:
        vars_text = "--variations=" + ",".join(
            [f"{k}={v}" for k, v in variation.items()]
        )
    image = _render(vars_text, font, text)
    image = Image.open(io.BytesIO(image))
    width, height = image.size
    if do_trim:
        image = trim(image)
        width, height = image.size
        scale = min(size / width, size / height)
        new_img = image.resize((int(width * scale), int(height * scale)))
        width, height = new_img.size
    else:
        new_img = image

    new_img2 = Image.new("L", (size, size))
    # White background
    new_img2.paste(255, (0, 0, size, size))
    if do_trim:
        new_img2.paste(
            new_img,
            (int((size - width) / 2), int((size - height) / 2)),
        )
    else:
        # Paste it at known coordinates. This is the bit that nobody
        # thinks about. The X coordinate is easy, we'll put it at 0.
        # The Y coordinate is a bit more tricky, because we want the
        # glyph to be aligned on the baseline, regardless of the font's
        # vertical metrics.
        upem = hb_face.upem
        ascent = hb.Font(hb_face).get_font_extents("ltr").ascender
        # We've received a glyph at upem scale, which means that the height of
        # the glyph image is ascent - descent. We want to scale and position the
        # glyph such that (a) all glyphs are scaled by the same amount, regardless
        # of their vertical metrics, and (b) the baseline is three quarters of the way
        # up the image. When we scale it let's assume that all glyphs fit inside
        # 1.5x the upem. So we have one upem worth of ascent above the baseline
        # and 0.5 upem worth of descent below the baseline.
        scale = size / (1.5 * upem)
        if int(new_img.width * scale) == 0 or int(new_img.height * scale) == 0:
            # Return zeros
            return np.zeros((3, size, size), dtype=np.float32)
        new_img = new_img.resize(
            (int(new_img.width * scale), int(new_img.height * scale))
        )
        scaled_ascent = ascent * scale
        baseline_y = int(size * 0.66)
        new_img2.paste(new_img, (0, int(baseline_y - scaled_ascent)))
    new_img2 = np.asarray(new_img2, dtype=np.float32)
    # (H, W) -> (3, H, W)
    new_img2 = np.stack([new_img2] * 3, axis=0)
    return new_img2 / 255.0
