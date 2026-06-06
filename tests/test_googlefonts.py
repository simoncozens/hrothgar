import os
import hrothgar.googlefonts as googlefonts

if "GOOGLE_FONTS_REPO" not in os.environ:
    raise ValueError("GOOGLE_FONTS_REPO environment variable not set, cannot run tests")
REPOSITORY_PATH = os.getenv("GOOGLE_FONTS_REPO")


def test_googlefonts():
    gf = googlefonts.GoogleFonts(REPOSITORY_PATH)
    assert len(gf.fonts) > 0
    for font in gf.fonts:
        assert font.family in gf.families_by_name


def test_londrina_outline():
    gf = googlefonts.GoogleFonts(REPOSITORY_PATH)
    londrina = gf.families_by_name["Londrina Outline"]
    assert "Sao Paulo" in londrina.description()
    assert "not at all sincere" in londrina.description_with_tags()
    tags = londrina.tags()
    assert tags["/Expressive/Sincere"] == 5
    assert tags["/Expressive/Playful"] > 40
    assert londrina.has_codepoint(0x20AC)  # Has euro
    assert not londrina.has_codepoint(0x20B9)  # No rupee
