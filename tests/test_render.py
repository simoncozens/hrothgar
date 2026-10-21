from pathlib import Path

import numpy as np
import pytest
import uharfbuzz as hb
from hrothgar.render import (
    _normalize_bitmap,
    _paste_bitmap_onto_canvas,
    denormalize_uniform,
    geometry_tensor,
    place_glyph,
    render,
)


def test_paste_bitmap_aligns_to_baseline() -> None:
    canvas = np.full((10, 10), 255, dtype=np.uint8)
    bitmap = np.array([[255, 255], [255, 255]], dtype=np.uint8)

    # baseline=6 and top=3 => bitmap top-left y is 3.
    _paste_bitmap_onto_canvas(
        canvas=canvas,
        bitmap_array=bitmap,
        bitmap_left=0,
        bitmap_top=3,
        baseline_y=6,
    )

    assert canvas[3, 0] == 0
    assert canvas[4, 1] == 0


def test_paste_bitmap_preserves_positive_left_sidebearing() -> None:
    canvas = np.full((8, 8), 255, dtype=np.uint8)
    bitmap = np.array([[255, 255], [255, 255]], dtype=np.uint8)

    _paste_bitmap_onto_canvas(
        canvas=canvas,
        bitmap_array=bitmap,
        bitmap_left=3,
        bitmap_top=2,
        baseline_y=4,
    )

    assert canvas[2, 2] == 255
    assert canvas[2, 3] == 0
    assert canvas[3, 4] == 0


def test_paste_bitmap_crops_negative_left_sidebearing() -> None:
    canvas = np.full((6, 6), 255, dtype=np.uint8)
    # Three columns: first should be cropped by negative bitmap_left.
    bitmap = np.array([[64, 128, 255]], dtype=np.uint8)

    _paste_bitmap_onto_canvas(
        canvas=canvas,
        bitmap_array=bitmap,
        bitmap_left=-1,
        bitmap_top=1,
        baseline_y=1,
    )

    # The first source column is out-of-bounds, so visible columns are 128, 255.
    assert canvas[0, 0] == 127
    assert canvas[0, 1] == 0
    assert canvas[0, 2] == 255


def test_variable_font_axis_positions_change_rendering() -> None:
    font = _roboto_font()
    gid = hb.Font(font.hb_face).get_nominal_glyph(ord("A"))

    regular, _ = render(font, gid=gid, size=128, axis_position=(100.0, 100.0))
    heavy, _ = render(font, gid=gid, size=128, axis_position=(900.0, 100.0))

    assert regular.shape == heavy.shape
    assert not np.allclose(regular.numpy(), heavy.numpy())


def test_render_phrase_variable_font_axis_changes_output() -> None:
    from hrothgar.render import render_phrase

    font_path = Path("tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf")
    phrase = "THE quick brown fox 1234"

    light = render_phrase(font_path, phrase, size=48, axis_position=(100.0, 100.0))
    heavy = render_phrase(font_path, phrase, size=48, axis_position=(900.0, 100.0))

    assert light.shape == heavy.shape, f"Shape mismatch: {light.shape} vs {heavy.shape}"
    assert light.shape[0] > 0 and light.shape[1] > 0, "Empty output"
    # Different weights should produce measurably different rendering.
    # Heavy weight has more ink (darker pixels).
    light_ink = (light < 255).sum()
    heavy_ink = (heavy < 255).sum()
    assert light_ink > 0, "Light rendering is blank"
    assert (
        heavy_ink > light_ink
    ), f"Heavy ({heavy_ink}) should have more ink than light ({light_ink})"


def test_normalize_bitmap_geometry_in_em_units() -> None:
    # 4 rows x 3 cols coverage; ink occupies rows 1..3 and cols 1..2.
    bitmap = np.zeros((4, 3), dtype=np.uint8)
    bitmap[1:4, 1:3] = 255

    image, geometry = _normalize_bitmap(
        bitmap, bitmap_left=2, bitmap_top=10, advance_px=7, size=10
    )

    assert image.shape == (1, 10, 10)
    assert geometry["scale_x"] == (2 - 1 + 1) / 10  # 0.2
    assert geometry["scale_y"] == (3 - 1 + 1) / 10  # 0.3
    assert geometry["left_sidebearing"] == (2 + 1) / 10  # 0.3
    # descender_depth = scale_y - baseline_offset = 0.3 - 0.9
    assert geometry["descender_depth"] == (3 - 1 + 1) / 10 - (10 - 1) / 10
    assert geometry["advance"] == 7 / 10  # 0.7


def test_normalize_bitmap_blank_glyph_preserves_advance() -> None:
    image, geometry = _normalize_bitmap(
        np.zeros((0, 0), dtype=np.uint8),
        bitmap_left=0,
        bitmap_top=0,
        advance_px=8,
        size=10,
    )

    assert image.shape == (1, 10, 10)
    assert float(image.min()) == 1.0 and float(image.max()) == 1.0
    assert geometry["scale_x"] == 0.0
    assert geometry["scale_y"] == 0.0
    assert geometry["left_sidebearing"] == 0.0
    assert geometry["descender_depth"] == 0.0
    assert geometry["advance"] == 0.8


def test_geometry_tensor_packs_canonical_order() -> None:
    import pytest

    geometry = {
        "advance": 0.7,
        "scale_x": 0.2,
        "descender_depth": 0.9,
        "left_sidebearing": 0.3,
        "scale_y": 0.3,
    }
    packed = geometry_tensor(geometry)
    assert packed.tolist() == pytest.approx([0.2, 0.3, 0.3, 0.9, 0.7])


def test_place_glyph_denormalizes_onto_baseline() -> None:
    # A fully-inked 32x32 square, geometry: 0.5em x 0.5em ink, no LSB, no
    # descender (ink bottom exactly on the baseline), 0.6em advance.  At 128 ppm
    # the ink should occupy a 64x64 block starting at x=64 (origin) and
    # y=128 (baseline - ascent).
    image = np.zeros((32, 32), dtype=np.float32)
    canvas, origin_x, baseline_y, advance_x = place_glyph(
        image, [0.5, 0.5, 0.0, 0.0, 0.6]
    )

    assert canvas.shape == (256, 384)  # 2em tall, 3em wide
    assert origin_x == 64
    assert baseline_y == 192
    assert advance_x == 64 + round(0.6 * 128)

    # Ink is present exactly in the placed block; margins stay white.
    assert canvas[128:192, 64:128].min() == 0.0
    assert canvas[:128, :].max() == 1.0
    assert canvas[192:, :].max() == 1.0
    assert canvas[:, :64].max() == 1.0


def test_place_glyph_blank_glyph_preserves_advance() -> None:
    image = np.ones((32, 32), dtype=np.float32)
    canvas, origin_x, _baseline_y, advance_x = place_glyph(
        image, [0.0, 0.0, 0.0, 0.0, 0.3]
    )

    assert canvas.max() == 1.0  # stays white
    assert advance_x == origin_x + round(0.3 * 128)


def _roboto_font():
    from hrothgar.googlefonts import Font

    font = Font()
    font.path = Path("tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf")
    font.hb_face = hb.Face(hb.Blob.from_file_path(str(font.path)))
    return font


def test_render_uniform_square_image_and_geometry() -> None:
    import torch

    font = _roboto_font()
    gid = hb.Font(font.hb_face).get_nominal_glyph(ord("g"))

    image, geometry = render(font, gid=gid, size=128)

    assert image.shape == (128, 128)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert float(image.min()) < 1.0  # some ink
    assert set(geometry) == {
        "scale_x",
        "scale_y",
        "left_sidebearing",
        "descender_depth",
        "advance",
    }


def test_render_normalized_returns_crop_to_ink_square() -> None:
    from hrothgar.render import NORMALIZED

    font = _roboto_font()
    image, geometry = render(font, codepoint=ord("A"), size=128, frame=NORMALIZED)

    assert image.shape == (128, 128)
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert float(image.min()) < 1.0  # some ink
    assert set(geometry) == {
        "scale_x",
        "scale_y",
        "left_sidebearing",
        "descender_depth",
        "advance",
    }


def test_render_requires_exactly_one_of_gid_or_codepoint() -> None:
    font = _roboto_font()
    with pytest.raises(ValueError):
        render(font, size=128)
    with pytest.raises(ValueError):
        render(font, gid=1, codepoint=ord("A"), size=128)


def test_render_rejects_unknown_frame() -> None:
    with pytest.raises(TypeError):
        render(_roboto_font(), codepoint=ord("A"), size=128, frame="square")


def test_render_uniform_preserves_descender() -> None:
    font = _roboto_font()
    hbfont = hb.Font(font.hb_face)
    g_img, _ = render(font, gid=hbfont.get_nominal_glyph(ord("g")), size=128)
    a_img, _ = render(font, gid=hbfont.get_nominal_glyph(ord("A")), size=128)

    baseline_row = int(round(0.75 * 128))
    g_below = int((g_img[baseline_row:, :] < 1.0).sum())
    a_below = int((a_img[baseline_row:, :] < 1.0).sum())
    assert g_below > 0  # 'g' has a descender below the baseline
    assert a_below < g_below  # 'A' sits above the baseline


def test_denormalize_uniform_places_onto_square() -> None:
    # A fully-inked 32x32 crop-to-ink square with 0.5em x 0.5em ink, no LSB,
    # no descender.  At size=128 (1em == 64px), the ink should be 32x32 px
    # starting at x=32 (origin) and y=64 (baseline - ascent).
    image = np.zeros((32, 32), dtype=np.float32)
    canvas = denormalize_uniform(image, [0.5, 0.5, 0.0, 0.0, 0.6], size=128)

    assert canvas.shape == (128, 128)
    assert float(canvas[64:96, 32:64].min()) == 0.0
    assert float(canvas[:64, :].max()) == 1.0
    assert float(canvas[96:, :].max()) == 1.0
    assert float(canvas[:, :32].max()) == 1.0
    assert float(canvas[:, 64:].max()) == 1.0


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


def test_frame_defaults_are_the_legacy_uniform_framing() -> None:
    from hrothgar.render import UNIFORM_FRAME, Frame

    assert UNIFORM_FRAME == Frame(-0.5, 1.5, -0.5, 1.5)
    assert (UNIFORM_FRAME.width, UNIFORM_FRAME.height) == (2.0, 2.0)
    assert UNIFORM_FRAME.em_span == 2.0


def test_frame_rejects_inverted_edges() -> None:
    from hrothgar.render import Frame

    with pytest.raises(ValueError):
        Frame(left=1.0, right=0.0)
    with pytest.raises(ValueError):
        Frame(bottom=1.0, top=0.0)


def test_frame_from_extents_grows_by_margin_and_floors_min_size() -> None:
    from hrothgar.render import Frame

    frame = Frame.from_extents(-0.1, -0.2, 0.7, 0.8)  # 0.8 x 1.0 em
    assert (frame.left, frame.right) == pytest.approx((-0.1, 0.7))
    assert (frame.bottom, frame.top) == pytest.approx((-0.2, 0.8))

    # A margin grows each dimension about the centre by that fraction.
    grown = Frame.from_extents(-0.1, -0.2, 0.7, 0.8, margin=0.25)
    assert (grown.width, grown.height) == pytest.approx((1.0, 1.25))
    assert (grown.left, grown.right) == pytest.approx((-0.2, 0.8))
    assert (grown.bottom, grown.top) == pytest.approx((-0.325, 0.925))

    # A degenerate box is floored, so the renderer never divides by a tiny span.
    dot = Frame.from_extents(0.3, 0.4, 0.3, 0.4, min_size=0.2)
    assert (dot.width, dot.height) == pytest.approx((0.2, 0.2))
    assert (dot.left, dot.top) == pytest.approx((0.2, 0.5))


def _ink_height(image) -> int:
    return int((image < 0.5).any(dim=1).sum())


def _ink_width(image) -> int:
    return int((image < 0.5).any(dim=0).sum())


def _ink_centre_row(image) -> float:
    rows = (image < 0.5).any(dim=1).nonzero().flatten().float()
    return float(rows.mean())


def test_tighter_frame_makes_the_glyph_proportionally_bigger() -> None:
    from hrothgar.render import Frame

    font = _roboto_font()
    default, geometry = render(font, codepoint=ord("x"), size=128)
    # A frame half the linear size covers half the em, so the glyph doubles.
    tight, tight_geometry = render(
        font, codepoint=ord("x"), size=128, frame=Frame(-0.25, 0.75, -0.25, 0.75)
    )

    # The geometry labels are em units and do not see the frame, so the ink in
    # pixels is exactly ``scale_y`` at the frame's own px/em.
    assert _ink_height(default) == pytest.approx(geometry["scale_y"] * 64, abs=2)
    assert _ink_height(tight) == pytest.approx(geometry["scale_y"] * 128, abs=2)
    assert tight_geometry["scale_y"] == pytest.approx(geometry["scale_y"], abs=0.02)


def test_non_square_frame_keeps_aspect_ratio_and_centres_the_glyph() -> None:
    from hrothgar.render import Frame

    font = _roboto_font()
    default, _ = render(font, codepoint=ord("o"), size=128)
    # 2.0 em wide but only 1.0 em tall: same em_span, so the same scale, but the
    # frame now sits centred in the canvas and the glyph moves with it.
    wide, _ = render(
        font, codepoint=ord("o"), size=128, frame=Frame(-0.5, 1.5, -0.25, 0.75)
    )

    assert _ink_height(wide) == _ink_height(default)  # not stretched vertically
    assert _ink_width(wide) == _ink_width(default)  # nor horizontally
    # The frame's centre moved 0.25em down (-0.25 of the 2em canvas) ...
    assert _ink_centre_row(wide) - _ink_centre_row(default) == pytest.approx(-16, abs=2)


def test_envelope_frame_contains_the_glyphs_it_was_measured_from() -> None:
    from hrothgar.render import frame_for_codepoints

    font = _roboto_font()
    codepoints = [ord(c) for c in "AEGOFCSafgo"]
    frame = frame_for_codepoints(font, codepoints)
    assert 0.0 < frame.em_span < 2.0  # tighter than the legacy window

    for codepoint in codepoints:
        image, _ = render(font, codepoint=codepoint, size=128, frame=frame)
        ink = image < 0.5
        assert ink.any()
        assert not (
            ink[0].any() or ink[-1].any() or ink[:, 0].any() or ink[:, -1].any()
        ), f"{chr(codepoint)} touches the canvas edge in its own envelope frame"


def test_frame_for_codepoints_falls_back_when_nothing_has_ink() -> None:
    from hrothgar.render import UNIFORM_FRAME, frame_for_codepoints

    assert frame_for_codepoints(_roboto_font(), [0x20, 0x0A]) == UNIFORM_FRAME


def test_denormalize_uniform_honours_a_tighter_frame() -> None:
    from hrothgar.render import Frame, denormalize_uniform

    # 0.5em of ink at 1em == 128px is 64px, and the 1.0em-span frame puts the
    # baseline at row 96 with the origin at column 32.
    image = np.zeros((32, 32), dtype=np.float32)
    canvas = denormalize_uniform(
        image,
        [0.5, 0.5, 0.0, 0.0, 0.6],
        size=128,
        frame=Frame(-0.25, 0.75, -0.25, 0.75),
    )

    assert canvas.shape == (128, 128)
    assert float(canvas[32:96, 32:96].min()) == 0.0
    assert float(canvas[:32, :].max()) == 1.0
    assert float(canvas[96:, :].max()) == 1.0
    assert float(canvas[:, :32].max()) == 1.0
