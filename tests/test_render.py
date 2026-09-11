import numpy as np
import uharfbuzz as hb
from pathlib import Path

from hrothgar.glyph_rendering import geometry_tensor, normalize_bitmap, place_glyph
from hrothgar.render import _paste_bitmap_onto_canvas, render_gid


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
    font_path = Path("tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf")
    face = hb.Face(hb.Blob.from_file_path(str(font_path)))
    gid = hb.Font(face).get_nominal_glyph(ord("A"))

    regular = render_gid(font_path, gid, size=128, axis_position=(100.0, 100.0))
    heavy = render_gid(font_path, gid, size=128, axis_position=(900.0, 100.0))

    assert regular.shape == heavy.shape
    assert not np.allclose(regular, heavy)


def test_render_phrase_variable_font_axis_changes_output() -> None:
    from hrothgar.render import render_phrase

    font_path = Path("tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf")
    phrase = "THE quick brown fox 1234"

    light = render_phrase(font_path, phrase, size=48, axis_position=(100.0, 100.0))
    heavy = render_phrase(font_path, phrase, size=48, axis_position=(900.0, 100.0))

    assert light.shape == heavy.shape, (
        f"Shape mismatch: {light.shape} vs {heavy.shape}"
    )
    assert light.shape[0] > 0 and light.shape[1] > 0, "Empty output"
    # Different weights should produce measurably different rendering.
    # Heavy weight has more ink (darker pixels).
    light_ink = (light < 255).sum()
    heavy_ink = (heavy < 255).sum()
    assert light_ink > 0, "Light rendering is blank"
    assert heavy_ink > light_ink, (
        f"Heavy ({heavy_ink}) should have more ink than light ({light_ink})"
    )


def test_normalize_bitmap_geometry_in_em_units() -> None:
    # 4 rows x 3 cols coverage; ink occupies rows 1..3 and cols 1..2.
    bitmap = np.zeros((4, 3), dtype=np.uint8)
    bitmap[1:4, 1:3] = 255

    image, geometry = normalize_bitmap(
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
    image, geometry = normalize_bitmap(
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
    assert advance_x == 64 + int(round(0.6 * 128))

    # Ink is present exactly in the placed block; margins stay white.
    assert canvas[128:192, 64:128].min() == 0.0
    assert canvas[:128, :].max() == 1.0
    assert canvas[192:, :].max() == 1.0
    assert canvas[:, :64].max() == 1.0


def test_place_glyph_blank_glyph_preserves_advance() -> None:
    image = np.ones((32, 32), dtype=np.float32)
    canvas, origin_x, baseline_y, advance_x = place_glyph(
        image, [0.0, 0.0, 0.0, 0.0, 0.3]
    )

    assert canvas.max() == 1.0  # stays white
    assert advance_x == origin_x + int(round(0.3 * 128))
