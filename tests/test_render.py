import numpy as np
import uharfbuzz as hb
from pathlib import Path

from hrothgar.render import _fit_bitmap_to_canvas, _paste_bitmap_onto_canvas, render_gid


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


def test_fit_bitmap_to_canvas_scales_and_centers_with_border() -> None:
    bitmap = np.full((20, 10), 255, dtype=np.uint8)

    image = _fit_bitmap_to_canvas(bitmap, size=12, trim_to_rsb=False)

    assert image.shape == (12, 12)
    assert np.all(image[0, :] == 255)
    assert np.all(image[-1, :] == 255)
    assert np.all(image[:, 0] == 255)
    assert np.all(image[:, -1] == 255)
    ink_rows, ink_cols = np.where(image < 255)
    assert ink_rows.min() == 1
    assert ink_rows.max() == 10
    assert ink_cols.min() >= 1
    assert ink_cols.max() <= 10
    assert abs((ink_cols.min() - 1) - (10 - ink_cols.max())) <= 1


def test_fit_bitmap_to_canvas_rejects_upscaling() -> None:
    bitmap = np.full((4, 2), 255, dtype=np.uint8)

    try:
        _fit_bitmap_to_canvas(bitmap, size=12, trim_to_rsb=False, allow_upscale=False)
    except ValueError as exc:
        assert "upscaling" in str(exc)
    else:
        raise AssertionError("expected upscaling guard to fail")


def test_variable_font_axis_positions_change_rendering() -> None:
    font_path = Path("tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf")
    face = hb.Face(hb.Blob.from_file_path(str(font_path)))
    gid = hb.Font(face).get_nominal_glyph(ord("A"))

    regular = render_gid(font_path, gid, size=128, axis_position=(100.0, 100.0))
    heavy = render_gid(font_path, gid, size=128, axis_position=(900.0, 100.0))

    assert regular.shape == heavy.shape
    assert not np.allclose(regular, heavy)
