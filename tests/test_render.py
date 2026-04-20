import numpy as np

from hrothgar.render import _paste_bitmap_onto_canvas


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
