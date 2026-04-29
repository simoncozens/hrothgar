from pathlib import Path

from hrothgar.googlefonts import Font


def test_sample_axis_positions_non_variable_font() -> None:
    font_path = Path("tests/dummy_repo/ofl/abeezee/ABeeZee-Italic.ttf")
    font = Font()
    font.path = font_path
    import uharfbuzz as hb

    font.hb_face = hb.Face(hb.Blob.from_file_path(str(font_path)))

    positions = font.sample_axis_positions(splits=3)
    assert positions == [[]]


def test_sample_axis_positions_variable_font() -> None:
    font_path = Path("tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf")
    font = Font()
    font.path = font_path
    import uharfbuzz as hb

    font.hb_face = hb.Face(hb.Blob.from_file_path(str(font_path)))

    positions = font.sample_axis_positions(splits=3)
    assert positions[0] == []
    # Two axes, 3-way split on each => 9 sampled positions + default []
    assert len(positions) == 10
    assert any(len(position) == 2 for position in positions[1:])
