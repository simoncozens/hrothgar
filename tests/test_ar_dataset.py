import numpy as np

from hrothgar.ar.dataset import ARPhase1DatasetMaker


class DummyFont:
    def __init__(self, codepoints, fill_scale: float = 1000.0, reference=None):
        self.codepoints = set(codepoints)
        self._fill_scale = fill_scale
        self._reference = reference

    def render(self, char: int, size: int = 64):
        value = char / self._fill_scale
        return np.full((3, size, size), value, dtype=np.float32)

    def reference_font(self):
        return self._reference

    def has_codepoint(self, char: int) -> bool:
        return char in self.codepoints

    def has_non_empty_codepoint(self, char: int) -> bool:
        return char in self.codepoints

    def render_char(self, char: int, size: int = 64, axis_position=None):
        _ = axis_position
        return self.render(char, size=size)

    def description_with_tags(self):
        return "dummy description"

    def description_with_tags_and_display(self):
        return "dummy description"


def test_collate_ar_phase1_shapes() -> None:
    maker = ARPhase1DatasetMaker(
        repo_url="tests/dummy_repo", batch_size=2, style_glyph_count=3, image_size=32
    )
    ref_font = DummyFont({65, 66, 67}, fill_scale=10.0)
    style_font = DummyFont({65, 66, 67, 68, 69}, reference=ref_font)
    batch = [
        {"char": 65, "font": style_font},
        {"char": 66, "font": style_font},
    ]

    out = maker.collate_fn(batch)

    assert out["char"].shape == (2,)
    assert out["target_rendering"].shape == (2, 3, 32, 32)
    assert out["content_rendering"].shape == (2, 3, 32, 32)
    assert out["style_renderings"].shape == (2, 3, 3, 32, 32)
    assert out["style_chars"].shape == (2, 3)
    assert len(out["description"]) == 2


def test_collate_ar_phase1_uses_reference_font_for_content() -> None:
    ref_font = DummyFont({65, 66, 67}, fill_scale=10.0)
    style_font = DummyFont({65, 66, 67, 68, 69}, fill_scale=1000.0, reference=ref_font)
    batch = [{"char": 65, "font": style_font}]

    maker = ARPhase1DatasetMaker(
        repo_url="tests/dummy_repo", batch_size=1, style_glyph_count=2
    )
    out = maker.collate_fn(batch)

    target_value = out["target_rendering"][0, 0, 0, 0].item()
    content_value = out["content_rendering"][0, 0, 0, 0].item()
    assert content_value != target_value
    assert abs(content_value - (65 / 10.0)) < 1e-6


def test_collate_ar_phase1_common_style_codepoints() -> None:
    style_font = DummyFont({65, 66, 67, 68, 69})
    batch = [{"char": 65, "font": style_font}]

    maker = ARPhase1DatasetMaker(
        repo_url="tests/dummy_repo",
        batch_size=1,
        style_glyph_count=3,
        common_style_codepoints=[66, 67, 68],
    )
    out = maker.collate_fn(batch)
    sampled = out["style_chars"].tolist()[0]
    assert len(sampled) == 3
    assert set(sampled) == {66, 67, 68}


def test_collate_ar_phase1_common_style_codepoints_do_not_leak() -> None:
    style_font = DummyFont({65, 66, 67, 68, 69, 0x0312})
    batch = [{"char": 65, "font": style_font}]

    maker = ARPhase1DatasetMaker(
        repo_url="tests/dummy_repo",
        batch_size=1,
        style_glyph_count=5,
        common_style_codepoints=[66, 67],
    )
    out = maker.collate_fn(batch)
    sampled = out["style_chars"].tolist()[0]
    assert all(cp in {66, 67} for cp in sampled)
