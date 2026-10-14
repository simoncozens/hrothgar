"""Build the Tier 0 residual dataset and a self-contained SVG viewer.

For each variable font we fit the two-parameter bolding model leave-one-out
(fit on every glyph except the one shown, so the residual is a fair
held-out residual), then write out, per node:

* the regular / predicted-bold / true-bold positions,
* the local frame (normal + tangent), width, corner flag,
* the true and predicted offsets and the residual (normal + tangential).

Two artefacts are produced in ``outputs/tier0/``:

* ``tier0_dataset.json`` -- the full per-node dataset (also the supervision a
  Tier 1 point-transformer would train on), and
* ``tier0_viewer.html`` -- a standalone viewer that overlays the regular,
  predicted-bold and true-bold outlines plus the residual vectors.

Run with::

    PYTHONPATH=Lib python scripts/build_tier0_dataset.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import uharfbuzz as hb

from hrothgar.weight_adjust.outlines import (
    Glyph,
    extract_glyph,
    is_corner,
    node_frame,
    stroke_widths,
    turn_angle,
)
from hrothgar.weight_adjust.tier0 import BoldParams, apply, fit, predict_offset

# (label, font path, regular wght, bold wght)
FONTS = [
    ("roboto", "tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf", 400.0, 700.0),
    ("playfair", "tests/dummy_repo/ofl/playfairdisplay/PlayfairDisplay[wght].ttf", 400.0, 700.0),
    ("shantell", "tests/dummy_repo/ofl/shantellsans/ShantellSans[BNCE,INFM,SPAC,wght].ttf", 300.0, 700.0),
]

GLYPHS = "HOSDBogaeemnlpuy"
OUT_DIR = Path("outputs/tier0")


def _svg_d(glyph: Glyph) -> str:
    return " ".join(c.to_svg() for c in glyph.contours)


def _glyph_bbox(glyph: Glyph) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for c in glyph.contours:
        r = c.bounding_box()
        xs += [r.min_x(), r.max_x()]
        ys += [r.min_y(), r.max_y()]
    return [min(xs), min(ys), max(xs), max(ys)]


def _union_bbox(bboxes: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def _nodes(regular: Glyph, bold: Glyph, params: BoldParams) -> tuple[list[dict], float]:
    """Per-contour node records plus the glyph's rms residual magnitude."""
    widths = stroke_widths(regular)
    contours: list[dict] = []
    ss = 0.0
    n = 0
    for ci, (cr, cb) in enumerate(zip(regular.contours, bold.contours)):
        segs_r = list(cr.segments())
        segs_b = list(cb.segments())
        nodes: list[dict] = []
        for i, (sr, sb) in enumerate(zip(segs_r, segs_b)):
            width = widths[ci][i]
            tangent, normal = node_frame(cr, i)
            dx = sb.p0.x - sr.p0.x
            dy = sb.p0.y - sr.p0.y
            o_true = dx * normal.x + dy * normal.y
            t_true = dx * tangent.x + dy * tangent.y
            o_pred = predict_offset(params, normal.x, normal.y, width)
            px = sr.p0.x + o_pred * normal.x
            py = sr.p0.y + o_pred * normal.y
            bx = sb.p0.x
            by = sb.p0.y
            rx = bx - px
            ry = by - py
            ss += rx * rx + ry * ry
            n += 1
            nodes.append(
                {
                    "x": sr.p0.x,
                    "y": sr.p0.y,
                    "px": px,
                    "py": py,
                    "bx": bx,
                    "by": by,
                    "nx": normal.x,
                    "ny": normal.y,
                    "width": width,
                    "is_corner": is_corner(cr, i),
                    "turn_angle": turn_angle(cr, i),
                    "offset_true": o_true,
                    "offset_pred": o_pred,
                    "tangential_true": t_true,
                    "normal_resid": o_true - o_pred,
                    "tangential_resid": t_true,
                }
            )
        contours.append({"outer": cr.area() < 0.0, "area": cr.area(), "nodes": nodes})
    rms = math.sqrt(ss / n) if n else 0.0
    return contours, rms


def _load_pairs(
    face: hb.Face, regular: float, bold: float
) -> list[tuple[str, int, Glyph, Glyph]]:
    font = hb.Font(face)
    pairs: list[tuple[str, int, Glyph, Glyph]] = []
    for ch in GLYPHS:
        gid = font.get_nominal_glyph(ord(ch))
        if gid == 0:
            continue
        reg = extract_glyph(face, gid, {"wght": regular})
        bld = extract_glyph(face, gid, {"wght": bold})
        if len(reg.contours) != len(bld.contours):
            continue
        if any(
            len(list(a.segments())) != len(list(b.segments()))
            for a, b in zip(reg.contours, bld.contours)
        ):
            continue
        pairs.append((ch, gid, reg, bld))
    return pairs


def build() -> dict:
    data = {"fonts": []}
    for label, path, regular_w, bold_w in FONTS:
        face = hb.Face(hb.Blob.from_file_path(path))
        pairs = _load_pairs(face, regular_w, bold_w)
        if len(pairs) < 2:
            continue

        font_params = fit([(r, b) for _, _, r, b in pairs])
        glyphs = []
        for idx, (ch, gid, reg, bold) in enumerate(pairs):
            fit_set = [(r, b) for j, (_, _, r, b) in enumerate(pairs) if j != idx]
            params = fit(fit_set)
            pred = apply(reg, params)
            contours, rms = _nodes(reg, bold, params)
            bbox = _union_bbox([_glyph_bbox(reg), _glyph_bbox(pred), _glyph_bbox(bold)])
            glyphs.append(
                {
                    "char": ch,
                    "gid": gid,
                    "params": {"vertical": params.vertical, "horizontal": params.horizontal},
                    "rms_residual": rms,
                    "regular": _svg_d(reg),
                    "predicted": _svg_d(pred),
                    "bold": _svg_d(bold),
                    "bbox": bbox,
                    "contours": contours,
                }
            )
        data["fonts"].append(
            {
                "label": label,
                "font_path": path,
                "regular_wght": regular_w,
                "bold_wght": bold_w,
                "font_params": {
                    "vertical": font_params.vertical,
                    "horizontal": font_params.horizontal,
                },
                "glyphs": glyphs,
            }
        )
    return data


# ---------------------------------------------------------------------------
# Viewer template (a plain string; ``__DATA_JSON__`` is replaced with the data).
# ---------------------------------------------------------------------------

_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Tier 0 bolding residuals</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem; background: #fafafa; color: #222; }
  h1 { font-size: 1.4rem; margin: 0 0 0.5rem; }
  h2 { font-size: 1.1rem; margin: 2rem 0 0.5rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
  .legend { display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.85rem; margin: 0.5rem 0 1rem; color: #444; }
  .legend span { display: inline-flex; align-items: center; gap: 0.35rem; }
  .swatch { display: inline-block; width: 1.2em; height: 1.2em; border-radius: 2px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 0.75rem; }
  .card { background: #fff; border: 1px solid #e2e2e2; border-radius: 6px; padding: 0.5rem; }
  .card .title { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.15rem; }
  .card .caption { font-size: 0.75rem; color: #777; margin-bottom: 0.3rem; }
  .card svg { width: 100%; height: auto; display: block; background: #fff; }
</style>
</head>
<body>
<h1>Tier 0 bolding residuals</h1>
<div class="legend">
  <span><span class="swatch" style="background:#bbb"></span> regular</span>
  <span><span class="swatch" style="background:rgba(42,107,255,0.6)"></span> predicted bold</span>
  <span><span class="swatch" style="background:rgba(0,160,0,0.6)"></span> actual bold</span>
  <span><span class="swatch" style="background:#ff5500"></span> residual @ corner</span>
  <span><span class="swatch" style="background:#00aaaa"></span> residual @ smooth</span>
</div>
<div id="container"></div>

<script>
const DATA = __DATA_JSON__;
const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function renderGlyph(g) {
  const [x0, y0, x1, y1] = g.bbox;
  const span = Math.max(x1 - x0, y1 - y0);
  const pad = span * 0.06;
  const vb = [x0 - pad, y0 - pad, (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad].join(" ");
  const flip = `matrix(1 0 0 -1 0 ${y0 + y1})`;
  const dot = span / 160;

  let inner = `<g transform="${flip}">`;
  inner += `<path d="${g.regular}" fill="#bdbdbd" fill-opacity="0.4"/>`;
  inner += `<path d="${g.predicted}" fill="rgba(42,107,255,0.10)" stroke="#2a6bff" stroke-width="1.4" fill-rule="evenodd" vector-effect="non-scaling-stroke"/>`;
  inner += `<path d="${g.bold}" fill="rgba(0,160,0,0.10)" stroke="#00a000" stroke-width="1.4" fill-rule="evenodd" vector-effect="non-scaling-stroke"/>`;
  for (const c of g.contours) {
    for (const n of c.nodes) {
      const col = n.is_corner ? "#ff5500" : "#00aaaa";
      inner += `<line x1="${n.px}" y1="${n.py}" x2="${n.bx}" y2="${n.by}" stroke="${col}" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
      inner += `<circle cx="${n.bx}" cy="${n.by}" r="${dot}" fill="${col}"/>`;
    }
  }
  inner += `</g>`;

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", vb);
  svg.innerHTML = inner;

  const card = el("div", "card");
  card.appendChild(el("div", "title", g.char));
  card.appendChild(el("div", "caption", `rms resid ${g.rms_residual.toFixed(1)}  (v=${g.params.vertical.toFixed(3)}, h=${g.params.horizontal.toFixed(3)})`));
  card.appendChild(svg);
  return card;
}

function renderFont(font) {
  const sec = el("div");
  const v = font.font_params.vertical, h = font.font_params.horizontal;
  const contrast = h ? (v / h) : Infinity;
  sec.appendChild(el("h2", "", `${font.label}  —  contrast ${contrast.toFixed(2)}:1  (v=${v.toFixed(3)}, h=${h.toFixed(3)})`));
  const grid = el("div", "grid");
  for (const g of font.glyphs) grid.appendChild(renderGlyph(g));
  sec.appendChild(grid);
  return sec;
}

const container = document.getElementById("container");
for (const font of DATA.fonts) container.appendChild(renderFont(font));
</script>
</body>
</html>
"""


def main() -> None:
    data = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUT_DIR / "tier0_dataset.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    html_path = OUT_DIR / "tier0_viewer.html"
    html = _TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    html_path.write_text(html, encoding="utf-8")

    n_nodes = sum(
        sum(len(c["nodes"]) for c in g["contours"])
        for f in data["fonts"]
        for g in f["glyphs"]
    )
    print(f"wrote {json_path} ({json_path.stat().st_size} bytes)")
    print(f"wrote {html_path} ({html_path.stat().st_size} bytes)")
    print(f"{len(data['fonts'])} fonts, {sum(len(f['glyphs']) for f in data['fonts'])} glyphs, {n_nodes} nodes")


if __name__ == "__main__":
    main()
