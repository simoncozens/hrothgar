from __future__ import annotations
import kurbopy
import pathops
import uharfbuzz as hb
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.qu2cuPen import Qu2CuPen
from fontTools.pens.svgPathPen import SVGPathPen, pointToString
from fontTools.ttLib.removeOverlaps import _simplify


class AbsoluteSVGPathPen(SVGPathPen):
    def _lineTo(self, pt):
        x, y = pt
        # duplicate point
        if x == self._lastX and y == self._lastY:
            return
        # write the string
        t = "L" + " " + pointToString(pt, self._ntos)  # type: ignore
        self._lastCommand = "L"
        self._commands.append(t)
        # store for future reference
        self._lastX, self._lastY = pt


class AddExtremaPen(FilterPen):
    def curveTo(self, *points):
        bez = kurbopy.CubicBez(
            kurbopy.Point(*self.current_pt),
            kurbopy.Point(*points[0]),
            kurbopy.Point(*points[1]),
            kurbopy.Point(*points[2]),
        )
        extrema = bez.extrema_ranges()
        # If a range is very small, coalesece it with its neighbor
        for ix, (left, right) in enumerate(extrema):
            left, right = extrema[ix]
            if right - left < 1e-5:
                if ix > 0:
                    # Merge with previous
                    extrema[ix - 1] = (extrema[ix - 1][0], right)
                    extrema[ix] = (right, right)
                else:
                    # Merge with next
                    if ix + 1 < len(extrema):
                        extrema[ix + 1] = (left, extrema[ix + 1][1])
                        extrema[ix] = (left, left)
        # Now remove any zero-length ranges
        extrema = [r for r in extrema if r[1] - r[0] >= 1e-5]
        if len(extrema) == 1:
            # No extrema, just draw the curve as usual
            self._outPen.curveTo(points[0], points[1], points[2])
            self.current_pt = (points[2][0], points[2][1])
            return
        for start_t, end_t in bez.extrema_ranges():
            localbez = bez.subsegment((start_t, end_t))
            self._outPen.curveTo(
                (localbez.p1.x, localbez.p1.y),
                (localbez.p2.x, localbez.p2.y),
                (localbez.p3.x, localbez.p3.y),
            )
            self.current_pt = (localbez.p3.x, localbez.p3.y)


def get_svg_glyph(
    face: hb.Face, glyph_id: int, location: dict | None = None, remove_overlaps: bool = False
) -> list[tuple[str, list[int]]]:
    scale = 1000 / face.upem  # type: ignore
    font = hb.Font(face)  # type: ignore
    svgpen = AbsoluteSVGPathPen({}, ntos=lambda f: str(int(f * scale)))
    pen = AddExtremaPen(svgpen)
    pen = Qu2CuPen(pen, max_err=5, all_cubic=True)
    if location:
        font.set_variations(location)
    path = []

    if remove_overlaps:
        skpath = pathops.Path()
        pathPen = skpath.getPen()
        font.draw_glyph_with_pen(glyph_id, pathPen)
        skpath = _simplify(skpath, chr(glyph_id))
        skpath.draw(pen)
    else:
        font.draw_glyph_with_pen(glyph_id, pen)

    for command in svgpen._commands:
        cmd = command[0] if command[0] != " " else "L"
        coords = [int(p) for p in command[1:].split()]
        path.append((cmd, coords))

    return path

if __name__ == "__main__":
    import sys

    font_path = sys.argv[1]
    char = sys.argv[2]
    face = hb.Face(hb.Blob.from_file_path(font_path))
    font = hb.Font(face)
    glyph_id = font.get_nominal_glyph(ord(char))
    svg_path = get_svg_glyph(face, glyph_id)
    print(svg_path)
