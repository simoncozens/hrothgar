from pathlib import Path
from typing import Dict

import pathops
import kurbopy
import uharfbuzz as hb
from fontTools.pens.qu2cuPen import Qu2CuPen
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.svgPathPen import SVGPathPen, pointToString
from fontTools.ttLib.removeOverlaps import _simplify

from hrothgar.vectorization.coordinate import to_image_space
from hrothgar.vectorization.svgglyph import SVGGlyph

from hrothgar.vectorization.representations.svgcommand import SVGCommand

# No point cacheing as we are storing the PNGs in our dataset
CACHING = False


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


cache_dir = Path("imgcache")


class Glyph:
    """A glyph defined by a font file, glyph ID, and location in design space.

    We don't store any vector representation here; that is generated on demand.
    We are simply representing the concept of a source glyph, with the ability
    to extract its vector representation or rasterized image.

    See also SVGGlyph for a vector representation, and NodeGlyph for a
    "designer-like" representation we will use for our model.
    """

    font_file: Path
    glyph_id: int
    location: Dict[str, float]
    name: str

    def __repr__(self) -> str:
        return f"Glyph(font_file={self.font_file}, glyph_id={self.name}, location={self.location})"

    def __init__(self, glyph_id: int, face, location: Dict[str, float]):
        # self.font_file = font_file
        self.glyph_id = glyph_id
        self.location = location
        self.face = face  # type: ignore
        font = hb.Font(face)  # type: ignore
        self.name = font.get_glyph_name(self.glyph_id)  # type: ignore

    # def rasterize(self, size: int = RASTER_IMG_SIZE) -> npt.NDArray[np.float64]:
    #     font_base = str(self.font_file).replace(BASE_DIR + "/", "").replace("/", "-")
    #     key = "-".join(
    #         [
    #             str(self.glyph_id),
    #             ",".join(
    #                 {f"{k}:{self.location[k]}" for k in sorted(self.location.keys())}
    #             ),
    #             str(size),
    #         ]
    #     )
    #     if CACHING and (cache_dir / font_base / (key + ".png")).exists():
    #         img = Image.open(cache_dir / font_base / (key + ".png")).convert("L")
    #         img = np.asarray(img, dtype=np.float64) / 255.0
    #         img = np.expand_dims(img, axis=-1)
    #     else:
    #         img = self._rasterize(size)
    #         if CACHING:
    #             pil_img = Image.fromarray(
    #                 (img.squeeze(-1) * 255).astype(np.uint8), mode="L"
    #             )
    #             (cache_dir / font_base).mkdir(exist_ok=True)
    #             print("Saving", font_base, key)
    #             pil_img.save(cache_dir / font_base / (key + ".png"))
    #     return img

    # def _rasterize(self, size: int) -> npt.NDArray[np.float64]:
    #     node_glyph = self.vectorize().to_node_glyph()
    #     contour_sequences = node_glyph.encode(SVGCommand)

    #     if contour_sequences is None:
    #         return np.zeros((size, size, 1), dtype=np.float64)

    #     contour_tensors = []
    #     for encoded_contour in contour_sequences:
    #         encoded_tensor = torch.from_numpy(encoded_contour).float()
    #         cmds_tensor, coords_tensor = SVGCommand.split_tensor(encoded_tensor)

    #         contour_tensors.append((cmds_tensor, coords_tensor))

    #     image_tensor = rasterize_batch(
    #         [contour_tensors],
    #         SVGCommand,
    #         seed=42,
    #         img_size=size,
    #         requires_grad=False,
    #         device=torch.device("cpu"),
    #     )

    #     numpy_image = image_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #     return np.expand_dims(numpy_image, axis=-1).astype(np.float64)

    def vectorize(self, remove_overlaps: bool = True) -> SVGGlyph:
        scale = 1000 / self.face.upem  # type: ignore
        font = hb.Font(self.face)  # type: ignore
        svgpen = AbsoluteSVGPathPen({}, ntos=lambda f: str(int(f * scale)))
        pen = AddExtremaPen(svgpen)
        pen = Qu2CuPen(pen, max_err=5, all_cubic=True)
        if self.location:
            font.set_variations(self.location)
        path = []
        if self.glyph_id is None:
            return SVGGlyph([])

        if remove_overlaps:
            skpath = pathops.Path()
            pathPen = skpath.getPen()
            font.draw_glyph_with_pen(self.glyph_id, pathPen)
            skpath = _simplify(skpath, chr(self.glyph_id))
            skpath.draw(pen)
        else:
            font.draw_glyph_with_pen(self.glyph_id, pen)

        for command in svgpen._commands:
            cmd = command[0] if command[0] != " " else "L"
            coords = [int(p) for p in command[1:].split()]

            image_space_coords = []
            for x, y in zip(coords[0::2], coords[1::2]):
                ix, iy = to_image_space((x, y))
                image_space_coords.extend([ix, iy])
            path.append(SVGCommand(cmd, image_space_coords))
        return SVGGlyph(path, "%s, %s" % (self.font_file, chr(self.glyph_id)))
