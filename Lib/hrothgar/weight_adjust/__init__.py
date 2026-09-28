"""Weight-adjustment model: regular raster -> bold raster in a shared frame.

A deterministic, exemplar-conditioned CNN.  Unlike the generator, it never
crop-to-ink normalizes its raster — the stroke-thickness signal that *is* the
weight axis must survive — so it renders both regular and bold into the same
origin-tracked frame and predicts only the advance width (the one metric that
never appears in ink).
"""
