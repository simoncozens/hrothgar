from typing import List, Tuple

from .representations import MAX_COORDINATE
from .hyperparameters import GEN_IMAGE_SIZE


def to_image_space(pts):
    """
    Transforms a coordinate pair from font coordinate space to image space.
    """
    x, y = pts
    x /= MAX_COORDINATE
    y /= MAX_COORDINATE

    img_size = GEN_IMAGE_SIZE[0]
    baseline = 0.33 * img_size

    # Add 100 units to X to stop italics from being cut off
    x += 100.0 / MAX_COORDINATE

    # Apply scaling and baseline translation
    # This seems to scale the glyph to fit roughly in the top 2/3 of the image
    transformed_x = x * img_size * 2.0 / 3.0
    transformed_y = y * img_size * 2.0 / 3.0
    # Add the baseline to the Y coordinate.
    transformed_y += baseline
    # Apply vertical flip for image coordinates (Y-down)
    transformed_y = img_size - transformed_y
    return transformed_x, transformed_y


def get_bounds(points: List[Tuple[float, float]]) -> "ImageSpaceBbox":
    """
    Computes the bounding box of the points in image space.
    Returns [x_min, y_min, x_max, y_max]
    """
    if not points:
        return ImageSpaceBbox([0.0, 0.0, 0.0, 0.0])

    x_min = min(points, key=lambda p: p[0])[0]
    x_max = max(points, key=lambda p: p[0])[0]
    y_min = min(points, key=lambda p: p[1])[1]
    y_max = max(points, key=lambda p: p[1])[1]

    return ImageSpaceBbox([x_min, y_min, x_max, y_max])


class ImageSpaceBbox(list):
    pass
