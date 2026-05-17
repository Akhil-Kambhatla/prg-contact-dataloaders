"""Polygon rasterization helpers."""

from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def polygons_to_mask(
    polygons: List[Sequence[Sequence[float]]],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Rasterize a list of polygons into a binary mask.

    Args:
        polygons: list of polygons. Each polygon is a list of [x, y] points.
            Coordinates are in the same space as `image_size`.
        image_size: (height, width) of the output mask.

    Returns:
        A uint8 numpy array of shape (height, width) with values 0 or 255.
        Pixels inside any polygon are 255.
    """
    height, width = image_size
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        flat = [(float(x), float(y)) for x, y in polygon]
        draw.polygon(flat, outline=255, fill=255)
    return np.array(img, dtype=np.uint8)
