"""Safe image decoding and normalization for camera uploads."""
from __future__ import annotations

import io
from typing import Final

import numpy as np
from PIL import Image, ImageOps

try:  # pillow-heif is optional in minimal test environments.
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - exercised by deployment packaging
    pass


MAX_IMAGE_PIXELS: Final = 80_000_000
MAX_IMAGE_DIMENSION: Final = 6000


class InvalidImageError(ValueError):
    """Raised when an upload cannot be safely decoded as a still image."""


def decode_camera_image(
    content: bytes,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> np.ndarray:
    """Decode a still image, apply EXIF orientation, and return a BGR array.

    iPhone images may store camera orientation only in EXIF and may be much larger
    than inference needs. Orientation is applied before the bounded resize.
    """
    if not content:
        raise InvalidImageError("Image file is empty.")

    try:
        with Image.open(io.BytesIO(content)) as source:
            if getattr(source, "n_frames", 1) > 1:
                source.seek(0)
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise InvalidImageError(
                    f"Image dimensions {width}x{height} exceed the {max_pixels:,}-pixel limit."
                )

            oriented = ImageOps.exif_transpose(source)
            oriented.load()
            rgb = oriented.convert("RGB")

        longest = max(rgb.size)
        if longest > max_dimension:
            scale = max_dimension / float(longest)
            resized = (
                max(1, round(rgb.width * scale)),
                max(1, round(rgb.height * scale)),
            )
            rgb = rgb.resize(resized, Image.Resampling.LANCZOS)

        rgb_array = np.asarray(rgb, dtype=np.uint8)
        return np.ascontiguousarray(rgb_array[:, :, ::-1])
    except InvalidImageError:
        raise
    except Exception as exc:
        raise InvalidImageError(f"Invalid or unsupported image file: {exc}") from exc
