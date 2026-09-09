from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np

from lane_finding.pipeline import Image


def load_image(path: str | Path) -> Image:
    """Load an image as an RGB uint8 array."""
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    bgr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise ValueError(f"Unable to decode image: {image_path}")

    return cast(
        Image,
        cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB),
    )


def save_image(
    path: str | Path,
    image: Image,
    *,
    overwrite: bool = False,
) -> None:
    """Save an RGB image, optionally refusing to overwrite an existing file."""
    output_path = Path(path)

    if image.dtype != np.uint8:
        raise TypeError("Image must use uint8 pixels")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have shape (height, width, 3)")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    bgr_image = cast(
        Image,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
    )

    try:
        written = cv2.imwrite(str(output_path), bgr_image)
    except cv2.error as error:
        raise OSError(f"Unable to write image: {output_path}") from error

    if not written:
        raise OSError(f"Unable to write image: {output_path}")
