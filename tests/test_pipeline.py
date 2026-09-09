from pathlib import Path

import numpy as np
import pytest

from lane_finding.config import load_config
from lane_finding.pipeline import (
    detect_line_segments,
    grayscale,
    region_of_interest,
    roi_vertices,
)

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "default.toml")


def test_grayscale_preserves_dimensions_and_dtype() -> None:
    image = np.zeros((60, 100, 3), dtype=np.uint8)

    result = grayscale(image)

    assert result.shape == (60, 100)
    assert result.dtype == np.uint8


def test_grayscale_rejects_non_rgb_image() -> None:
    image = np.zeros((60, 100), dtype=np.uint8)

    with pytest.raises(ValueError, match="height, width, 3"):
        grayscale(image)


def test_grayscale_rejects_non_uint8_image() -> None:
    image = np.zeros((60, 100, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        grayscale(image)  # type: ignore[arg-type]


def test_roi_vertices_are_inside_image() -> None:
    vertices = roi_vertices((100, 200, 3), CONFIG.roi)

    assert vertices.shape == (4, 2)
    assert np.all(vertices[:, 0] >= 0)
    assert np.all(vertices[:, 0] < 200)
    assert np.all(vertices[:, 1] >= 0)
    assert np.all(vertices[:, 1] < 100)


def test_region_of_interest_masks_outside_pixels() -> None:
    image = np.full((100, 200), 255, dtype=np.uint8)
    vertices = roi_vertices(image.shape, CONFIG.roi)

    result = region_of_interest(image, vertices)

    assert result[0, 0] == 0
    assert np.count_nonzero(result) > 0
    assert np.count_nonzero(result) < image.size


def test_blank_image_produces_no_segments() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    segments = detect_line_segments(image, CONFIG)

    assert segments.shape == (0, 4)
    assert segments.dtype == np.int32
