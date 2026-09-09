from __future__ import annotations

import math
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from lane_finding.config import HoughConfig, LaneFindingConfig, RoiConfig

Image = NDArray[np.uint8]
LineSegments = NDArray[np.int32]


def _validate_rgb_image(image: Image) -> None:
    if image.dtype != np.uint8:
        raise TypeError("Image must use uint8 pixels")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have shape (height, width, 3)")


def grayscale(image: Image) -> Image:
    """Convert an RGB image to grayscale."""
    _validate_rgb_image(image)
    return cast(Image, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))


def gaussian_blur(image: Image, kernel_size: int) -> Image:
    """Apply Gaussian smoothing to a grayscale image."""
    return cast(Image, cv2.GaussianBlur(image, (kernel_size, kernel_size), 0))


def canny_edges(
    image: Image,
    low_threshold: int,
    high_threshold: int,
) -> Image:
    """Find image edges using the Canny detector."""
    return cast(Image, cv2.Canny(image, low_threshold, high_threshold))


def roi_vertices(
    image_shape: tuple[int, ...],
    config: RoiConfig,
) -> NDArray[np.int32]:
    """Convert relative ROI coordinates into pixel coordinates."""
    height, width = image_shape[:2]
    maximum_x = width - 1
    maximum_y = height - 1

    return np.array(
        [
            [
                round(maximum_x * config.bottom_left_x),
                round(maximum_y * config.bottom_y),
            ],
            [
                round(maximum_x * config.top_left_x),
                round(maximum_y * config.top_y),
            ],
            [
                round(maximum_x * config.top_right_x),
                round(maximum_y * config.top_y),
            ],
            [
                round(maximum_x * config.bottom_right_x),
                round(maximum_y * config.bottom_y),
            ],
        ],
        dtype=np.int32,
    )


def region_of_interest(image: Image, vertices: NDArray[np.int32]) -> Image:
    """Keep only pixels inside the supplied polygon."""
    mask = np.zeros_like(image)
    fill_color: int | tuple[int, ...] = 255

    if image.ndim == 3:
        fill_color = (255,) * image.shape[2]

    cv2.fillPoly(mask, [vertices], fill_color)
    return cast(Image, cv2.bitwise_and(image, mask))


def hough_segments(
    edges: Image,
    config: HoughConfig,
) -> LineSegments:
    """Detect line segments, returning an empty array when none exist."""
    detected = cv2.HoughLinesP(
        edges,
        rho=config.rho,
        theta=math.radians(config.theta_degrees),
        threshold=config.threshold,
        minLineLength=config.min_line_length,
        maxLineGap=config.max_line_gap,
    )

    if detected is None:
        return np.empty((0, 4), dtype=np.int32)

    return np.asarray(detected, dtype=np.int32).reshape(-1, 4)


def detect_line_segments(
    image: Image,
    config: LaneFindingConfig,
) -> LineSegments:
    """Run the complete stateless preprocessing and Hough pipeline."""
    gray = grayscale(image)
    blurred = gaussian_blur(gray, config.blur.kernel_size)
    edges = canny_edges(
        blurred,
        config.canny.low_threshold,
        config.canny.high_threshold,
    )
    vertices = roi_vertices(image.shape, config.roi)
    masked_edges = region_of_interest(edges, vertices)
    return hough_segments(masked_edges, config.hough)
