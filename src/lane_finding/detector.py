from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from lane_finding.config import LaneFindingConfig, LanesConfig, RenderConfig
from lane_finding.pipeline import Image, LineSegments, detect_line_segments

Segment = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class LaneLine:
    bottom_x: int
    bottom_y: int
    top_x: int
    top_y: int


@dataclass(frozen=True, slots=True)
class LanePair:
    left: LaneLine | None
    right: LaneLine | None


def _partition_segments(
    segments: LineSegments,
    width: int,
    config: LanesConfig,
) -> tuple[list[Segment], list[Segment]]:
    left: list[Segment] = []
    right: list[Segment] = []
    center_x = width / 2

    for segment in segments:
        x1 = int(segment[0])
        y1 = int(segment[1])
        x2 = int(segment[2])
        y2 = int(segment[3])

        delta_x = x2 - x1
        delta_y = y2 - y1

        if delta_x == 0:
            continue

        slope = delta_y / delta_x
        absolute_slope = abs(slope)

        if not (
            config.minimum_absolute_slope
            <= absolute_slope
            <= config.maximum_absolute_slope
        ):
            continue

        midpoint_x = (x1 + x2) / 2
        candidate = (x1, y1, x2, y2)

        if slope < 0 and midpoint_x < center_x:
            left.append(candidate)
        elif slope > 0 and midpoint_x > center_x:
            right.append(candidate)

    return left, right


def _fit_line(
    segments: list[Segment],
    width: int,
    height: int,
    top_y_ratio: float,
) -> LaneLine | None:
    if not segments:
        return None

    x_values: list[float] = []
    y_values: list[float] = []
    weights: list[float] = []

    for x1, y1, x2, y2 in segments:
        length = math.hypot(x2 - x1, y2 - y1)

        x_values.extend((float(x1), float(x2)))
        y_values.extend((float(y1), float(y2)))
        weights.extend((length, length))

    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    weight_array = np.sqrt(np.asarray(weights, dtype=np.float64))

    design = np.column_stack((y_array, np.ones_like(y_array)))
    weighted_design = design * weight_array[:, np.newaxis]
    weighted_x = x_array * weight_array

    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design,
        weighted_x,
        rcond=None,
    )

    if int(rank) < 2:
        return None

    x_per_y = float(coefficients[0])
    intercept = float(coefficients[1])

    bottom_y = height - 1
    top_y = round((height - 1) * top_y_ratio)

    def calculate_x(y: int) -> int:
        calculated = round(x_per_y * y + intercept)
        return max(0, min(width - 1, calculated))

    return LaneLine(
        bottom_x=calculate_x(bottom_y),
        bottom_y=bottom_y,
        top_x=calculate_x(top_y),
        top_y=top_y,
    )


def fit_lane_lines(
    segments: LineSegments,
    image_shape: tuple[int, ...],
    config: LanesConfig,
) -> LanePair:
    """Fit one continuous line to each side of the lane."""
    height, width = image_shape[:2]
    left_segments, right_segments = _partition_segments(
        segments,
        width,
        config,
    )

    return LanePair(
        left=_fit_line(left_segments, width, height, config.top_y),
        right=_fit_line(right_segments, width, height, config.top_y),
    )


def _average_lines(lines: deque[LaneLine | None]) -> LaneLine | None:
    available = [line for line in lines if line is not None]

    if not available:
        return None

    return LaneLine(
        bottom_x=round(sum(line.bottom_x for line in available) / len(available)),
        bottom_y=round(sum(line.bottom_y for line in available) / len(available)),
        top_x=round(sum(line.top_x for line in available) / len(available)),
        top_y=round(sum(line.top_y for line in available) / len(available)),
    )


def draw_lane_lines(
    image: Image,
    lanes: LanePair,
    config: RenderConfig,
) -> Image:
    """Draw detected lane lines over an RGB image."""
    if lanes.left is None and lanes.right is None:
        return image.copy()

    overlay = np.zeros_like(image)

    for line in (lanes.left, lanes.right):
        if line is None:
            continue

        cv2.line(
            overlay,
            (line.bottom_x, line.bottom_y),
            (line.top_x, line.top_y),
            color=(255, 0, 0),
            thickness=config.line_thickness,
        )

    return cast(
        Image,
        cv2.addWeighted(
            image,
            config.alpha,
            overlay,
            config.line_weight,
            0,
        ),
    )


class LaneDetector:
    """Detect and smooth lane lines across successive video frames."""

    def __init__(self, config: LaneFindingConfig) -> None:
        self._config = config
        history_size = config.lanes.smoothing_frames
        self._left_history: deque[LaneLine | None] = deque(maxlen=history_size)
        self._right_history: deque[LaneLine | None] = deque(maxlen=history_size)

    def reset(self) -> None:
        self._left_history.clear()
        self._right_history.clear()

    def detect_lanes(
        self,
        segments: LineSegments,
        image_shape: tuple[int, ...],
    ) -> LanePair:
        current = fit_lane_lines(
            segments,
            image_shape,
            self._config.lanes,
        )

        self._left_history.append(current.left)
        self._right_history.append(current.right)

        return LanePair(
            left=_average_lines(self._left_history),
            right=_average_lines(self._right_history),
        )

    def process_frame(self, image: Image) -> Image:
        segments = detect_line_segments(image, self._config)
        lanes = self.detect_lanes(segments, image.shape)
        return draw_lane_lines(image, lanes, self._config.render)
