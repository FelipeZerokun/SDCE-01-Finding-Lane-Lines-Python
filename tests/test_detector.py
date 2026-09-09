from pathlib import Path

import numpy as np

from lane_finding.config import load_config
from lane_finding.detector import LaneDetector, fit_lane_lines
from lane_finding.pipeline import LineSegments

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "default.toml")

SEGMENTS = np.array(
    [
        [20, 99, 80, 60],
        [30, 99, 85, 65],
        [120, 60, 180, 99],
        [115, 65, 170, 99],
    ],
    dtype=np.int32,
)

EMPTY_SEGMENTS = np.empty((0, 4), dtype=np.int32)


def test_fits_left_and_right_lanes() -> None:
    lanes = fit_lane_lines(SEGMENTS, (100, 200, 3), CONFIG.lanes)

    assert lanes.left is not None
    assert lanes.right is not None

    assert lanes.left.bottom_x < lanes.left.top_x < 100
    assert 100 < lanes.right.top_x < lanes.right.bottom_x


def test_ignores_vertical_and_horizontal_segments() -> None:
    segments: LineSegments = np.array(
        [
            [50, 90, 50, 50],
            [20, 80, 80, 80],
        ],
        dtype=np.int32,
    )

    lanes = fit_lane_lines(segments, (100, 200, 3), CONFIG.lanes)

    assert lanes.left is None
    assert lanes.right is None


def test_empty_segments_produce_no_lanes() -> None:
    lanes = fit_lane_lines(
        EMPTY_SEGMENTS,
        (100, 200, 3),
        CONFIG.lanes,
    )

    assert lanes.left is None
    assert lanes.right is None


def test_detector_instances_do_not_share_history() -> None:
    first = LaneDetector(CONFIG)
    second = LaneDetector(CONFIG)

    first_result = first.detect_lanes(SEGMENTS, (100, 200, 3))
    second_result = second.detect_lanes(EMPTY_SEGMENTS, (100, 200, 3))

    assert first_result.left is not None
    assert first_result.right is not None
    assert second_result.left is None
    assert second_result.right is None


def test_missing_lanes_expire_from_history() -> None:
    detector = LaneDetector(CONFIG)
    detector.detect_lanes(SEGMENTS, (100, 200, 3))

    result = None
    for _ in range(CONFIG.lanes.smoothing_frames):
        result = detector.detect_lanes(
            EMPTY_SEGMENTS,
            (100, 200, 3),
        )

    assert result is not None
    assert result.left is None
    assert result.right is None


def test_reset_clears_history() -> None:
    detector = LaneDetector(CONFIG)
    detector.detect_lanes(SEGMENTS, (100, 200, 3))

    detector.reset()
    result = detector.detect_lanes(EMPTY_SEGMENTS, (100, 200, 3))

    assert result.left is None
    assert result.right is None


def test_blank_frame_preserves_shape_and_dtype() -> None:
    detector = LaneDetector(CONFIG)
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    result = detector.process_frame(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result, image)
