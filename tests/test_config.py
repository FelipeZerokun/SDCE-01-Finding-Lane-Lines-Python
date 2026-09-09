from pathlib import Path

import pytest

from lane_finding.config import ConfigError, load_config

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"


def test_loads_default_configuration() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.blur.kernel_size == 7
    assert config.canny.low_threshold == 50
    assert config.roi.top_left_x == 0.45
    assert config.hough.min_line_length == 35
    assert config.lanes.smoothing_frames == 8
    assert config.render.line_thickness == 12


def test_rejects_even_blur_kernel(tmp_path: Path) -> None:
    contents = DEFAULT_CONFIG.read_text(encoding="utf-8")
    contents = contents.replace("kernel_size = 7", "kernel_size = 6")

    invalid_config = tmp_path / "invalid.toml"
    invalid_config.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match="positive odd integer"):
        load_config(invalid_config)


def test_reports_missing_configuration() -> None:
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config("does-not-exist.toml")


def test_loads_packaged_default_configuration() -> None:
    config = load_config()

    assert config.blur.kernel_size == 7
    assert config.canny.low_threshold == 50
    assert config.lanes.smoothing_frames == 8


def test_packaged_default_matches_editable_example() -> None:
    assert load_config() == load_config(DEFAULT_CONFIG)
