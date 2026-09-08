from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when lane-finding configuration is invalid."""


def _require_between(name: str, value: float, minimum: float, maximum: float) -> None:
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")


@dataclass(frozen=True, slots=True)
class BlurConfig:
    kernel_size: int

    def __post_init__(self) -> None:
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ConfigError("blur.kernel_size must be a positive odd integer")


@dataclass(frozen=True, slots=True)
class CannyConfig:
    low_threshold: int
    high_threshold: int

    def __post_init__(self) -> None:
        if not 0 <= self.low_threshold < self.high_threshold <= 255:
            raise ConfigError(
                "Canny thresholds must satisfy "
                "0 <= low_threshold < high_threshold <= 255"
            )


@dataclass(frozen=True, slots=True)
class RoiConfig:
    bottom_left_x: float
    top_left_x: float
    top_right_x: float
    bottom_right_x: float
    top_y: float
    bottom_y: float

    def __post_init__(self) -> None:
        for name, value in (
            ("bottom_left_x", self.bottom_left_x),
            ("top_left_x", self.top_left_x),
            ("top_right_x", self.top_right_x),
            ("bottom_right_x", self.bottom_right_x),
            ("top_y", self.top_y),
            ("bottom_y", self.bottom_y),
        ):
            _require_between(f"roi.{name}", value, 0.0, 1.0)

        if not (
            self.bottom_left_x
            < self.top_left_x
            < self.top_right_x
            < self.bottom_right_x
        ):
            raise ConfigError("ROI horizontal coordinates must increase left to right")

        if self.top_y >= self.bottom_y:
            raise ConfigError("roi.top_y must be smaller than roi.bottom_y")


@dataclass(frozen=True, slots=True)
class HoughConfig:
    rho: float
    theta_degrees: float
    threshold: int
    min_line_length: int
    max_line_gap: int

    def __post_init__(self) -> None:
        if self.rho <= 0:
            raise ConfigError("hough.rho must be positive")
        if self.theta_degrees <= 0:
            raise ConfigError("hough.theta_degrees must be positive")
        if self.threshold <= 0:
            raise ConfigError("hough.threshold must be positive")
        if self.min_line_length <= 0:
            raise ConfigError("hough.min_line_length must be positive")
        if self.max_line_gap < 0:
            raise ConfigError("hough.max_line_gap cannot be negative")


@dataclass(frozen=True, slots=True)
class LanesConfig:
    top_y: float
    minimum_absolute_slope: float
    maximum_absolute_slope: float
    smoothing_frames: int

    def __post_init__(self) -> None:
        _require_between("lanes.top_y", self.top_y, 0.0, 1.0)

        if not (
            0
            < self.minimum_absolute_slope
            < self.maximum_absolute_slope
        ):
            raise ConfigError(
                "Lane slopes must satisfy 0 < minimum < maximum"
            )

        if self.smoothing_frames <= 0:
            raise ConfigError("lanes.smoothing_frames must be positive")


@dataclass(frozen=True, slots=True)
class RenderConfig:
    alpha: float
    line_weight: float
    line_thickness: int

    def __post_init__(self) -> None:
        _require_between("render.alpha", self.alpha, 0.0, 1.0)

        if self.line_weight < 0:
            raise ConfigError("render.line_weight cannot be negative")
        if self.line_thickness <= 0:
            raise ConfigError("render.line_thickness must be positive")


@dataclass(frozen=True, slots=True)
class LaneFindingConfig:
    blur: BlurConfig
    canny: CannyConfig
    roi: RoiConfig
    hough: HoughConfig
    lanes: LanesConfig
    render: RenderConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid [{name}] section")
    return value


def load_config(path: str | Path) -> LaneFindingConfig:
    config_path = Path(path)

    try:
        with config_path.open("rb") as file:
            data = tomllib.load(file)

        return LaneFindingConfig(
            blur=BlurConfig(**_section(data, "blur")),
            canny=CannyConfig(**_section(data, "canny")),
            roi=RoiConfig(**_section(data, "roi")),
            hough=HoughConfig(**_section(data, "hough")),
            lanes=LanesConfig(**_section(data, "lanes")),
            render=RenderConfig(**_section(data, "render")),
        )
    except ConfigError:
        raise
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found: {config_path}") from None
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(
            f"Invalid configuration in {config_path}: {error}"
        ) from error