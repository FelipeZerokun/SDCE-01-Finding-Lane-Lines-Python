from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import lane_finding.video as video_module
from lane_finding.config import load_config
from lane_finding.pipeline import Image
from lane_finding.video import process_video

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "default.toml")


class FakeProcessedClip:
    def __init__(self, transform: Callable[[Image], Image]) -> None:
        self._transform = transform
        self.closed = False

    def write_videofile(
        self,
        path: str,
        **_options: object,
    ) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        processed = self._transform(frame)

        assert processed.shape == frame.shape
        Path(path).write_bytes(b"processed video")

    def close(self) -> None:
        self.closed = True


class FakeVideoFileClip:
    def __init__(self, _path: str) -> None:
        self.closed = False

    def image_transform(
        self,
        transform: Callable[[Image], Image],
    ) -> FakeProcessedClip:
        return FakeProcessedClip(transform)

    def close(self) -> None:
        self.closed = True


def test_process_video_creates_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video_module,
        "VideoFileClip",
        FakeVideoFileClip,
    )

    source = tmp_path / "input.mp4"
    destination = tmp_path / "nested" / "output.mp4"
    source.write_bytes(b"source video")

    process_video(source, destination, CONFIG)

    assert destination.read_bytes() == b"processed video"
    assert not list(destination.parent.glob(".*.tmp.mp4"))


def test_process_video_rejects_missing_input(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        process_video(
            tmp_path / "missing.mp4",
            tmp_path / "output.mp4",
            CONFIG,
        )


def test_process_video_refuses_to_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mp4"
    destination = tmp_path / "output.mp4"
    source.write_bytes(b"source")
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        process_video(source, destination, CONFIG)


def test_process_video_can_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video_module,
        "VideoFileClip",
        FakeVideoFileClip,
    )

    source = tmp_path / "input.mp4"
    destination = tmp_path / "output.mp4"
    source.write_bytes(b"source")
    destination.write_bytes(b"existing")

    process_video(source, destination, CONFIG, overwrite=True)

    assert destination.read_bytes() == b"processed video"


def test_process_video_rejects_same_input_and_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")

    with pytest.raises(ValueError, match="must differ"):
        process_video(source, source, CONFIG, overwrite=True)


def test_process_video_requires_mp4_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")

    with pytest.raises(ValueError, match=r"\.mp4"):
        process_video(
            source,
            tmp_path / "output.avi",
            CONFIG,
        )