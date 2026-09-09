from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from moviepy import VideoFileClip
from numpy.typing import NDArray

from lane_finding.config import LaneFindingConfig
from lane_finding.detector import LaneDetector


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    config: LaneFindingConfig,
    *,
    overwrite: bool = False,
) -> None:
    """Process an MP4 video, preserving detector state between frames."""
    source = Path(input_path)
    destination = Path(output_path)

    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")

    if source.resolve() == destination.resolve():
        raise ValueError("Input and output video paths must differ")

    if destination.suffix.lower() != ".mp4":
        raise ValueError("Output video must use the .mp4 extension")

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp.mp4")
    detector = LaneDetector(config)

    clip = None
    processed_clip = None

    def transform(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return detector.process_frame(frame)

    try:
        clip = VideoFileClip(str(source))
        processed_clip = clip.image_transform(transform)
        processed_clip.write_videofile(
            str(temporary),
            codec="libx264",
            audio=False,
        )
        temporary.replace(destination)
    finally:
        if processed_clip is not None:
            processed_clip.close()
        if clip is not None:
            clip.close()
        temporary.unlink(missing_ok=True)
