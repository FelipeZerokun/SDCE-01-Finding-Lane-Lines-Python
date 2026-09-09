from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lane_finding.config import ConfigError, load_config
from lane_finding.detector import LaneDetector
from lane_finding.image_io import load_image, save_image
from lane_finding.video import process_video


def _add_processing_arguments(
    command_parser: argparse.ArgumentParser,
) -> None:
    command_parser.add_argument(
        "input",
        type=Path,
        help="Path to the source file.",
    )
    command_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path for the processed file.",
    )
    command_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional configuration file; otherwise use the packaged default.",
    )
    command_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lane-finding",
        description="Detect road lane lines in images and videos.",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    image_parser = commands.add_parser(
        "image",
        help="Detect lane lines in one image.",
    )
    _add_processing_arguments(image_parser)

    video_parser = commands.add_parser(
        "video",
        help="Detect lane lines in an MP4 video.",
    )
    _add_processing_arguments(video_parser)

    return parser


def _run_image(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    image = load_image(args.input)

    detector = LaneDetector(config)
    result = detector.process_frame(image)

    save_image(
        args.output,
        result,
        overwrite=args.force,
    )

    print(f"Wrote processed image to {args.output}")


def _run_video(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    process_video(
        args.input,
        args.output,
        config,
        overwrite=args.force,
    )

    print(f"Wrote processed video to {args.output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "image":
            _run_image(args)
        elif args.command == "video":
            _run_video(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except (
        ConfigError,
        FileNotFoundError,
        FileExistsError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))

    return 0
