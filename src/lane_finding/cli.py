from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lane_finding.config import ConfigError, load_config
from lane_finding.detector import LaneDetector
from lane_finding.image_io import load_image, save_image

DEFAULT_CONFIG_PATH = Path("configs/default.toml")


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
    image_parser.add_argument(
        "input",
        type=Path,
        help="Path to the source image.",
    )
    image_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path for the processed image.",
    )
    image_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Configuration file (default: {DEFAULT_CONFIG_PATH}).",
    )
    image_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "image":
            _run_image(args)
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