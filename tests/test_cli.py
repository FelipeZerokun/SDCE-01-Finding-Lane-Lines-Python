from pathlib import Path

import numpy as np
import pytest

from lane_finding.cli import main
from lane_finding.image_io import load_image, save_image

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"


def test_image_command_creates_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = np.zeros((100, 200, 3), dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    save_image(input_path, source)

    exit_code = main(
        [
            "image",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(CONFIG_PATH),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    assert np.array_equal(load_image(output_path), source)
    assert "Wrote processed image" in capsys.readouterr().out


def test_image_command_refuses_to_overwrite(
    tmp_path: Path,
) -> None:
    source = np.zeros((100, 200, 3), dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    save_image(input_path, source)
    save_image(output_path, source)

    with pytest.raises(SystemExit) as error:
        main(
            [
                "image",
                str(input_path),
                "--output",
                str(output_path),
                "--config",
                str(CONFIG_PATH),
            ]
        )

    assert error.value.code == 2


def test_image_command_reports_missing_input(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "output.png"

    with pytest.raises(SystemExit) as error:
        main(
            [
                "image",
                str(tmp_path / "missing.png"),
                "--output",
                str(output_path),
                "--config",
                str(CONFIG_PATH),
            ]
        )

    assert error.value.code == 2
    assert not output_path.exists()