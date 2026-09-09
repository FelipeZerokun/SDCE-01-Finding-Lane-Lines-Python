from pathlib import Path

import numpy as np
import pytest

from lane_finding.image_io import load_image, save_image


def test_image_round_trip_preserves_pixels(tmp_path: Path) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :] = (255, 40, 10)
    path = tmp_path / "image.png"

    save_image(path, image)
    loaded = load_image(path)

    assert loaded.dtype == np.uint8
    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        load_image(path)


def test_save_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    path = tmp_path / "image.png"

    save_image(path, image)

    with pytest.raises(FileExistsError, match="already exists"):
        save_image(path, image)


def test_save_can_overwrite_existing_file(tmp_path: Path) -> None:
    first = np.zeros((20, 30, 3), dtype=np.uint8)
    second = np.full((20, 30, 3), 255, dtype=np.uint8)
    path = tmp_path / "image.png"

    save_image(path, first)
    save_image(path, second, overwrite=True)

    assert np.array_equal(load_image(path), second)


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    path = tmp_path / "nested" / "output.png"

    save_image(path, image)

    assert path.is_file()
