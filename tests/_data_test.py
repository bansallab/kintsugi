from collections.abc import Generator
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
import requests
from platformdirs import user_cache_path

from kintsugi._data import (
    DATASETS,
    GetDatasetError,
    download_dataset,
    file_valid,
    get_cache_dir,
    get_dataset,
)


@pytest.fixture
def mock_get_cache_dir(tmp_path: Path) -> Generator[Callable[[], Path]]:
    def _get_cache_dir() -> Path:
        return tmp_path

    yield _get_cache_dir


def test_get_dataset(mock_get_cache_dir: Callable[[], Path]) -> None:
    """
    Test basic dataset fetch. Cached file doesn't exist.
    """
    file_name = "test/diamonds.csv"
    file_cached = mock_get_cache_dir() / file_name
    assert not file_cached.is_file()

    with (
        patch.dict(
            "kintsugi._data.DATASETS",
            {
                file_name: "9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4"
            },
        ),
        patch("kintsugi._data.get_cache_dir", mock_get_cache_dir),
    ):
        assert get_dataset(file_name) == file_cached
        assert file_cached.is_file()


def test_get_all_datasets(mock_get_cache_dir: Callable[[], Path]) -> None:
    """
    Test fetching all real datasets. Cached files don't exist.
    """
    with patch("kintsugi._data.get_cache_dir", mock_get_cache_dir):
        for file_name in DATASETS.keys():
            file_cached = mock_get_cache_dir() / file_name
            assert not file_cached.is_file()
            assert get_dataset(file_name) == file_cached
            assert file_cached.is_file()


def test_get_dataset_no_cached(mock_get_cache_dir: Callable[[], Path]) -> None:
    """
    Test that get_dataset() calls download_dataset() when cached version doesn't exist
    """
    file_name = "test/diamonds.csv"
    file_cached = mock_get_cache_dir() / file_name

    with (
        patch.dict(
            "kintsugi._data.DATASETS",
            {
                file_name: "9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4"
            },
        ),
        patch("kintsugi._data.get_cache_dir", mock_get_cache_dir),
        patch("kintsugi._data.download_dataset") as mock_download_dataset,
    ):
        get_dataset(file_name)
        mock_download_dataset.assert_called_once_with(file_name, file_cached)


def test_get_dataset_cache_invalid(mock_get_cache_dir: Callable[[], Path]) -> None:
    """
    Test that get_dataset() calls download_dataset() when cached file exists but is invalid
    """
    file_name = "test/diamonds.csv"
    file_cached = mock_get_cache_dir() / file_name
    file_cached.parent.mkdir()
    with open(file_cached, "w") as f:
        f.write("Invalid content that doesn't match diamonds.csv")

    assert file_cached.is_file()

    with (
        patch.dict(
            "kintsugi._data.DATASETS",
            {
                file_name: "9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4"
            },
        ),
        patch("kintsugi._data.get_cache_dir", mock_get_cache_dir),
        patch("kintsugi._data.download_dataset") as mock_download_dataset,
    ):
        get_dataset(file_name)
        mock_download_dataset.assert_called_once_with(file_name, file_cached)


def test_get_dataset_unknown_dataset_exception() -> None:
    """
    Test unknown dataset error
    """
    with pytest.raises(ValueError, match="not in dataset$"):
        get_dataset("unknown.csv")


def test_get_cache_dir_default() -> None:
    cache_dir = get_cache_dir()
    assert cache_dir == user_cache_path("kintsugi-data").resolve()
    assert cache_dir.is_dir()


def test_get_cache_dir_env_var() -> None:
    with patch.dict("os.environ", {"KINTSUGI_CACHE": "~/.cache/kintsugi-test"}):
        cache_dir = get_cache_dir()
        assert cache_dir == Path("~/.cache/kintsugi-test").expanduser().resolve()
        assert cache_dir.is_dir()


def test_download_dataset(mock_get_cache_dir: Callable[[], Path]) -> None:
    """
    Test downloading and verifying a dataset
    """
    file_name = "test/diamonds.csv"
    file_cached = mock_get_cache_dir() / file_name

    with (
        patch.dict(
            "kintsugi._data.DATASETS",
            {
                file_name: "9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4"
            },
        ),
        patch("kintsugi._data.get_cache_dir", mock_get_cache_dir),
    ):
        download_dataset(file_name, file_cached)
        assert file_cached.is_file()


def test_download_dataset_retry_error(mock_get_cache_dir: Callable[[], Path]) -> None:
    """
    Test that RetryError during download_dataset() leads to GetDatasetError
    """
    with (
        patch("kintsugi._data.get_cache_dir", mock_get_cache_dir),
        patch(
            "kintsugi._data.requests.Session.get",
            side_effect=requests.exceptions.RetryError,
        ),
        pytest.raises(GetDatasetError, match="^Error getting data file"),
    ):
        file_cached = mock_get_cache_dir() / "test/diamonds.csv"
        download_dataset("test/diamonds.csv", file_cached)


def test_file_valid_content_correct(mock_get_cache_dir: Callable[[], Path]) -> None:
    """
    Test file validity check when contents are correct
    """
    file_name = "test.txt"
    file_cached = mock_get_cache_dir() / file_name
    with open(file_cached, "w") as f:
        f.write("Valid content for test.txt")

    with patch.dict(
        "kintsugi._data.DATASETS",
        {file_name: "9227f3934df8fd3b9cde4a201195921c90b3d15d174af1a6831b11cdc78ee5b8"},
    ):
        assert file_valid(file_name, file_cached) is True


def test_file_valid_content_incorrect(mock_get_cache_dir: Callable[[], Path]) -> None:
    """
    Test file validity check when contents are wrong
    """
    file_name = "test/diamonds.csv"
    file_cached = mock_get_cache_dir() / file_name
    file_cached.parent.mkdir()
    with open(file_cached, "w") as f:
        f.write("Invalid content for diamonds.csv")

    with patch.dict(
        "kintsugi._data.DATASETS",
        {file_name: "9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4"},
    ):
        assert file_valid(file_name, file_cached) is False
