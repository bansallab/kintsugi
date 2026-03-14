import hashlib
import importlib.resources
import json
import logging
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import cast

import requests
from platformdirs import user_cache_path
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger("kintsugi")
logger.addHandler(logging.NullHandler())

BASE_URL = "https://raw.githubusercontent.com/winter-again/kintsugi-data/main/data"


class GetDatasetError(Exception):
    pass


type ChecksumTable = dict[str, ChecksumTable]


@lru_cache(maxsize=1)
def load_checksums() -> ChecksumTable:
    with (importlib.resources.files("kintsugi") / "_data/checksums.json").open() as f:
        checksums = cast(ChecksumTable, json.load(f))

    return checksums


def get_dataset(file_name: str) -> Path:
    """
    Ensure valid dataset file is present in cache and return its path
    """
    file_path_keys = file_name.split("/")
    checksums = load_checksums()
    for key in file_path_keys:
        try:
            checksums = checksums[key]
        except KeyError:
            raise ValueError(f"{file_name} not in dataset")

    file_cached = get_cache_dir() / file_name
    if not file_cached.is_file() or not file_valid(file_name, file_cached):
        logger.debug(
            f"{file_name} not in cache or cached file is invalid. Getting file from kintsugi-data repository."
        )
        try:
            download_dataset(file_name, file_cached)
        except GetDatasetError as err:
            logger.exception(f"Unable to get dataset: {file_name}", exc_info=err)
            raise
    else:
        logger.debug(f"{file_name} already exists in cache")

    return file_cached


def get_cache_dir() -> Path:
    """
    Ensure cache directory exists and return its absolute path
    """
    cache_dir = (
        Path(os.getenv("KINTSUGI_CACHE", user_cache_path("kintsugi-data")))
        .expanduser()
        .resolve()
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as err:
        logger.exception(
            f"Error while setting up cache directory at {cache_dir}", exc_info=err
        )
        raise

    return cache_dir


def download_dataset(file_name: str, file_cached: Path) -> None:
    """
    Download dataset file, verify integrity via checksum, and save to cache
    """
    url = f"{BASE_URL}/{file_name}"
    # 5 total retries, sleep for [0.0, 0.2, 0.4, 0.8, ...] seconds between retries after second try
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=True,
    )
    adapter = HTTPAdapter(max_retries=retries)

    with requests.Session() as s:
        s.mount("https://raw.githubusercontent.com/", adapter)
        try:
            res = s.get(url)
        except requests.exceptions.RetryError as err:
            raise GetDatasetError(
                f"Error getting data file {file_name} despite retry strategy"
            ) from err

    with tempfile.NamedTemporaryFile("wb", delete_on_close=False) as fp:
        fp.write(res.content)
        fp.close()

        tmp_file = Path(fp.name)
        if not file_valid(file_name, tmp_file):
            raise GetDatasetError(
                f"Checksum for {file_name} is invalid. File not copied to cache"
            )

        logger.info(f"File {file_name} valid. Copying to cache")
        file_cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_file, file_cached)


def file_valid(file_name: str, file: Path) -> bool:
    """
    Validate dataset file via sha256 checksum
    """
    with open(file, "rb") as f:
        digest = hashlib.file_digest(f, "sha256")

    checksum = get_checksum(file_name)
    return digest.hexdigest() == checksum


def get_checksum(file_name: str) -> str:
    file_path_keys = file_name.split("/")
    checksums = load_checksums()
    for key in file_path_keys:
        try:
            checksums = checksums[key]
        except KeyError:
            raise ValueError(f"{file_name} not in dataset")

    assert isinstance(checksums, str)
    return checksums
