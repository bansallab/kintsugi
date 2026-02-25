import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

import requests
from platformdirs import user_cache_path
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger("kintsugi")
logger.addHandler(logging.NullHandler())

BASE_URL = "https://raw.githubusercontent.com/winter-again/kintsugi-data/main/data"
DATASETS = {
    "county_neighbors/county_adjacency2010.txt": "7edda309ad38a4dfc6a6c6c30e1753e5490b9c8a3aa4563188841989b4fe9a96",
    "county_neighbors/county_adjacency2023.txt": "2dbf9a8bae1b7c50db3a9db4864b073d2423c3e6e63518c97d649453e2809843",
    "county_neighbors/county_adjacency2024.txt": "20cffeb48ba46972fb949c453d3fbf62620115039c45c88bc80c094885650816",
    "county_neighbors/county_adjacency2025.txt": "27046a5f09f66205fd9869afbfb6dcae744e1c9b85cb19dd767c580211fb4575",
    "geo/cb_2020_us_county_5m.zip": "187e7118304428e5450083beb375e67c2c516c58a01ce52db95aaf24f18df3ba",
    "geo/cb_2020_us_state_5m.zip": "aedc60e0d1924a9030ee6d39ff0ed27ad7d1b0bc86807ea809391a6b9008ffb3",
    "geo/cb_2024_us_county_5m.zip": "a867f8734059b45d1d54a0ba56189dd7e73c42eb451418fa56de44c35232614b",
    "geo/cb_2024_us_state_5m.zip": "c9db0e395c11a1f94a8017fde4f4c7cbee1dca6eb37ba8f1ccaab927df70885f",
    "state.txt": "bea4e03f71a1fa0045ae732aabad11fa541e5932b071c2369bb0d325e8cba5a0",
}


class GetDatasetError(Exception):
    pass


def get_dataset(file_name: str) -> Path:
    """
    Ensure valid dataset file is present in cache and return its path
    """
    if file_name not in DATASETS:
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

    return digest.hexdigest() == DATASETS[file_name]
