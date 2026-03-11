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
    "pop/county_cc/county_pop_2016.parquet": "74caad19bf5eed856ad9b6f63c65f7fceca612dec680d0768890de2265116607",
    "pop/county_cc/county_pop_2017.parquet": "d93d027929861e115cf34b15f1ff7c697c8eaa327b73cd8132710a11860a63d5",
    "pop/county_cc/county_pop_2018.parquet": "be3d3bab642a9f6f111c792a431f940b1753373194993885e4d47c136feed91a",
    "pop/county_cc/county_pop_2019.parquet": "98801f118cd795c026a8269d5ac6674f98b9d47e0207c6a2721a5b7f4b6e5c08",
    "pop/county_cc/county_pop_2020.parquet": "f1e4f282d297dc5498b6f839412c0815ca6f9e0a15d83d5d3867f2d70aa8413d",
    "pop/county_cc/county_pop_2021.parquet": "3af369564ebb0e1fda25b440e5bf133ecb2d2eab60ab40f5db1f0a0955db713b",
    "pop/county_cc/county_pop_2022.parquet": "977856eb5fffd508442ccedaa54c92e338b037135e5a9be55a03c7132863d9ca",
    "pop/county_cc/county_pop_2023.parquet": "a4d66c302a557c1565ec9f43bad5ea9d4267576d1fbd17d8939e5a858a3d73e7",
    "pop/county_cc/county_pop_2024.parquet": "12b16c7c20329a3df2f4120f6ec9a9a7313147fad0fd03bc360b1de5769c8abd",
    "pop/state/state_pop_2016.parquet": "bac51c5ba4a9ff7305e92b3b2804c854fc20b9cbcf01156e5439d92668c0c81e",
    "pop/state/state_pop_2017.parquet": "6fb950b1b78409af8130317b08b437b742c0906ff9d5c38655c1189103b8dddc",
    "pop/state/state_pop_2018.parquet": "913fca35299028a842325000e58e33cd3912c1e900d480f00b468095398e57f8",
    "pop/state/state_pop_2019.parquet": "7ca2c87065f24857178bb33a7512cb799a92890596bac6fff1cbeb3c69f6fc36",
    "pop/state/state_pop_2020.parquet": "275b861e07f1c2327fb5382a28e84a5fb7ac4f896ae9f91b06612f6197af9611",
    "pop/state/state_pop_2021.parquet": "8b47a5c9fdca838954c8ddac8265ad00d590281c7b444019070c81b9942a727e",
    "pop/state/state_pop_2022.parquet": "ea113b3766c44bbf250e01b0b9509e810590119b3b9470b13dc347d43aed042b",
    "pop/state/state_pop_2023.parquet": "e96a982342510fe6a1ba90fc85a9bd6fbdd8687bceaf76e6e117606429d2d160",
    "pop/state/state_pop_2024.parquet": "b79bca471a68b8c3742ec30d41a2b65ab1227152e81239faf00763188752c6ff",
    "county_groups.parquet": "7d7c150b5efd5596e0eaaed27abd6dc86137f08ff677c2606d402b9d165b87fa",
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
