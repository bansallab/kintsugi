from typing import Literal, NamedTuple, overload

import pandas as pd
import polars as pl

from ._data import get_dataset

type VintageYear = Literal[
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
]


class Vintage(NamedTuple):
    year_lb: int
    year_ub: int
    county_fips: set[str]


def get_vintage(vintage_year: VintageYear) -> Vintage:
    """
    Get info like year bounds for a given vintage year
    """
    vintage_year_lb = 2016
    vintage_year_ub = 2024
    if not (vintage_year_lb <= vintage_year <= vintage_year_ub):
        raise ValueError(
            f"Must choose a vintage year between {vintage_year_lb} and {vintage_year_ub}"
        )

    data = get_dataset(f"pop/county_cc/county_pop_{vintage_year}.parquet")
    county_fips = set(
        pl.scan_parquet(data)
        .select("county_fips")
        .unique()
        .collect()
        .to_series()
        .to_list()
    )
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    return Vintage(year_lb, vintage_year, county_fips)


@overload
def county_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def county_pop(
    year: int, *, vintage_year: VintageYear | None = ..., as_pandas: Literal[True]
) -> pd.DataFrame: ...


def county_pop(
    year: int, *, vintage_year: VintageYear | None = None, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    County population estimates for select years. Uses county population
    by characteristics data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html
    The raw files are not present in the kintsugi-data repo because of their large size.
    Instead, we use parquet files containing a subset of columns.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    you may specify a specific vintage year if, for example, you need a certain set of county
    geographies. If `vintage_year` is `None` (by default), data for years in the range [2010, 2019]
    are sourced from the 2020 vintage (2010-2020 data), while data for years in the range
    [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    vintage = get_vintage(vintage_year)
    if not (vintage.year_lb <= year <= vintage.year_ub):
        raise ValueError(
            f"Must choose a year between {vintage.year_lb} and {vintage.year_ub}"
        )

    data = get_dataset(f"pop/county_cc/county_pop_{vintage_year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(
            pl.col("year") == year,
            pl.col("age_grp") == "tot",
        )
        .select("state_name", "county_name", "county_fips", "year", "tot_pop")
        .sort("county_fips")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def county_age_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def county_age_pop(
    year: int, *, vintage_year: VintageYear | None = ..., as_pandas: Literal[True]
) -> pd.DataFrame: ...


def county_age_pop(
    year: int, *, vintage_year: VintageYear | None = None, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    County-age population estimates for select years. Uses county population
    by characteristics data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html
    The raw files are not present in the kintsugi-data repo because of their large size.
    Instead, we use parquet files containing a subset of columns.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    you may specify a specific vintage year if, for example, you need a certain set of county
    geographies. If `vintage_year` is `None` (by default), data for years in the range [2010, 2019]
    are sourced from the 2020 vintage (2010-2020 data), while data for years in the range
    [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    vintage = get_vintage(vintage_year)
    if not (vintage.year_lb <= year <= vintage.year_ub):
        raise ValueError(
            f"Must choose a year between {vintage.year_lb} and {vintage.year_ub}"
        )

    data = get_dataset(f"pop/county_cc/county_pop_{vintage_year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(
            pl.col("year") == year,
            pl.col("age_grp") != "tot",
        )
        .select(
            "state_name", "county_name", "county_fips", "year", "age_grp", "tot_pop"
        )
        .sort("county_fips", "age_grp")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf
