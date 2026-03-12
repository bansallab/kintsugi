from typing import Literal, overload

import pandas as pd
import polars as pl

from ._data import get_dataset


@overload
def states(as_pandas: Literal[False] = ...) -> pl.LazyFrame: ...


@overload
def states(as_pandas: Literal[True]) -> pd.DataFrame: ...


def states(as_pandas: bool = False) -> pl.LazyFrame | pd.DataFrame:
    """
    State names, abbreviations, and (2-digit) FIPS codes for 50 states and Washington, D.C.

    Source (see first expandable section): https://www.census.gov/library/reference/code-lists/ansi/ansi-codes-for-states.html
    FTP: https://www2.census.gov/geo/docs/reference/state.txt
    """
    data = get_dataset("state.txt")
    lf = (
        pl.scan_csv(
            data,
            separator="|",
            schema_overrides={
                "STATE": pl.String,
                "STUSAB": pl.String,
                "STATE_NAME": pl.String,
            },
        )
        .select("STATE_NAME", "STUSAB", "STATE")
        .rename(
            {
                "STATE_NAME": "state_name",
                "STUSAB": "state_abb",
                "STATE": "state_fips",
            }
        )
        .filter(pl.col("state_fips").is_between(pl.lit("01"), pl.lit("56")))
        .sort("state_fips")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def counties(year: int, as_pandas: Literal[False] = ...) -> pl.LazyFrame: ...


@overload
def counties(year: int, as_pandas: Literal[True]) -> pd.DataFrame: ...


def counties(year: int, as_pandas: bool = False) -> pl.LazyFrame | pd.DataFrame:
    """
    Generate county names and FIPS codes specific to a given year.

    County FIPS codes are determined by the vintage year of the data. They align
    with the expected changes over time.

    See: https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.2010.html
    See: https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.January_2020.html
    """
    data = get_dataset(f"pop/county_cc/county_pop_{year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .select("county_name", "county_fips")
        .unique()
        .sort("county_fips")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf
