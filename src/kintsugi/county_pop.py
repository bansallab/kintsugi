from typing import Literal, overload

import pandas as pd
import polars as pl

from ._data import get_dataset

type CountyPopulationYear = Literal[
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
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


@overload
def county_pop(
    year: CountyPopulationYear, as_pandas: Literal[False] = ...
) -> pl.LazyFrame: ...


@overload
def county_pop(
    year: CountyPopulationYear, as_pandas: Literal[True]
) -> pd.DataFrame: ...


def county_pop(
    year: CountyPopulationYear, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    County population estimates for select years. Uses county population
    by characteristics data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html
    These files are not present in the kintsugi-data repo because of their large size.
    Instead, we use parquet files with a selection of columns.

    It's recommended to use the latest possible vintage to get a given year's data.
    Thus, data for years in the range [2010, 2019] are sourced from the 2020 vintage
    (2010-2020 data), while data for years in the range [2020, 2024] are sourced from
    the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv
    """
    years = {
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
    }
    if year not in years:
        raise ValueError(f"Must choose a year in {years}")

    data = get_dataset(f"pop/county_cc/county_pop_{year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(pl.col("age_grp") == "tot")
        .select("state_name", "county_name", "county_fips", "year", "tot_pop")
        .sort("county_fips")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def county_age_pop(
    year: CountyPopulationYear, as_pandas: Literal[False] = ...
) -> pl.LazyFrame: ...


@overload
def county_age_pop(
    year: CountyPopulationYear, as_pandas: Literal[True]
) -> pd.DataFrame: ...


def county_age_pop(
    year: CountyPopulationYear, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    County-age population estimates for select years. Uses county population
    by characteristics data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html
    These files are not present in the kintsugi-data repo because of their large size.
    Instead, we use parquet files with a selection of columns.

    It's recommended to use the latest possible vintage to get a given year's data.
    Thus, data for years in the range [2010, 2019] are sourced from the 2020 vintage
    (2010-2020 data), while data for years in the range [2020, 2024] are sourced from
    the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv
    """
    years = {
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
    }
    if year not in years:
        raise ValueError(f"Must choose a year in {years}")

    data = get_dataset(f"pop/county_cc/county_pop_{year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(pl.col("age_grp") != "tot")
        .select(
            "state_name", "county_name", "county_fips", "year", "age_grp", "tot_pop"
        )
        .sort("county_fips", "age_grp")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf
