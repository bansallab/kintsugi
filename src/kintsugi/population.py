from typing import Literal, overload

import pandas as pd
import polars as pl

from ._data.data import get_dataset

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


def validate_vintage_year(year: int, vintage_year: VintageYear) -> None:
    """Validate year against vintage_year"""
    vintage_year_lb = 2016
    vintage_year_ub = 2024
    if not (vintage_year_lb <= vintage_year <= vintage_year_ub):
        raise ValueError(
            f"Must choose a vintage year between {vintage_year_lb} and {vintage_year_ub}"
        )

    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if not (year_lb <= year <= vintage_year):
        raise ValueError(f"Must choose a year between {year_lb} and {vintage_year}")


# match conventions in kintsugi-data processing script
sex_enum = pl.Enum(["tot", "male", "female"])
race_enum_no_hispanic = pl.Enum(["white", "black", "aian", "asian", "nhpi"])
race_enum_incl_hispanic = pl.Enum(
    ["white", "black", "aian", "asian", "nhpi", "hispanic"]
)
hispanic_enum = pl.Enum(["tot", "not_hispanic", "hispanic"])


@overload
def state_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def state_pop(
    year: int, *, vintage_year: VintageYear | None = ..., as_pandas: Literal[True]
) -> pd.DataFrame: ...


def state_pop(
    year: int, *, vintage_year: VintageYear | None = None, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    State population estimates for select years.

    Uses state population by characteristics data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html.
    The raw files are not present in the kintsugi-data repo. Instead, parquet files containing a subset of columns are used.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    a specific vintage year may be provided. If `vintage_year` is `None` (the default), data
    for years in the range [2010, 2019] are sourced from the 2020 vintage (2010-2020 data),
    while data for years in the range [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/sc-est2024-alldata5.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    validate_vintage_year(year, vintage_year)
    data = get_dataset(f"pop/state/state_pop_{vintage_year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(
            pl.col("year") == year,
            pl.col("sex") == "tot",
            pl.col("hispanic_origin") == "tot",
        )
        .group_by(["state_name", "state_fips", "year"])
        .agg(tot_pop=pl.col("tot_pop").sum())
        .sort("state_fips")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def state_age_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def state_age_pop(
    year: int, *, vintage_year: VintageYear | None = ..., as_pandas: Literal[True]
) -> pd.DataFrame: ...


def state_age_pop(
    year: int, *, vintage_year: VintageYear | None = None, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    State-age population estimates for select years.

    Age is given in years, not binned groups. Note that an age value of `85` corresponds to >= 85 years old.
    Uses state population by characteristics data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html.
    The raw files are not present in the kintsugi-data repo. Instead, parquet files containing a subset of columns are used.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    a specific vintage year may be provided. If `vintage_year` is `None` (the default), data
    for years in the range [2010, 2019] are sourced from the 2020 vintage (2010-2020 data),
    while data for years in the range [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/sc-est2024-alldata5.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    validate_vintage_year(year, vintage_year)
    data = get_dataset(f"pop/state/state_pop_{vintage_year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(
            pl.col("year") == year,
            pl.col("sex") == "tot",
            pl.col("hispanic_origin") == "tot",
        )
        .group_by(["state_name", "state_fips", "year", "age"])
        .agg(tot_pop=pl.col("tot_pop").sum())
        .sort("state_fips", "age")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def state_sex_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def state_sex_pop(
    year: int, *, vintage_year: VintageYear | None = ..., as_pandas: Literal[True]
) -> pd.DataFrame: ...


def state_sex_pop(
    year: int, *, vintage_year: VintageYear | None = None, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    State-sex population estimates for select years. Uses state population by characteristics
    data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html
    The raw files are not present in the kintsugi-data repo. Instead, we use parquet files containing a subset of columns.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    you may specify a specific vintage year. If `vintage_year` is `None` (by default), data
    for years in the range [2010, 2019] are sourced from the 2020 vintage (2010-2020 data),
    while data for years in the range [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/sc-est2024-alldata5.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    validate_vintage_year(year, vintage_year)
    data = get_dataset(f"pop/state/state_pop_{vintage_year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(
            pl.col("year") == year,
            pl.col("sex") != "tot",
            pl.col("hispanic_origin") == "tot",
        )
        .group_by(["state_name", "state_fips", "year", "sex"])
        .agg(tot_pop=pl.col("tot_pop").sum())
        .sort("state_fips", "sex")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def state_race_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    incl_hispanic_orig: bool = ...,
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def state_race_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    incl_hispanic_orig: bool = ...,
    as_pandas: Literal[True],
) -> pd.DataFrame: ...


def state_race_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = None,
    incl_hispanic_orig: bool = False,
    as_pandas: bool = False,
) -> pl.LazyFrame | pd.DataFrame:
    """
    State-race population estimates for select years. Specify `incl_hispanic_orig=True` to include
    Hispanic counts column. Uses state population by characteristics
    data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html
    The raw files are not present in the kintsugi-data repo. Instead, we use parquet files containing a subset of columns.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    you may specify a specific vintage year. If `vintage_year` is `None` (by default), data
    for years in the range [2010, 2019] are sourced from the 2020 vintage (2010-2020 data),
    while data for years in the range [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/sc-est2024-alldata5.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    validate_vintage_year(year, vintage_year)
    data = get_dataset(f"pop/state/state_pop_{vintage_year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(
            pl.col("year") == year,
            pl.col("sex") == "tot",
            pl.col("hispanic_origin") != "tot"
            if incl_hispanic_orig
            else pl.col("hispanic_origin") == "tot",
        )
        .group_by(
            ["state_name", "state_fips", "year", "race", "hispanic_origin"]
            if incl_hispanic_orig
            else ["state_name", "state_fips", "year", "race"]
        )
        .agg(tot_pop=pl.col("tot_pop").sum())
        .sort(
            ["state_fips", "race", "hispanic_origin"]
            if incl_hispanic_orig
            else ["state_fips", "race"]
        )
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def state_age_sex_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def state_age_sex_pop(
    year: int, *, vintage_year: VintageYear | None = ..., as_pandas: Literal[True]
) -> pd.DataFrame: ...


def state_age_sex_pop(
    year: int, *, vintage_year: VintageYear | None = None, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    State-age-sex population estimates for select years.

    Age is given in years, not binned groups. Note that an age value of `85` corresponds to >= 85 years old.
    Uses state population by characteristics data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html.
    The raw files are not present in the kintsugi-data repo. Instead, parquet files containing a subset of columns are used.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    a specific vintage year may be provided. If `vintage_year` is `None` (the default), data
    for years in the range [2010, 2019] are sourced from the 2020 vintage (2010-2020 data),
    while data for years in the range [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/sc-est2024-alldata5.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    validate_vintage_year(year, vintage_year)
    data = get_dataset(f"pop/state/state_pop_{vintage_year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(
            pl.col("year") == year,
            pl.col("sex") != "tot",
            pl.col("hispanic_origin") == "tot",
        )
        .group_by(["state_name", "state_fips", "year", "age", "sex"])
        .agg(tot_pop=pl.col("tot_pop").sum())
        .sort("state_fips", "age", "sex")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


age_grps = [
    "tot",
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    ">=85",
]
age_grp_enum = pl.Enum(age_grps)


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
    Instead, parquet files containing a subset of columns are used.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    a specific vintage year may be provided. If `vintage_year` is `None` (the default), data
    for years in the range [2010, 2019] are sourced from the 2020 vintage (2010-2020 data),
    while data for years in the range [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    validate_vintage_year(year, vintage_year)
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
    Instead, parquet files containing a subset of columns are used.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    a specific vintage year may be provided. If `vintage_year` is `None` (the default), data
    for years in the range [2010, 2019] are sourced from the 2020 vintage (2010-2020 data),
    while data for years in the range [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    validate_vintage_year(year, vintage_year)
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


@overload
def county_sex_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def county_sex_pop(
    year: int, *, vintage_year: VintageYear | None = ..., as_pandas: Literal[True]
) -> pd.DataFrame: ...


def county_sex_pop(
    year: int, *, vintage_year: VintageYear | None = None, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    County-sex population estimates for select years. Uses county population
    by characteristics data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html
    The raw files are not present in the kintsugi-data repo because of their large size.
    Instead, parquet files containing a subset of columns are used.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    a specific vintage year may be provided. If `vintage_year` is `None` (the default), data
    for years in the range [2010, 2019] are sourced from the 2020 vintage (2010-2020 data),
    while data for years in the range [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    validate_vintage_year(year, vintage_year)
    data = get_dataset(f"pop/county_cc/county_pop_{vintage_year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(
            pl.col("year") == year,
            pl.col("age_grp") == "tot",
        )
        .select(
            "state_name", "county_name", "county_fips", "year", "tot_male", "tot_female"
        )
        .unpivot(
            index=["state_name", "county_name", "county_fips", "year"],
            variable_name="sex",
            value_name="tot_pop",
        )
        .with_columns(sex=pl.col("sex").str.replace("tot_", "").cast(sex_enum))
        .sort("county_fips", "sex")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def county_race_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    incl_hispanic_orig: bool = ...,
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def county_race_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    incl_hispanic_orig: bool = ...,
    as_pandas: Literal[True],
) -> pd.DataFrame: ...


def county_race_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = None,
    incl_hispanic_orig: bool = False,
    as_pandas: bool = False,
) -> pl.LazyFrame | pd.DataFrame:
    """
    County-race population estimates for select years. Specify `incl_hispanic_orig=True` to include
    Hispanic counts column. Uses county population by characteristics
    data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html
    The raw files are not present in the kintsugi-data repo because of their large size.
    Instead, parquet files containing a subset of columns are used.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    a specific vintage year may be provided. If `vintage_year` is `None` (the default), data
    for years in the range [2010, 2019] are sourced from the 2020 vintage (2010-2020 data),
    while data for years in the range [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    validate_vintage_year(year, vintage_year)
    data = get_dataset(f"pop/county_cc/county_pop_{vintage_year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(
            pl.col("year") == year,
            pl.col("age_grp") == "tot",
        )
        .select(
            "state_name",
            "county_name",
            "county_fips",
            "year",
            "white_male",
            "white_female",
            "black_male",
            "black_female",
            "aian_male",
            "aian_female",
            "asian_male",
            "asian_female",
            "nhpi_male",
            "nhpi_female",
            "hispanic_male",
            "hispanic_female",
        )
        .with_columns(
            (pl.col(f"{r}_male") + pl.col(f"{r}_female")).alias(r)
            for r in ["white", "black", "aian", "asian", "nhpi", "hispanic"]
        )
        .select(
            "state_name",
            "county_name",
            "county_fips",
            "year",
            "white",
            "black",
            "aian",
            "asian",
            "nhpi",
            "hispanic",
        )
    )

    if not incl_hispanic_orig:
        lf = lf.drop("hispanic")

    lf = (
        lf.unpivot(
            index=["state_name", "county_name", "county_fips", "year"],
            variable_name="race",
            value_name="tot_pop",
        )
        .cast({"race": race_enum_incl_hispanic})
        .sort("county_fips", "race")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def county_age_sex_pop(
    year: int,
    *,
    vintage_year: VintageYear | None = ...,
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def county_age_sex_pop(
    year: int, *, vintage_year: VintageYear | None = ..., as_pandas: Literal[True]
) -> pd.DataFrame: ...


def county_age_sex_pop(
    year: int, *, vintage_year: VintageYear | None = None, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    County-age-sex population estimates for select years. Uses county population
    by characteristics data: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html
    The raw files are not present in the kintsugi-data repo because of their large size.
    Instead, parquet files containing a subset of columns are used.

    It's recommended to use the latest possible vintage to get a given year's data. However,
    a specific vintage year may be provided. If `vintage_year` is `None` (the default), data
    for years in the range [2010, 2019] are sourced from the 2020 vintage (2010-2020 data),
    while data for years in the range [2020, 2024] are sourced from the 2024 vintage (2020-2024 data).

    Source (2024 example): https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv
    """
    if vintage_year is None:
        if 2010 <= year <= 2019:
            vintage_year = 2020
        else:
            vintage_year = 2024

    validate_vintage_year(year, vintage_year)
    data = get_dataset(f"pop/county_cc/county_pop_{vintage_year}.parquet")
    lf = (
        pl.scan_parquet(data)
        .filter(
            pl.col("year") == year,
            pl.col("age_grp") != "tot",
        )
        .select(
            "state_name",
            "county_name",
            "county_fips",
            "year",
            "age_grp",
            "tot_male",
            "tot_female",
        )
        .unpivot(
            index=["state_name", "county_name", "county_fips", "year", "age_grp"],
            variable_name="sex",
            value_name="tot_pop",
        )
        .with_columns(sex=pl.col("sex").str.replace("tot_", "").cast(sex_enum))
        .sort("county_fips", "age_grp", "sex")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf
