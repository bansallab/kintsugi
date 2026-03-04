from typing import Literal, overload

import pandas as pd
import polars as pl

from ._data import get_dataset
from .county_pop import county_pop


@overload
def county_groups(as_pandas: Literal[False] = ...) -> pl.LazyFrame: ...


@overload
def county_groups(as_pandas: Literal[True]) -> pd.DataFrame: ...


def county_groups(as_pandas: bool = False) -> pl.LazyFrame | pd.DataFrame:
    """
    County FIPS (2020) groupings that combine low-population (< 20k people) counties.
    Includes population-based weights (derived from 2020 population data) for
    redistributing data. The weights represent a given county's population as
    a proportion of the pooled county group population.

    Note that this table only contains counties that are part of groups.
    """
    lf_county_pop = county_pop(2020).select("county_fips", "tot_pop")
    data = get_dataset("county_groups.parquet")
    lf = (
        pl.scan_parquet(data)
        .select("county_fips", "county_group")
        .unique()
        .join(
            lf_county_pop,
            on="county_fips",
            how="inner",
            validate="1:1",
        )
        .with_columns(county_group_pop=pl.col("tot_pop").sum().over("county_group"))
        .with_columns(pop_wt=pl.col("tot_pop") / pl.col("county_group_pop"))
        .sort("county_fips")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf
