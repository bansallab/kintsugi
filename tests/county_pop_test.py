import pandera.polars as pa
import polars as pl
import pytest
from pandas import DataFrame
from pandera.polars import PolarsData

from kintsugi.county_pop import (
    VintageYear,
    county_age_pop,
    county_pop,
    get_vintage,
)

from .models import BasePolarsModel

age_grps = {
    0: "tot",
    1: "0-4",
    2: "5-9",
    3: "10-14",
    4: "15-19",
    5: "20-24",
    6: "25-29",
    7: "30-34",
    8: "35-39",
    9: "40-44",
    10: "45-49",
    11: "50-54",
    12: "55-59",
    13: "60-64",
    14: "65-69",
    15: "70-74",
    16: "75-79",
    17: "80-84",
    18: ">=85",
}
age_grp_enum = pl.Enum(age_grps.values())


class CountyPopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String = pa.Field(unique=True)  # pyright: ignore [reportAny]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    tot_pop: pl.Int64 = pa.Field(gt=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "county_name", "county_fips", "year"]

    @pa.dataframe_check
    def has_correct_states(cls, data: PolarsData) -> bool:
        return (
            data.lazyframe.select(
                pl.col("county_fips")
                .str.slice(0, 2)
                .is_between(pl.lit("01"), pl.lit("56"))
                .all()
            )
            .collect()
            .item()
            is True
        )


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_county_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        county_pop(year, vintage_year=vintage_year).collect().pipe(
            CountyPopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            county_pop(year, vintage_year=vintage_year)


def test_county_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        county_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]


def test_get_vintage_info() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        get_vintage(2000)  # pyright: ignore [reportArgumentType]


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_county_pop_as_pandas(year: int) -> None:
    df = county_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


class CountyAgePopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    age_grp: pl.Enum = pa.Field(dtype_kwargs={"categories": age_grp_enum.categories})  # pyright: ignore [reportAny]
    tot_pop: pl.Int64 = pa.Field(ge=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = [
            "state_name",
            "county_name",
            "county_fips",
            "year",
            "age_grp",
        ]

    @pa.dataframe_check
    def has_correct_states(cls, data: PolarsData) -> bool:
        return (
            data.lazyframe.select(
                pl.col("county_fips")
                .str.slice(0, 2)
                .is_between(pl.lit("01"), pl.lit("56"))
                .all()
            )
            .collect()
            .item()
            is True
        )


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_county_age_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        county_age_pop(year, vintage_year=vintage_year).collect().pipe(
            CountyAgePopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            county_age_pop(year, vintage_year=vintage_year)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_county_age_pop_as_pandas(year: int) -> None:
    df = county_age_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_county_age_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        county_age_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]
