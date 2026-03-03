import pandera.polars as pa
import polars as pl
import pytest
from pandas import DataFrame
from pandera.polars import PolarsData

from kintsugi.county_pop import CountyPopulationYear, county_age_pop, county_pop

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
    (
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
    ),
)
def test_county_pop(year: CountyPopulationYear) -> None:
    county_pop(year).collect().pipe(CountyPopulation.validate, lazy=True)


def test_county_pop_invalid_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a year in"):
        county_pop(2000)  # pyright: ignore [reportArgumentType]


@pytest.mark.parametrize(
    ("year"),
    (
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
    ),
)
def test_county_pop_as_pandas(year: CountyPopulationYear) -> None:
    df = county_pop(year, True)

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
    (
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
    ),
)
def test_county_age_pop(year: CountyPopulationYear) -> None:
    county_age_pop(year).collect().pipe(CountyAgePopulation.validate, lazy=True)


@pytest.mark.parametrize(
    ("year"),
    (
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
    ),
)
def test_county_age_pop_as_pandas(year: CountyPopulationYear) -> None:
    df = county_age_pop(year, True)

    assert isinstance(df, DataFrame)


def test_county_age_pop_invalid_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a year in"):
        county_age_pop(2000)  # pyright: ignore [reportArgumentType]
