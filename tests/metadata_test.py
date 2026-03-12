import pandera.polars as pa
import polars as pl
import pytest
from pandas import DataFrame
from pandera.polars import PolarsData

from kintsugi.metadata import counties, states

from .models import BasePolarsModel


class States(BasePolarsModel):
    state_name: pl.String = pa.Field(unique=True)  # pyright: ignore [reportAny]
    state_abb: pl.String = pa.Field(unique=True)  # pyright: ignore [reportAny]
    state_fips: pl.String = pa.Field(unique=True, in_range=("01", "56"))  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "state_abb", "state_fips"]

    @pa.dataframe_check
    def has_correct_height(cls, data: PolarsData) -> bool:
        return data.lazyframe.select(pl.len()).collect().item() == 51  # pyright: ignore [reportAny] pyrgi


def test_states() -> None:
    states().collect().pipe(States.validate, lazy=True)


def test_states_as_pandas() -> None:
    df_states = states(as_pandas=True)

    assert isinstance(df_states, DataFrame)


class Counties(BasePolarsModel):
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String = pa.Field(unique=True)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["county_name", "county_fips"]

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
    range(2016, 2025),
)
def test_counties(year: int) -> None:
    counties(year).collect().pipe(Counties.validate, lazy=True)


@pytest.mark.parametrize(
    ("year"),
    range(2016, 2025),
)
def test_counties_as_pandas(year: int) -> None:
    df_counties = counties(year, as_pandas=True)

    assert isinstance(df_counties, DataFrame)
