import pandera.polars as pa
import polars as pl
from pandas import DataFrame
from pandera.polars import PolarsData

from kintsugi.states import states

from .models import BasePolarsModel


class States(BasePolarsModel):
    state_name: pl.String = pa.Field(unique=True)  # pyright: ignore [reportAny]
    state_abb: pl.String = pa.Field(unique=True)  # pyright: ignore [reportAny]
    state_fips: pl.String = pa.Field(unique=True)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "state_abb", "state_fips"]

    @pa.dataframe_check
    def has_correct_height(cls, data: PolarsData) -> bool:
        return data.lazyframe.select(pl.len()).collect().item() == 51  # pyright: ignore [reportAny] pyrgi

    @pa.dataframe_check
    def has_correct_states(cls, data: PolarsData) -> bool:
        return (
            data.lazyframe.select(
                pl.col("state_fips").is_between(pl.lit("01"), pl.lit("56")).all()
            )
            .collect()
            .item()
            is True
        )


def test_states() -> None:
    states().collect().pipe(States.validate, lazy=True)


def test_states_as_pandas() -> None:
    df_states = states(as_pandas=True)

    assert isinstance(df_states, DataFrame)
