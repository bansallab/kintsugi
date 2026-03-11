import pandera.polars as pa
import polars as pl
from pandas import DataFrame
from pandera.polars import PolarsData

from kintsugi.county_groups import county_groups

from .models import BasePolarsModel


class CountyGroups(BasePolarsModel):
    county_fips: pl.String = pa.Field(unique=True)  # pyright: ignore [reportAny]
    county_group: pl.List  # pyright: ignore [reportUninitializedInstanceVariable]
    tot_pop: pl.Int64 = pa.Field(gt=0)  # pyright: ignore [reportAny]
    county_group_pop: pl.Int64 = pa.Field(gt=0)  # pyright: ignore [reportAny]
    pop_wt: pl.Float64 = pa.Field(gt=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["county_fips", "county_group"]

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


def test_county_groups() -> None:
    county_groups().collect().pipe(CountyGroups.validate, lazy=True)


def test_county_groups_as_pandas() -> None:
    df = county_groups(True)

    assert isinstance(df, DataFrame)
