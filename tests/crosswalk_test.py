import pandera.polars as pa
import polars as pl
import pytest
from pandas import DataFrame
from pandera.polars import PolarsData

from kintsugi.crosswalk import (
    counties_CT,
    county_to_zip,
    puma_2010_2020,
    puma_2010_county_2020,
    zip_to_county,
)

from .models import BasePolarsModel


class PUMAVersionCrosswalk(BasePolarsModel):
    puma_geoid_2010: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    puma_geoid_2020: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    wt_PUMA_2010_to_2020_MCDC: pl.Float64 = pa.Field(ge=0, le=1.0)  # pyright: ignore [reportAny]
    wt_PUMA_2020_to_2010_MCDC: pl.Float64 = pa.Field(ge=0, le=1.0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["puma_geoid_2010", "puma_geoid_2020"]

    @pa.dataframe_check
    def has_correct_states(cls, data: PolarsData) -> bool:
        return (
            data.lazyframe.select(
                pl.all_horizontal(
                    pl.col("puma_geoid_2010")
                    .str.slice(0, 2)
                    .is_between(pl.lit("01"), pl.lit("56")),
                    pl.col("puma_geoid_2020")
                    .str.slice(0, 2)
                    .is_between(pl.lit("01"), pl.lit("56")),
                ).all()
            )
            .collect()
            .item()
            is True
        )


def test_puma_2010_2020() -> None:
    puma_2010_2020().collect().pipe(PUMAVersionCrosswalk.validate, lazy=True)


def test_puma_2010_2020_as_pandas() -> None:
    df = puma_2010_2020(as_pandas=True)

    assert isinstance(df, DataFrame)


class PUMACountyCrosswalk(BasePolarsModel):
    puma_geoid_2010: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    wt_PUMA_2010_to_county: pl.Float64 = pa.Field(ge=0, le=1.0)  # pyright: ignore [reportAny]
    wt_county_to_PUMA_2010: pl.Float64 = pa.Field(ge=0, le=1.0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["puma_geoid_2010", "county_fips"]

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


def test_puma_2010_county_2020() -> None:
    puma_2010_county_2020().collect().pipe(PUMACountyCrosswalk.validate, lazy=True)


def test_puma_2010_county_2020_as_pandas() -> None:
    df = puma_2010_county_2020(as_pandas=True)

    assert isinstance(df, DataFrame)


class ZipCountyCrosswalk(BasePolarsModel):
    zip_code: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    res_ratio: pl.Float64 = pa.Field(ge=0, le=1.0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["zip_code", "county_fips"]

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
    range(2016, 2026),
)
def test_county_to_zip(year: int) -> None:
    county_to_zip(year).collect().pipe(ZipCountyCrosswalk.validate, lazy=True)


@pytest.mark.parametrize(
    ("year"),
    range(2016, 2026),
)
def test_county_to_zip_as_pandas(year: int) -> None:
    df = county_to_zip(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_county_to_zip_year_exception() -> None:
    with pytest.raises(ValueError, match="Must choose a year between 2016 and 2025"):
        county_to_zip(2010)


@pytest.mark.parametrize(
    ("year"),
    range(2016, 2026),
)
def test_zip_to_county(year: int) -> None:
    zip_to_county(year).collect().pipe(ZipCountyCrosswalk.validate, lazy=True)


@pytest.mark.parametrize(
    ("year"),
    range(2016, 2026),
)
def test_zip_to_county_as_pandas(year: int) -> None:
    df = zip_to_county(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_zip_to_county_year_exception() -> None:
    with pytest.raises(ValueError, match="Must choose a year between 2016 and 2025"):
        zip_to_county(2010)


class CountiesCT(BasePolarsModel):
    county_fips_old: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name_old: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips_new: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name_new: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    wt_new_to_old: pl.Float64 = pa.Field(ge=0, le=1.0)  # pyright: ignore [reportAny]
    wt_old_to_new: pl.Float64 = pa.Field(ge=0, le=1.0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["county_fips_old", "county_fips_new"]

    @pa.dataframe_check
    def has_correct_states(cls, data: PolarsData) -> bool:
        return (
            data.lazyframe.select(
                pl.all_horizontal(
                    pl.col("county_fips_old")
                    .str.slice(0, 2)
                    .is_between(pl.lit("01"), pl.lit("56")),
                    pl.col("county_fips_new")
                    .str.slice(0, 2)
                    .is_between(pl.lit("01"), pl.lit("56")),
                ).all()
            )
            .collect()
            .item()
            is True
        )


def test_counties_CT() -> None:
    counties_CT().collect().pipe(CountiesCT.validate, lazy=True)


def test_counties_CT_as_pandas() -> None:
    df = counties_CT(as_pandas=True)

    assert isinstance(df, DataFrame)
