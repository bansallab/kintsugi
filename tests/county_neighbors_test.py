import pandera.polars as pa
import polars as pl
import pytest
from pandas import DataFrame
from pandera.polars import PolarsData

from kintsugi.county_neighbors import (
    NeighborsYear,
    county_adj_list,
    county_neighbors,
    county_neighbors_from_shapefile,
)
from kintsugi.geo import ShapefileYear

from .models import BasePolarsModel


class CountyNeighbors(BasePolarsModel):
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    state_abb: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name_neighbor: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    state_abb_neighbor: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips_neighbor: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = [
            "county_name",
            "state_abb",
            "county_fips",
            "county_name_neighbor",
            "state_abb_neighbor",
            "county_fips_neighbor",
        ]

    @pa.dataframe_check
    def self_is_not_neighbor(cls, data: PolarsData) -> bool:
        return (
            data.lazyframe.select(
                (pl.col("county_fips") == pl.col("county_fips_neighbor")).any()
            )
            .collect()
            .item()
            is False
        )

    @pa.dataframe_check
    def has_correct_states(cls, data: PolarsData) -> bool:
        county_fips = (
            data.lazyframe.select(
                (
                    pl.col("county_fips")
                    .str.slice(0, 2)
                    .is_between(pl.lit("01"), pl.lit("56"))
                ).all()
            )
            .collect()
            .item()
            is True
        )
        county_fips_neighbor = (
            data.lazyframe.select(
                (
                    pl.col("county_fips_neighbor")
                    .str.slice(0, 2)
                    .is_between(pl.lit("01"), pl.lit("56"))
                ).all()
            )
            .collect()
            .item()
            is True
        )

        return county_fips and county_fips_neighbor


@pytest.mark.parametrize(("year"), (2010, 2023, 2024, 2025))
def test_county_neighbors(year: NeighborsYear) -> None:
    county_neighbors(year).collect().pipe(CountyNeighbors.validate, lazy=True)


@pytest.mark.parametrize(("year"), (2010, 2023, 2024, 2025))
def test_county_neighbors_as_pandas(year: NeighborsYear) -> None:
    county_nbrs = county_neighbors(year, True)

    assert isinstance(county_nbrs, DataFrame)


def test_county_neighbors_invalid_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a year in"):
        county_neighbors(2019)  # pyright: ignore [reportArgumentType]


class CountyAdjList(BasePolarsModel):
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips_neighbor: pl.List  # pyright: ignore [reportUninitializedInstanceVariable]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["county_fips", "county_fips_neighbor"]

    @pa.dataframe_check
    def has_correct_states(cls, data: PolarsData) -> bool:
        return (
            data.lazyframe.select(
                (
                    pl.col("county_fips")
                    .str.slice(0, 2)
                    .is_between(pl.lit("01"), pl.lit("56"))
                ).all()
            )
            .collect()
            .item()
            is True
        )


@pytest.mark.parametrize(("year"), (2010, 2023, 2024, 2025))
def test_county_adj_list(year: NeighborsYear) -> None:
    county_adj_list(year).collect().pipe(CountyAdjList.validate, lazy=True)


class CountyNeighborsFromShapefile(BasePolarsModel):
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    state_abb: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name_neighbor: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    state_abb_neighbor: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips_neighbor: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = [
            "county_name",
            "state_abb",
            "county_fips",
            "county_name_neighbor",
            "state_abb_neighbor",
            "county_fips_neighbor",
        ]

    @pa.dataframe_check
    def self_is_not_neighbor(cls, data: PolarsData) -> bool:
        return (
            data.lazyframe.select(
                (pl.col("county_fips") == pl.col("county_fips_neighbor")).any()
            )
            .collect()
            .item()
            is False
        )

    @pa.dataframe_check
    def has_correct_states(cls, data: PolarsData) -> bool:
        county_fips = (
            data.lazyframe.select(
                (
                    pl.col("county_fips")
                    .str.slice(0, 2)
                    .is_between(pl.lit("01"), pl.lit("56"))
                ).all()
            )
            .collect()
            .item()
            is True
        )
        county_fips_neighbor = (
            data.lazyframe.select(
                (
                    pl.col("county_fips_neighbor")
                    .str.slice(0, 2)
                    .is_between(pl.lit("01"), pl.lit("56"))
                ).all()
            )
            .collect()
            .item()
            is True
        )

        return county_fips and county_fips_neighbor


@pytest.mark.parametrize(("year"), (2020, 2024))
def test_county_neighbors_from_shapefile(year: ShapefileYear) -> None:
    county_neighbors_from_shapefile(year).collect().pipe(
        CountyNeighbors.validate, lazy=True
    )


def test_county_neighbors_from_shapefile_invalid_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a year in"):
        county_neighbors_from_shapefile(2019)  # pyright: ignore [reportArgumentType]
