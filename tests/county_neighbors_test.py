import polars as pl
import pytest
from pandas import DataFrame

from kintsugi.county_neighbors import (
    NeighborsYear,
    county_adj_list,
    county_neighbors,
    county_neighbors_from_shapefile,
)
from kintsugi.geo import ShapefileYear


@pytest.mark.parametrize(("year"), (2010, 2023, 2024, 2025))
def test_county_neighbors(year: NeighborsYear) -> None:
    county_nbrs = county_neighbors(year)

    assert (
        county_nbrs.unique().select(pl.len()).collect().item()
        == county_nbrs.select(pl.len()).collect().item()
    )
    assert (
        county_nbrs.select(
            (pl.col("county_fips") == pl.col("county_fips_neighbor")).any()
        )
        .collect()
        .item()
        is False
    )
    assert (
        county_nbrs.select(
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
def test_county_neighbors_as_pandas(year: NeighborsYear) -> None:
    county_nbrs = county_neighbors(year, True)

    assert isinstance(county_nbrs, DataFrame)


def test_county_neighbors_invalid_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a year in"):
        county_neighbors(2019)  # pyright: ignore [reportArgumentType]


@pytest.mark.parametrize(("year"), (2010, 2023, 2024, 2025))
def test_county_adj_list(year: NeighborsYear) -> None:
    county_nbrs = county_adj_list(year)

    assert (
        county_nbrs.unique().select(pl.len()).collect().item()
        == county_nbrs.select(pl.len()).collect().item()
    )
    assert (
        county_nbrs.select(
            (pl.col("county_fips").is_in(pl.col("county_fips_neighbor"))).any()
        )
        .collect()
        .item()
        is False
    )
    assert (
        county_nbrs.select(
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


@pytest.mark.parametrize(("year"), (2020, 2024))
def test_county_neighbors_from_shapefile(year: ShapefileYear) -> None:
    county_nbrs = county_neighbors_from_shapefile(year)

    assert (
        county_nbrs.unique().select(pl.len()).collect().item()
        == county_nbrs.select(pl.len()).collect().item()
    )
    assert (
        county_nbrs.select(
            (pl.col("county_fips") == pl.col("county_fips_neighbor")).any()
        )
        .collect()
        .item()
        is False
    )
    assert (
        county_nbrs.select(
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


def test_county_neighbors_from_shapefile_invalid_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a year in"):
        county_neighbors_from_shapefile(2019)  # pyright: ignore [reportArgumentType]
