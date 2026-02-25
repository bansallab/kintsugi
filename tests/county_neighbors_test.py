import polars as pl
import pytest
from pandas import DataFrame

from kintsugi.county_neighbors import NeighborsYear, county_neighbors


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


@pytest.mark.parametrize(("year"), (2010, 2023, 2024, 2025))
def test_county_neighbors_as_pandas(year: NeighborsYear) -> None:
    county_nbrs = county_neighbors(year, True)

    assert isinstance(county_nbrs, DataFrame)


def test_county_neighbors_invalid_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a year in"):
        county_neighbors(2019)  # pyright: ignore [reportArgumentType]
