import geopandas as gpd
import pandera.pandas as pa
import pytest
from pandera.typing import Series
from pandera.typing.geopandas import GeoSeries

from kintsugi.geo import ShapefileYear, county_geo, state_geo

from .models import BasePandasModel


class StateGeo(BasePandasModel):
    state_name: Series[str] = pa.Field(unique=True)  # pyright: ignore [reportAny]
    state_abb: Series[str] = pa.Field(unique=True)  # pyright: ignore [reportAny]
    state_fips: Series[str] = pa.Field(unique=True)  # pyright: ignore [reportAny]
    geometry: GeoSeries  # pyright: ignore [reportUninitializedInstanceVariable, reportMissingTypeArgument]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "state_abb", "state_fips"]

    @pa.dataframe_check
    def has_correct_states(cls, df: gpd.GeoDataFrame) -> bool:
        return all(df["state_fips"].between("01", "56"))


class StateGeoArea(StateGeo):
    area_land: Series[int]  # pyright: ignore [reportUninitializedInstanceVariable]
    area_water: Series[int]  # pyright: ignore [reportUninitializedInstanceVariable]


class CountyGeo(BasePandasModel):
    state_name: Series[str]  # pyright: ignore [reportUninitializedInstanceVariable]
    state_abb: Series[str]  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name: Series[str]  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: Series[str] = pa.Field(unique=True)  # pyright: ignore [reportAny]
    geometry: GeoSeries  # pyright: ignore [reportUninitializedInstanceVariable, reportMissingTypeArgument]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "state_abb", "county_name", "county_fips"]

    @pa.dataframe_check
    def has_correct_states(cls, df: gpd.GeoDataFrame) -> bool:
        return all(df["county_fips"].str.slice(0, 2).between("01", "56"))


class CountyGeoArea(CountyGeo):
    area_land: Series[int]  # pyright: ignore [reportUninitializedInstanceVariable]
    area_water: Series[int]  # pyright: ignore [reportUninitializedInstanceVariable]


@pytest.mark.parametrize(
    ("year", "expected"),
    (
        (2020, 51),
        (2024, 51),
    ),
)
def test_state_geo(year: ShapefileYear, expected: int) -> None:
    df_geo = state_geo(year)

    StateGeo.validate(df_geo, lazy=True)
    assert df_geo.shape[0] == expected


@pytest.mark.parametrize(
    ("year", "expected"),
    (
        (2020, 51),
        (2024, 51),
    ),
)
def test_state_geo_incl_area(year: ShapefileYear, expected: int) -> None:
    df_geo = state_geo(year, True)

    StateGeoArea.validate(df_geo, lazy=True)
    assert df_geo.shape[0] == expected


def test_state_geo_invalid_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a year in"):
        state_geo(2019)  # pyright: ignore [reportArgumentType]


@pytest.mark.parametrize(
    ("year", "expected"),
    (
        (2020, 3_143),
        (2024, 3_144),
    ),
)
def test_county_geo(year: ShapefileYear, expected: int) -> None:
    df_geo = county_geo(year)

    CountyGeo.validate(df_geo, lazy=True)
    assert df_geo.shape[0] == expected


@pytest.mark.parametrize(
    ("year", "expected"),
    (
        (2020, 3_143),
        (2024, 3_144),
    ),
)
def test_county_geo_incl_area(year: ShapefileYear, expected: int) -> None:
    df_geo = county_geo(year, True)

    CountyGeoArea.validate(df_geo, lazy=True)
    assert df_geo.shape[0] == expected


def test_county_geo_invalid_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a year in"):
        county_geo(2019)  # pyright: ignore [reportArgumentType]
