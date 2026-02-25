from typing import Literal

import geopandas as gpd

from ._data import get_dataset

type ShapefileYear = Literal[2020, 2024]

shapefile_years = {2020, 2024}
# num_counties_2020 = 3_143
# num_counties_2024 = 3_144


# NOTE: 2020 and 2024 should cover the effective dates of the most recent county FIPS changes:
# AK changes effective 2019-01-02 -> appear starting in 2020
# CT changes effective 2022


def county_geo(year: ShapefileYear, incl_area: bool = False) -> gpd.GeoDataFrame:
    """
    Cartographic Boundary county shapefile (1:5,000,000 scale) data as a GeoPandas GeoDataFrame.

    These are designed for small-scale thematic mapping. Original CRS is 'EPSG:4269' but gets converted
    to 'ESRI:102003' (US contiguous Albers Equal Area Conic). 1:500,000 (500k) and 1:20,000,000 (20m)
    scales are also available.

    Source: https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html
    FTP: https://www2.census.gov/geo/tiger/
        - Example 2020: https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip
    """
    if year not in shapefile_years:
        raise ValueError(f"Must choose a year in {shapefile_years}")

    data = get_dataset(f"geo/cb_{year}_us_county_5m.zip")
    counties = gpd.read_file(
        f"zip://{data}",
        use_arrow=True,
        columns=[
            "STATEFP",
            "GEOID",
            "NAMELSAD",
            "STUSPS",
            "STATE_NAME",
            "ALAND",
            "AWATER",
            "geometry",
        ],
    ).to_crs("ESRI:102003")

    counties = (
        counties.loc[
            counties["STATEFP"].between("01", "56"),
            [
                "GEOID",
                "NAMELSAD",
                "STUSPS",
                "STATE_NAME",
                "ALAND",
                "AWATER",
                "geometry",
            ],
        ]
        .rename(
            columns={
                "GEOID": "county_fips",
                "NAMELSAD": "county_name",
                "STUSPS": "state_abb",
                "STATE_NAME": "state_name",
                "ALAND": "area_land",
                "AWATER": "area_water",
            }
        )[
            [
                "state_name",
                "state_abb",
                "county_name",
                "county_fips",
                "area_land",
                "area_water",
                "geometry",
            ]
        ]
        .sort_values("county_fips", ignore_index=True)
    )

    if not incl_area:
        counties = counties.drop(["area_land", "area_water"], axis=1)

    return counties


def state_geo(year: ShapefileYear, incl_area: bool = False) -> gpd.GeoDataFrame:
    """
    Cartographic Boundary state shapefile (1:5,000,000 scale) data as a GeoPandas GeoDataFrame.

    These are designed for small-scale thematic mapping. Original CRS is 'EPSG:4269' but gets converted
    to 'ESRI:102003' (US contiguous Albers Equal Area Conic). 1:500,000 (500k) and 1:20,000,000 (20m)
    scales are also available.

    Source: https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html
    FTP: https://www2.census.gov/geo/tiger/
        - Example (2020): https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_5m.zip
    """
    if year not in shapefile_years:
        raise ValueError(f"Must choose a year in {shapefile_years}")

    data = get_dataset(f"geo/cb_{year}_us_state_5m.zip")
    states = gpd.read_file(
        f"zip://{data}",
        use_arrow=True,
        columns=[
            "GEOID",
            "STUSPS",
            "NAME",
            "ALAND",
            "AWATER",
            "geometry",
        ],
    ).to_crs("ESRI:102003")

    states = (
        states.loc[
            states["GEOID"].between("01", "56"),
            [
                "GEOID",
                "NAME",
                "STUSPS",
                "ALAND",
                "AWATER",
                "geometry",
            ],
        ]
        .rename(
            columns={
                "GEOID": "state_fips",
                "NAME": "state_name",
                "STUSPS": "state_abb",
                "ALAND": "area_land",
                "AWATER": "area_water",
            }
        )[
            [
                "state_name",
                "state_abb",
                "state_fips",
                "area_land",
                "area_water",
                "geometry",
            ]
        ]
        .sort_values("state_fips", ignore_index=True)
    )

    if not incl_area:
        states = states.drop(["area_land", "area_water"], axis=1)

    return states
