import pytest

from kintsugi.geo import ShapefileYear, county_geo, state_geo


@pytest.mark.parametrize(
    ("year", "expected"),
    (
        (2020, 51),
        (2024, 51),
    ),
)
def test_state_geo(year: ShapefileYear, expected: int) -> None:
    df_geo = state_geo(year)

    assert list(df_geo.columns) == [
        "state_name",
        "state_abb",
        "state_fips",
        "geometry",
    ]
    assert df_geo.shape[0] == expected
    assert df_geo.drop_duplicates("state_fips").shape[0] == expected


@pytest.mark.parametrize("year", (2020, 2024))
def test_state_geo_incl_area(year: ShapefileYear) -> None:
    df_geo = state_geo(year, True)

    assert "area_land" in df_geo.columns
    assert "area_water" in df_geo.columns


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

    assert list(df_geo.columns) == [
        "state_name",
        "state_abb",
        "county_name",
        "county_fips",
        "geometry",
    ]
    assert df_geo.shape[0] == expected
    assert df_geo.drop_duplicates("county_fips").shape[0] == expected


@pytest.mark.parametrize("year", (2020, 2024))
def test_county_geo_incl_area(year: ShapefileYear) -> None:
    df_geo = county_geo(year, True)

    assert "area_land" in df_geo.columns
    assert "area_water" in df_geo.columns


def test_county_geo_invalid_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a year in"):
        county_geo(2019)  # pyright: ignore [reportArgumentType]
