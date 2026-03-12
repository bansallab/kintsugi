import pandera.polars as pa
import polars as pl
import pytest
from pandas import DataFrame

from kintsugi.crosswalk import county_to_zip, zip_to_county

from .models import BasePolarsModel


class ZipCountyCrosswalk(BasePolarsModel):
    zip_code: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    res_ratio: pl.Float64 = pa.Field(ge=0)  # pyright: ignore [reportAny]


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
