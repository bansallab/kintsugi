import polars as pl
import pytest
from polars.testing import assert_frame_equal

from kintsugi.complete import complete_rows


@pytest.fixture(scope="module")
def lf_inp() -> pl.LazyFrame:
    lf = pl.LazyFrame(
        data={
            "country": ["France", "France", "UK", "UK", "Spain"],
            "year": [2020, 2021, 2019, 2020, 2022],
            "value": [1, 2, 3, 4, 5],
        }
    ).sort("country", "year")

    return lf


@pytest.fixture(scope="module")
def df_inp(lf_inp: pl.LazyFrame) -> pl.DataFrame:
    return lf_inp.collect()


@pytest.fixture(scope="module")
def df_out() -> pl.DataFrame:
    df = pl.DataFrame(
        data={
            "country": [
                country for country in ["France", "UK", "Spain"] for _ in range(4)
            ],
            "year": list(range(2019, 2023)) * 3,
            "value": [None, 1, 2, None, 3, 4, None, None, None, None, None, 5],
        },
    ).sort("country", "year")

    return df


def test_complete_existing_eager(df_inp: pl.DataFrame, df_out: pl.DataFrame) -> None:
    df = complete_rows(df_inp, "country", "year").sort("country", "year")

    assert_frame_equal(df, df_out)


def test_complete_existing_eager_pipe(
    df_inp: pl.DataFrame, df_out: pl.DataFrame
) -> None:
    df = df_inp.pipe(complete_rows, "country", "year").sort("country", "year")

    assert_frame_equal(df, df_out)


def test_complete_existing_eager_series(
    df_inp: pl.DataFrame, df_out: pl.DataFrame
) -> None:
    country = pl.Series("country", ["France", "UK", "Spain"])
    year = pl.Series("year", list(range(2019, 2023)))
    df = complete_rows(df_inp, country, year).sort("country", "year")

    assert_frame_equal(df, df_out)


def test_complete_existing_eager_series_pipe(
    df_inp: pl.DataFrame, df_out: pl.DataFrame
) -> None:
    country = pl.Series("country", ["France", "UK", "Spain"])
    year = pl.Series("year", list(range(2019, 2023)))
    df = df_inp.pipe(complete_rows, country, year).sort("country", "year")

    assert_frame_equal(df, df_out)


def test_complete_exception(df_inp: pl.DataFrame) -> None:
    with pytest.raises(
        TypeError, match="^The columns must be specified using str or Polars Series"
    ):
        df_inp.pipe(complete_rows, "country", 1).sort("country", "year")  # pyright: ignore [reportArgumentType]


def test_complete_existing_lazy(lf_inp: pl.LazyFrame, df_out: pl.DataFrame) -> None:
    lf = complete_rows(lf_inp, "country", "year").sort("country", "year")

    assert_frame_equal(lf.collect(), df_out)


def test_complete_existing_lazy_pipe(
    lf_inp: pl.LazyFrame, df_out: pl.DataFrame
) -> None:
    lf = lf_inp.pipe(complete_rows, "country", "year").sort("country", "year")

    assert_frame_equal(lf.collect(), df_out)


def test_complete_existing_lazy_series(
    lf_inp: pl.LazyFrame, df_out: pl.DataFrame
) -> None:
    country = pl.Series("country", ["France", "UK", "Spain"])
    year = pl.Series("year", list(range(2019, 2023)))
    lf = complete_rows(lf_inp, country, year).sort("country", "year")

    assert_frame_equal(lf.collect(), df_out)


def test_complete_existing_lazy_series_pipe(
    lf_inp: pl.LazyFrame, df_out: pl.DataFrame
) -> None:
    country = pl.Series("country", ["France", "UK", "Spain"])
    year = pl.Series("year", list(range(2019, 2023)))
    lf = lf_inp.pipe(complete_rows, country, year).sort("country", "year")

    assert_frame_equal(lf.collect(), df_out)


@pytest.fixture(scope="module")
def lf_out_non_exist() -> pl.LazyFrame:
    lf = pl.LazyFrame(
        data={
            "country": [
                country
                for country in ["China", "France", "UK", "Spain"]
                for _ in range(4)
            ],
            "year": list(range(2019, 2023)) * 4,
            "value": [
                None,
                None,
                None,
                None,
                None,
                1,
                2,
                None,
                3,
                4,
                None,
                None,
                None,
                None,
                None,
                5,
            ],
        },
        schema={
            "country": pl.String,
            "year": pl.Int64,
            "value": pl.Int64,
        },
    ).sort("country", "year")

    return lf


@pytest.fixture(scope="module")
def df_out_non_exist(lf_out_non_exist: pl.LazyFrame) -> pl.DataFrame:
    return lf_out_non_exist.collect()


def test_complete_non_existing_eager(
    df_inp: pl.DataFrame, df_out_non_exist: pl.DataFrame
) -> None:
    df = complete_rows(
        df_inp, pl.Series("country", ["France", "UK", "Spain", "China"]), "year"
    ).sort("country", "year")

    assert_frame_equal(df, df_out_non_exist)


def test_complete_non_existing_eager_pipe(
    df_inp: pl.DataFrame, df_out_non_exist: pl.DataFrame
) -> None:
    df = df_inp.pipe(
        complete_rows,
        pl.Series("country", ["France", "UK", "Spain", "China"]),
        "year",
    ).sort("country", "year")

    assert_frame_equal(df, df_out_non_exist)


def test_complete_non_existing_lazy(
    lf_inp: pl.LazyFrame, lf_out_non_exist: pl.DataFrame
) -> None:
    lf = complete_rows(
        lf_inp,
        pl.Series("country", ["France", "UK", "Spain", "China"]),
        "year",
    ).sort("country", "year")

    assert_frame_equal(lf, lf_out_non_exist)


def test_complete_non_existing_lazy_pipe(
    lf_inp: pl.LazyFrame, lf_out_non_exist: pl.DataFrame
) -> None:
    lf = lf_inp.pipe(
        complete_rows,
        pl.Series("country", ["France", "UK", "Spain", "China"]),
        "year",
    ).sort("country", "year")

    assert_frame_equal(lf, lf_out_non_exist)
