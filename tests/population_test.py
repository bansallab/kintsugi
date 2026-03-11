import pandera.polars as pa
import polars as pl
import pytest
from pandas import DataFrame
from pandera.polars import PolarsData

from kintsugi.population import (
    VintageYear,
    age_grp_enum,
    county_age_pop,
    county_age_sex_pop,
    county_pop,
    county_race_pop,
    county_sex_pop,
    hispanic_enum,
    race_enum_incl_hispanic,
    race_enum_no_hispanic,
    sex_enum,
    state_age_pop,
    state_age_sex_pop,
    state_pop,
    state_race_pop,
    state_sex_pop,
)

from .models import BasePolarsModel


class StatePopulation(BasePolarsModel):
    state_name: pl.String = pa.Field(unique=True)  # pyright: ignore [reportAny]
    state_fips: pl.String = pa.Field(unique=True, in_range=("01", "56"))  # pyright: ignore [reportAny]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    tot_pop: pl.Int64 = pa.Field(gt=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "state_fips", "year"]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())

    @pa.dataframe_check
    def has_correct_height(cls, data: PolarsData) -> bool:
        return data.lazyframe.select(pl.len()).collect().item() == 51  # pyright: ignore [reportAny]


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_state_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        state_pop(year, vintage_year=vintage_year).collect().pipe(
            StatePopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            state_pop(year, vintage_year=vintage_year)


def test_state_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        state_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_state_pop_as_pandas(year: int) -> None:
    df = state_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


class StateAgePopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    state_fips: pl.String = pa.Field(in_range=("01", "56"))  # pyright: ignore [reportAny]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    age: pl.Int64 = pa.Field(in_range=(0, 85))  # pyright: ignore [reportAny]
    tot_pop: pl.Int64 = pa.Field(gt=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "state_fips", "year", "age"]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_state_age_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        state_age_pop(year, vintage_year=vintage_year).collect().pipe(
            StateAgePopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            state_age_pop(year, vintage_year=vintage_year)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_state_age_pop_as_pandas(year: int) -> None:
    df = state_age_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_state_age_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        state_age_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]


class StateSexPopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    state_fips: pl.String = pa.Field(in_range=("01", "56"))  # pyright: ignore [reportAny]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    sex: pl.Enum = pa.Field(dtype_kwargs={"categories": sex_enum.categories})  # pyright: ignore [reportAny]
    tot_pop: pl.Int64 = pa.Field(ge=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "state_fips", "year", "sex"]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_state_sex_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        state_sex_pop(year, vintage_year=vintage_year).collect().pipe(
            StateSexPopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            state_sex_pop(year, vintage_year=vintage_year)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_state_sex_pop_as_pandas(year: int) -> None:
    df = state_sex_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_state_sex_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        state_sex_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]


class StateRacePopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    state_fips: pl.String = pa.Field(in_range=("01", "56"))  # pyright: ignore [reportAny]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    race: pl.Enum = pa.Field(  # pyright: ignore [reportAny]
        dtype_kwargs={"categories": race_enum_no_hispanic.categories}
    )
    tot_pop: pl.Int64 = pa.Field(ge=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "state_fips", "year", "race"]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_state_race_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        state_race_pop(year, vintage_year=vintage_year).collect().pipe(
            StateRacePopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            state_race_pop(year, vintage_year=vintage_year)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_state_race_pop_as_pandas(year: int) -> None:
    df = state_race_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_state_race_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        state_race_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]


class StateRaceHispanicPopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    state_fips: pl.String = pa.Field(in_range=("01", "56"))  # pyright: ignore [reportAny]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    race: pl.Enum = pa.Field(  # pyright: ignore [reportAny]
        dtype_kwargs={"categories": race_enum_no_hispanic.categories}
    )
    hispanic_origin: pl.Enum = pa.Field(  # pyright: ignore [reportAny]
        dtype_kwargs={"categories": hispanic_enum.categories}
    )
    tot_pop: pl.Int64 = pa.Field(ge=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = [
            "state_name",
            "state_fips",
            "year",
            "race",
            "hispanic_origin",
        ]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_state_race_hispanic_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        state_race_pop(
            year, vintage_year=vintage_year, incl_hispanic_orig=True
        ).collect().pipe(StateRaceHispanicPopulation.validate, lazy=True)
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            state_race_pop(year, vintage_year=vintage_year, incl_hispanic_orig=True)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_state_race_hispanic_pop_as_pandas(year: int) -> None:
    df = state_race_pop(year, as_pandas=True, incl_hispanic_orig=True)

    assert isinstance(df, DataFrame)


def test_state_race_hispanic_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        state_race_pop(2023, vintage_year=2000, incl_hispanic_orig=True)  # pyright: ignore [reportArgumentType]


class StateAgeSexPopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    state_fips: pl.String = pa.Field(in_range=("01", "56"))  # pyright: ignore [reportAny]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    age: pl.Int64 = pa.Field(in_range=(0, 85))  # pyright: ignore [reportAny]
    sex: pl.Enum = pa.Field(dtype_kwargs={"categories": sex_enum.categories})  # pyright: ignore [reportAny]
    tot_pop: pl.Int64 = pa.Field(ge=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "state_fips", "year", "age", "sex"]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_state_age_sex_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        state_age_sex_pop(year, vintage_year=vintage_year).collect().pipe(
            StateAgeSexPopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            state_age_sex_pop(year, vintage_year=vintage_year)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_state_age_sex_pop_as_pandas(year: int) -> None:
    df = state_age_sex_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_state_age_sex_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        state_age_sex_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]


class CountyPopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String = pa.Field(unique=True)  # pyright: ignore [reportAny]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    tot_pop: pl.Int64 = pa.Field(gt=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "county_name", "county_fips", "year"]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())

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
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_county_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        county_pop(year, vintage_year=vintage_year).collect().pipe(
            CountyPopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            county_pop(year, vintage_year=vintage_year)


def test_county_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        county_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_county_pop_as_pandas(year: int) -> None:
    df = county_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


class CountyAgePopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    age_grp: pl.Enum = pa.Field(dtype_kwargs={"categories": age_grp_enum.categories})  # pyright: ignore [reportAny]
    tot_pop: pl.Int64 = pa.Field(ge=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = [
            "state_name",
            "county_name",
            "county_fips",
            "year",
            "age_grp",
        ]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())

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
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_county_age_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        county_age_pop(year, vintage_year=vintage_year).collect().pipe(
            CountyAgePopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            county_age_pop(year, vintage_year=vintage_year)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_county_age_pop_as_pandas(year: int) -> None:
    df = county_age_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_county_age_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        county_age_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]


class CountySexPopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    sex: pl.Enum = pa.Field(dtype_kwargs={"categories": sex_enum.categories})  # pyright: ignore [reportAny]
    tot_pop: pl.Int64 = pa.Field(ge=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "county_name", "county_fips", "year", "sex"]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())

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
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_county_sex_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        county_sex_pop(year, vintage_year=vintage_year).collect().pipe(
            CountySexPopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            county_sex_pop(year, vintage_year=vintage_year)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_county_sex_pop_as_pandas(year: int) -> None:
    df = county_sex_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_county_sex_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        county_sex_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]


class CountyRacePopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    race: pl.Enum = pa.Field(  # pyright: ignore [reportAny]
        dtype_kwargs={"categories": race_enum_incl_hispanic.categories}
    )
    tot_pop: pl.Int64 = pa.Field(ge=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "county_name", "county_fips", "year", "race"]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())

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
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_county_race_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        county_race_pop(year, vintage_year=vintage_year).collect().pipe(
            CountyRacePopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            county_race_pop(year, vintage_year=vintage_year)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_county_race_pop_as_pandas(year: int) -> None:
    df = county_race_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_county_race_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        county_race_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]


class CountyRaceHispanicPopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    race: pl.Enum = pa.Field(  # pyright: ignore [reportAny]
        dtype_kwargs={"categories": race_enum_incl_hispanic.categories}
    )
    tot_pop: pl.Int64 = pa.Field(ge=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = ["state_name", "county_name", "county_fips", "year", "race"]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())

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
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_county_race_hispanic_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        county_race_pop(
            year, vintage_year=vintage_year, incl_hispanic_orig=True
        ).collect().pipe(CountyRaceHispanicPopulation.validate, lazy=True)
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            county_race_pop(year, vintage_year=vintage_year, incl_hispanic_orig=True)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_county_race_hispanic_pop_as_pandas(year: int) -> None:
    df = county_race_pop(year, as_pandas=True, incl_hispanic_orig=True)

    assert isinstance(df, DataFrame)


def test_county_race_hispanic_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        county_race_pop(2023, vintage_year=2000, incl_hispanic_orig=True)  # pyright: ignore [reportArgumentType]


class CountyAgeSexPopulation(BasePolarsModel):
    state_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_name: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    county_fips: pl.String  # pyright: ignore [reportUninitializedInstanceVariable]
    year: pl.Int64  # pyright: ignore [reportUninitializedInstanceVariable]
    age_grp: pl.Enum = pa.Field(dtype_kwargs={"categories": age_grp_enum.categories})  # pyright: ignore [reportAny]
    sex: pl.Enum = pa.Field(dtype_kwargs={"categories": sex_enum.categories})  # pyright: ignore [reportAny]
    tot_pop: pl.Int64 = pa.Field(ge=0)  # pyright: ignore [reportAny]

    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        unique: list[str] = [
            "state_name",
            "county_name",
            "county_fips",
            "year",
            "age_grp",
            "sex",
        ]

    @pa.check("year")
    def all_identical(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select((pl.col(data.key).n_unique() == 1).all())

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
    range(2010, 2025),
)
@pytest.mark.parametrize(
    ("vintage_year"),
    range(2016, 2025),
)
def test_county_age_sex_pop(year: int, vintage_year: VintageYear) -> None:
    if vintage_year <= 2020:
        year_lb = 2010
    else:
        year_lb = 2020

    if year_lb <= year <= vintage_year:
        county_age_sex_pop(year, vintage_year=vintage_year).collect().pipe(
            CountyAgeSexPopulation.validate, lazy=True
        )
    else:
        with pytest.raises(ValueError, match="^Must choose a year between"):
            county_age_sex_pop(year, vintage_year=vintage_year)


@pytest.mark.parametrize(
    ("year"),
    range(2010, 2025),
)
def test_county_age_sex_pop_as_pandas(year: int) -> None:
    df = county_age_sex_pop(year, as_pandas=True)

    assert isinstance(df, DataFrame)


def test_county_age_sex_pop_invalid_vintage_year_exception() -> None:
    with pytest.raises(ValueError, match="^Must choose a vintage year between"):
        county_age_sex_pop(2023, vintage_year=2000)  # pyright: ignore [reportArgumentType]
