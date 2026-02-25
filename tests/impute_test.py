import io
import textwrap

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from kintsugi.impute import impute_column_pair, impute_columns


@pytest.fixture(scope="module")
def df_inp_flags() -> pl.DataFrame:
    # NOTE: imp_num and imp_denom independently denote whether col needs imputation
    data = """
    id,numerator,denominator,imp_num,imp_denom
    A,10,15,false,false
    A,<=5,<=5,true,true
    A,12,23,false,false
    B,<=5,<=5,true,true
    A,22,24,false,false
    B,<=5,13,true,false
    B,<=5,<=5,true,true
    A,10,15,false,false
    C,<=5,<=5,false,true
    C,<=5,<=5,true,true
    A,<=5,<=5,true,true
    A,22,15,false,false
    B,<=5,13,true,false
    A,<=5,<=5,false,true
    C,100,128,false,false
    C,<=5,<=5,true,true
    D,<=5,<=5,true,true
    A,22,23,false,false
    B,<=5,18,true,false
    H,8,17,false,false
    A,10,16,false,false
    A,<=5,<=5,true,true
    H,<=5,<=5,true,true
    A,22,88,false,false
    B,<=5,23,true,false
    C,<=5,<=5,true,true
    A,<=5,<=5,false,true
    C,100,1300,false,false
    C,<=5,<=5,true,true
    D,<=5,<=5,true,true
    """
    df = pl.read_csv(
        io.StringIO(textwrap.dedent(data)),
        schema={
            "id": pl.String,
            "numerator": pl.String,
            "denominator": pl.String,
            "imp_num": pl.Boolean,
            "imp_denom": pl.Boolean,
        },
    )

    return df


@pytest.fixture(scope="module")
def df_inp_nulls(df_inp_flags: pl.DataFrame) -> pl.DataFrame:
    return df_inp_flags.with_columns(
        pl.col(col).replace({"<=5": None}) for col in ["numerator", "denominator"]
    )


@pytest.fixture(scope="module")
def df(
    df_inp_flags: pl.DataFrame, df_inp_nulls: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return df_inp_flags, df_inp_nulls


@pytest.fixture(scope="module")
def lf(df: tuple[pl.DataFrame, pl.DataFrame]) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    return df[0].lazy(), df[1].lazy()


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, None),
        ("<=5", 1, 5, 8),
        (None, 1, 5, None),
        (None, 1, 5, 8),
    ),
)
def test_impute_columns_single_eager(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]
    df_out = df_inp.pipe(
        impute_columns, "numerator", flag=flag, lb=lb, ub=ub, seed=seed
    )

    assert df_inp.shape == df_out.shape
    assert (
        df_out.select(
            (pl.col("numerator").cast(pl.String).eq_missing(flag)).any()
        ).item()
        is False
    )
    assert (
        df_out.filter(pl.col("imp_num"))
        .select((pl.col("numerator").is_between(lb, ub)).all())
        .item()
        is True
    )


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, None),
        ("<=5", 1, 5, 8),
        (None, 1, 5, None),
        (None, 1, 5, 8),
    ),
)
def test_impute_columns_single_lazy(
    lf: tuple[pl.LazyFrame, pl.LazyFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    lf_inp = lf[1] if flag is None else lf[0]
    lf_out = lf_inp.pipe(
        impute_columns, "numerator", flag=flag, lb=lb, ub=ub, seed=seed
    )

    assert lf_inp.collect().shape == lf_out.collect().shape
    assert (
        lf_out.select((pl.col("numerator").cast(pl.String).eq_missing(flag)).any())
        .collect()
        .item()
        is False
    )
    assert (
        lf_out.filter(pl.col("imp_num"))
        .select((pl.col("numerator").is_between(lb, ub)).all())
        .collect()
        .item()
        is True
    )


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, None),
        ("<=5", 1, 5, 8),
        (None, 1, 5, None),
        (None, 1, 5, 8),
    ),
)
def test_impute_columns_multi_eager(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]
    df_out = df_inp.pipe(
        impute_columns, "numerator", "denominator", flag=flag, lb=lb, ub=ub, seed=seed
    )

    assert df_inp.shape == df_out.shape
    for col, imp_col in zip(("numerator", "denominator"), ("imp_num", "imp_denom")):
        assert (
            df_out.select((pl.col(col).cast(pl.String).eq_missing(flag)).any()).item()
            is False
        )
        assert (
            df_out.filter(pl.col(imp_col))
            .select((pl.col(col).is_between(lb, ub)).all())
            .item()
            is True
        )


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, None),
        ("<=5", 1, 5, 8),
        (None, 1, 5, None),
        (None, 1, 5, 8),
    ),
)
def test_impute_columns_multi_lazy(
    lf: tuple[pl.LazyFrame, pl.LazyFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    lf_inp = lf[1] if flag is None else lf[0]
    lf_out = lf_inp.pipe(
        impute_columns, "numerator", "denominator", flag=flag, lb=lb, ub=ub, seed=seed
    )

    assert lf_inp.collect().shape == lf_out.collect().shape
    for col, imp_col in zip(("numerator", "denominator"), ("imp_num", "imp_denom")):
        assert (
            lf_out.select((pl.col(col).cast(pl.String).eq_missing(flag)).any())
            .collect()
            .item()
            is False
        )
        assert (
            lf_out.filter(pl.col(imp_col))
            .select((pl.col(col).is_between(lb, ub)).all())
            .collect()
            .item()
            is True
        )


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, 75),
        (None, 1, 5, 75),
    ),
)
def test_impute_columns_seed_eager(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]

    df_1 = df_inp.pipe(impute_columns, "numerator", flag=flag, lb=lb, ub=ub, seed=seed)
    df_2 = df_inp.pipe(impute_columns, "numerator", flag=flag, lb=lb, ub=ub, seed=seed)

    assert df_1.shape == df_2.shape
    assert_frame_equal(df_1, df_2)


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, 75),
        (None, 1, 5, 75),
    ),
)
def test_impute_columns_seed_lazy(
    lf: tuple[pl.LazyFrame, pl.LazyFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    lf_inp = lf[1] if flag is None else lf[0]

    lf_1 = lf_inp.pipe(impute_columns, "numerator", flag=flag, lb=lb, ub=ub, seed=seed)
    lf_2 = lf_inp.pipe(impute_columns, "numerator", flag=flag, lb=lb, ub=ub, seed=seed)

    assert lf_1.collect().shape == lf_2.collect().shape
    assert_frame_equal(lf_1, lf_2)


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, None),
        ("<=5", 1, 5, 8),
        (None, 1, 5, None),
        (None, 1, 5, 8),
    ),
)
def test_impute_columns_no_cols_exception(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]
    with pytest.raises(ValueError, match="Must specify at least one column to impute"):
        df_inp.pipe(impute_columns, flag=flag, lb=lb, ub=ub, seed=seed)


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, None),
        ("<=5", 1, 5, 8),
        (None, 1, 5, None),
        (None, 1, 5, 8),
    ),
)
def test_impute_columns_missing_cols_exception(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]
    with pytest.raises(ValueError, match="Column '(.+)' does not exist"):
        df_inp.pipe(impute_columns, "foo", flag=flag, lb=lb, ub=ub, seed=seed)


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=10", 1, 5, None),
        ("<=10", 1, 5, 8),
    ),
)
def test_impute_columns_no_flags_exception(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]
    with pytest.raises(
        ValueError, match="Column '(.+)' does not contain any instances of"
    ):
        df_inp.pipe(impute_columns, "numerator", flag=flag, lb=lb, ub=ub, seed=seed)


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 5, 1, None),
        ("<=5", 5, 1, 8),
        (None, 5, 1, None),
        (None, 5, 1, 8),
    ),
)
def test_impute_columns_invalid_bounds_exception(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]
    with pytest.raises(
        ValueError, match="Lower bound cannot be greater than upper bound"
    ):
        df_inp.pipe(impute_columns, "numerator", flag=flag, lb=lb, ub=ub, seed=seed)


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, None),
        ("<=5", 1, 5, 8),
        (None, 1, 5, None),
        (None, 1, 5, 8),
    ),
)
def test_impute_column_pair_eager(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]
    df_out = df_inp.pipe(
        impute_column_pair,
        "numerator",
        "denominator",
        flag=flag,
        lb=lb,
        ub=ub,
        seed=seed,
    )

    assert df_inp.shape == df_out.shape
    assert (
        df_out.select(
            (pl.col("numerator").cast(pl.String).eq_missing("<=5")).any()
        ).item()
        is False
        and df_out.select(
            (pl.col("denominator").cast(pl.String).eq_missing("<=5")).any()
        ).item()
        is False
    )
    assert (
        df_out.filter(pl.col("imp_num"))
        .select((pl.col("numerator").is_between(lb, pl.col("denominator"))).all())
        .item()
        is True
    )
    assert (
        df_out.filter(pl.col("imp_denom"))
        .select((pl.col("denominator").is_between(lb, ub)).all())
        .item()
        is True
    )


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, None),
        ("<=5", 1, 5, 8),
        (None, 1, 5, None),
        (None, 1, 5, 8),
    ),
)
def test_impute_column_pair_lazy(
    lf: tuple[pl.LazyFrame, pl.LazyFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    lf_inp = lf[1] if flag is None else lf[0]
    lf_out = lf_inp.pipe(
        impute_column_pair,
        "numerator",
        "denominator",
        flag=flag,
        lb=lb,
        ub=ub,
        seed=seed,
    )

    assert lf_inp.collect().shape == lf_out.collect().shape
    assert (
        lf_out.select((pl.col("numerator").cast(pl.String).eq_missing("<=5")).any())
        .collect()
        .item()
        is False
        and lf_out.select(
            (pl.col("denominator").cast(pl.String).eq_missing("<=5")).any()
        )
        .collect()
        .item()
        is False
    )
    assert (
        lf_out.filter(pl.col("imp_num"))
        .select((pl.col("numerator").is_between(lb, pl.col("denominator"))).all())
        .collect()
        .item()
        is True
    )
    assert (
        lf_out.filter(pl.col("imp_denom"))
        .select((pl.col("denominator").is_between(lb, ub)).all())
        .collect()
        .item()
        is True
    )


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 1, 5, None),
        ("<=5", 1, 5, 8),
        (None, 1, 5, None),
        (None, 1, 5, 8),
    ),
)
def test_impute_column_pair_missing_cols_exception(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]
    with pytest.raises(ValueError, match="Column '(.+)' does not exist"):
        df_inp.pipe(
            impute_column_pair, "foo", "numerator", flag=flag, lb=lb, ub=ub, seed=seed
        )

    with pytest.raises(ValueError, match="Column '(.+)' does not exist"):
        df_inp.pipe(
            impute_column_pair, "numerator", "foo", flag=flag, lb=lb, ub=ub, seed=seed
        )


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=10", 1, 5, None),
        ("<=10", 1, 5, 8),
    ),
)
def test_impute_column_pair_no_flags_exception(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]
    with pytest.raises(
        ValueError, match="Column '(.+)' does not contain any instances of"
    ):
        df_inp.pipe(
            impute_column_pair,
            "numerator",
            "denominator",
            flag=flag,
            lb=lb,
            ub=ub,
            seed=seed,
        )


@pytest.mark.parametrize(
    ("flag", "lb", "ub", "seed"),
    (
        ("<=5", 5, 1, None),
        ("<=5", 5, 1, 8),
        (None, 5, 1, None),
        (None, 5, 1, 8),
    ),
)
def test_impute_column_pair_invalid_bounds_exception(
    df: tuple[pl.DataFrame, pl.DataFrame],
    flag: str | None,
    lb: int,
    ub: int,
    seed: int | None,
) -> None:
    df_inp = df[1] if flag is None else df[0]
    with pytest.raises(
        ValueError, match="Lower bound cannot be greater than upper bound"
    ):
        df_inp.pipe(
            impute_column_pair,
            "numerator",
            "denominator",
            flag=flag,
            lb=lb,
            ub=ub,
            seed=seed,
        )
