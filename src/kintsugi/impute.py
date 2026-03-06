import polars as pl
from polars.datatypes.classes import FloatType, IntegerType


def impute_columns[T: (pl.DataFrame, pl.LazyFrame)](
    df: T,
    *columns: str,
    flag: str | None = None,
    lb: int,
    ub: int,
    seed: int | None = None,
    dtype: type[IntegerType] | type[FloatType] = pl.Int64,
) -> T:
    """
    Independently fill instances of `flag` (some string) in
    the given column(s) with random integers from the interval
    [lb, ub]. If `flag` is `None`, then null values are filled instead.
    To impute with a single value, specify the same value for both `lb` and `ub`.

    Attempt to cast the filled column(s) to a Polars integer (default) or float type.

    `seed` is passed along to the underlying random number generator.
    """
    if len(columns) == 0:
        raise ValueError("Must specify at least one column to impute")

    col_names = df.lazy().collect_schema().names()
    for col in columns:
        if col not in col_names:
            raise ValueError(f"Column '{col}' does not exist")

        if flag is not None and not _fill_flag_exists(df, col, flag):
            raise ValueError(
                f"Column '{col}' does not contain any instances of '{flag}'"
            )

    if lb > ub:
        raise ValueError(
            f"Lower bound cannot be greater than upper bound: {lb=}, {ub=}"
        )

    df = df.with_columns(
        pl.when(pl.col(col).eq_missing(flag))
        .then(
            pl.int_range(lb, ub + 1).sample(
                pl.len(),
                with_replacement=True,
                seed=seed + i if seed is not None else seed,
            )
        )
        .otherwise(pl.col(col))
        .alias(col)
        .cast(dtype)
        for i, col in enumerate(columns)
    )

    return df

    # using numpy
    # if n_cols == 1:
    #     column = columns[0]
    #     # NOTE: this implementation and numpy implementation for filling values
    #     # are roughly the same speed; Polars-only implementation marginally faster
    #     df = df.with_columns(
    #         pl.when(pl.col(column) == flag)
    #         .then(
    #             pl.int_range(lb, ub + 1).sample(
    #                 pl.len(), with_replacement=True, seed=seed
    #             )
    #         )
    #         .otherwise(pl.col(column))
    #         .alias(column)
    #         .cast(dtype)
    #     )
    # else:
    #     rng = np.random.default_rng(seed)
    #     n_rows = df.lazy().select(pl.len()).collect().item()
    #     # must generate enough numbers for all columns up-front, otherwise they get reused
    #     fill_nums = rng.integers(
    #         lb,
    #         ub,
    #         size=(n_cols, n_rows),  # generate n_cols * n_rows samples with given shape
    #         endpoint=True,  # make ub inclusive
    #     )
    #
    #     exprs = (
    #         pl.when(pl.col(col) == flag)
    #         .then(pl.lit(num))
    #         .otherwise(pl.col(col))
    #         .alias(col)
    #         .cast(dtype)
    #         for col, num in zip(columns, fill_nums)
    #     )
    #     df = df.with_columns(*exprs)


def impute_column_pair[T: (pl.DataFrame, pl.LazyFrame)](
    df: T,
    numerator: str,
    denominator: str,
    *,
    flag: str | None = None,
    lb: int,
    ub: int,
    seed: int | None = None,
    dtype: type[IntegerType] | type[FloatType] = pl.Int64,
) -> T:
    """
    Fill instances of `flag` in both the `numerator` column
    and the `denominator` column such that numerator <= denominator.

    Attempt to cast the filled columns to a Polars integer (default) or float type.

    Note: `seed` is only used for (1) imputing the denominator and (2) the
    numerator case where the denominator is greater than the upper bound `ub`.
    This is because we cannot guarantee desired reproducible behavior for the
    numerator when denominator is less than or equal to the upper bound since
    such imputation happens per-row.
    """
    col_names = df.lazy().collect_schema().names()

    if numerator not in col_names:
        raise ValueError(f"Column '{numerator}' does not exist")

    if denominator not in col_names:
        raise ValueError(f"Column '{denominator}' does not exist")

    if flag is not None:
        if not _fill_flag_exists(df, numerator, flag):
            raise ValueError(
                f"Column '{numerator}' does not contain any instances of '{flag}'"
            )

        if not _fill_flag_exists(df, denominator, flag):
            raise ValueError(
                f"Column '{denominator}' does not contain any instances of '{flag}'"
            )

    if lb > ub:
        raise ValueError(
            f"Lower bound cannot be greater than upper bound: {lb=}, {ub=}"
        )

    df = df.with_columns(
        pl.when(pl.col(denominator).eq_missing(flag))
        .then(
            pl.int_range(lb, ub + 1).sample(
                pl.len(),
                with_replacement=True,
                seed=seed,
            )
        )
        .otherwise(pl.col(denominator))
        .alias(denominator)
        .cast(dtype)
    ).with_columns(
        # NOTE: sometimes oddly high memory consumption because of pl.int_ranges(),
        # but not sure if this can be fixed
        pl.when(
            pl.col(numerator).eq_missing(flag),
            pl.col(denominator) <= ub,
        )
        .then(pl.int_ranges(lb, pl.col(denominator) + 1).list.sample(1).explode())
        .when(
            pl.col(numerator).eq_missing(flag),
            pl.col(denominator) > ub,
        )
        .then(
            pl.int_range(lb, ub + 1).sample(
                pl.len(),
                with_replacement=True,
                seed=seed + 1 if seed is not None else seed,
            )
        )
        .otherwise(pl.col(numerator))
        .alias(numerator)
        .cast(dtype)
    )

    return df


def _fill_flag_exists(df: pl.DataFrame | pl.LazyFrame, column: str, flag: str) -> bool:
    return df.lazy().select((pl.col(column) == flag).any()).collect().item()  # pyright: ignore [reportAny]
