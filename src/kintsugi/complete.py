import polars as pl


def complete_rows[T: (pl.DataFrame, pl.LazyFrame)](
    df: T, *columns: str | pl.Series
) -> T:
    """
    Generate rows for implicit missing values based on column combinations,
    thus making them explicit missing values. Generated values marked with null.

    If columns are referenced by name (str), then only existing values in those
    columns are used for completion. If Series are specified instead, then
    those Series can specify the complete set of possible values. The Series must be
    named after an existing column.

    See issue here for simple test case: https://github.com/pola-rs/polars/issues/9722
    """
    if len(columns) == 0:
        raise ValueError("Must specify at least one column to impute")

    col_names = df.lazy().collect_schema().names()
    for col in columns:
        if (isinstance(col, pl.Series) and col.name not in col_names) or (
            isinstance(col, str) and col not in col_names
        ):
            raise ValueError(f"Column '{col}' does not exist")

    cols: list[pl.Expr | pl.Series] = []
    for col in columns:
        if isinstance(col, pl.Series):
            cols.append(col.unique().implode())
        elif isinstance(col, str):  # pyright: ignore [reportUnnecessaryIsInstance]
            cols.append(pl.col(col).unique().implode())
        else:
            raise TypeError(
                f"The columns must be specified using str or Polars Series. Got {type(col)} instead."
            )

    unique_col_vals = df.select(cols)
    targ_cols = unique_col_vals.collect_schema().names()
    for col in targ_cols:
        unique_col_vals = unique_col_vals.explode(col)

    df_complete = unique_col_vals.join(
        df,
        on=targ_cols,
        how="left",
        validate="1:1",
    )

    return df_complete
