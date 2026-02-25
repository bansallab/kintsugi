import polars as pl
from pandas import DataFrame

from kintsugi.states import states


def test_states() -> None:
    lf_states = states()

    assert lf_states.collect_schema().names() == [
        "state_name",
        "state_abb",
        "state_fips",
    ]
    assert lf_states.select(pl.len()).collect().item() == 51
    assert lf_states.unique().select(pl.len()).collect().item() == 51
    assert (
        lf_states.select(pl.any_horizontal(pl.all().is_null().any())).collect().item()
        is False
    )


def test_states_as_pandas() -> None:
    df_states = states(as_pandas=True)

    assert isinstance(df_states, DataFrame)
