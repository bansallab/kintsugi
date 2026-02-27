import pandera.pandas as pa
import pandera.polars as pla


class BasePolarsModel(pla.DataFrameModel):
    # NOTE: Config is effectively inherited by sub-classes of BaseModel
    # See: https://stackoverflow.com/questions/74167556/pandera-schemamodels-dont-seem-to-inherit-config
    # And: https://github.com/unionai-oss/pandera/issues/983
    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        coerce: bool = False
        strict: bool = True


class BasePandasModel(pa.DataFrameModel):
    class Config:  # pyright: ignore [reportIncompatibleVariableOverride]
        coerce: bool = False
        strict: bool = True
