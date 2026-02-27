import csv
import io
from typing import Literal, overload

import pandas as pd
import polars as pl

from ._data import get_dataset
from .geo import ShapefileYear, county_geo

type NeighborsYear = Literal[2010, 2023, 2024, 2025]


@overload
def county_neighbors(
    year: NeighborsYear, as_pandas: Literal[False] = ...
) -> pl.LazyFrame: ...


@overload
def county_neighbors(year: NeighborsYear, as_pandas: Literal[True]) -> pd.DataFrame: ...


def county_neighbors(
    year: NeighborsYear, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    County neighbors/adjacency data for 50 states and D.C. Each row represents a neighbor relationship,
    and relationships are shown in both directions. A county is never considered a neighbor of itself
    (this is filtered out of the raw data). Starting in 2025 this becomes the standard in the raw data.

    Source: https://www.census.gov/geographies/reference-files/time-series/geo/county-adjacency.html

    FTP:
        - 2010: https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2010.txt
        - 2023: https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2023.txt
        - 2024: https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2024.txt
        - 2025: https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2025.txt
    """
    years = {2010, 2023, 2024, 2025}
    if year not in years:
        raise ValueError(f"Must choose a year in {years}")

    data = get_dataset(f"county_neighbors/county_adjacency{year}.txt")

    if year == 2010:
        with io.StringIO() as buf:
            writer = csv.DictWriter(
                buf,
                fieldnames=[
                    "county_name",
                    "county_fips",
                    "county_name_neighbor",
                    "county_fips_neighbor",
                ],
            )
            writer.writeheader()

            with data.open(encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")

                cur_county = None
                cur_fips = None
                for row in reader:
                    if row[0] != "" and row[1] != "":
                        # update because new focal county
                        cur_county = row[0]
                        cur_fips = row[1]

                    writer.writerow(
                        {
                            "county_name": cur_county,
                            "county_fips": cur_fips,
                            "county_name_neighbor": row[2],
                            "county_fips_neighbor": row[3],
                        }
                    )

            buf.seek(0)  # return to start before Polars read
            lf = pl.scan_csv(
                buf,
                schema={
                    "county_name": pl.String,
                    "county_fips": pl.String,
                    "county_name_neighbor": pl.String,
                    "county_fips_neighbor": pl.String,
                },
            )
    else:
        lf = pl.scan_csv(
            data,
            separator="|",
            new_columns=[
                "county_name",
                "county_fips",
                "county_name_neighbor",
                "county_fips_neighbor",
            ],
            schema_overrides={
                "county_name": pl.String,
                "county_fips": pl.String,
                "county_name_neighbor": pl.String,
                "county_fips_neighbor": pl.String,
            },
        ).select(
            "county_name",
            "county_fips",
            "county_name_neighbor",
            "county_fips_neighbor",
        )

    lf = (
        lf.filter(
            pl.col("county_fips")
            .str.slice(0, 2)
            .is_between(pl.lit("01"), pl.lit("56")),
            pl.col("county_fips_neighbor")
            .str.slice(0, 2)
            .is_between(pl.lit("01"), pl.lit("56")),
            pl.col("county_fips") != pl.col("county_fips_neighbor"),
        )
        .with_columns(
            county_name=pl.col("county_name")
            .str.split_exact(", ", 1)
            .struct.rename_fields(["county_name", "state_abb"])
        )
        .unnest("county_name")
        .with_columns(
            county_name_neighbor=pl.col("county_name_neighbor")
            .str.split_exact(", ", 1)
            .struct.rename_fields(["county_name_neighbor", "state_abb_neighbor"])
        )
        .unnest("county_name_neighbor")
        .sort("county_fips", "county_fips_neighbor")
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def county_adj_list(
    year: NeighborsYear, as_pandas: Literal[False] = ...
) -> pl.LazyFrame: ...


@overload
def county_adj_list(year: NeighborsYear, as_pandas: Literal[True]) -> pd.DataFrame: ...


def county_adj_list(
    year: NeighborsYear, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    Data from `county_neighbors()` but in adjacency list format
    """
    lf = (
        county_neighbors(year)
        .select("county_fips", "county_fips_neighbor")
        .group_by("county_fips")
        .agg(county_fips_neighbor=pl.col("county_fips_neighbor"))
    )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def county_neighbors_from_shapefile(
    year: ShapefileYear, as_pandas: Literal[False] = ...
) -> pl.LazyFrame: ...


@overload
def county_neighbors_from_shapefile(
    year: ShapefileYear, as_pandas: Literal[True]
) -> pd.DataFrame: ...


def county_neighbors_from_shapefile(
    year: ShapefileYear, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """
    Generate county neighbor relationships based on Cartographic Boundary
    county shapefile (self spatial join).
    """
    df_geo = county_geo(year)
    neighbors = (
        df_geo.sjoin(
            df_geo.loc[:, ["state_abb", "county_name", "county_fips", "geometry"]],
            how="inner",
            predicate="touches",
        )
        .loc[
            :,
            [
                "county_name_left",
                "state_abb_left",
                "county_fips_left",
                "county_name_right",
                "state_abb_right",
                "county_fips_right",
            ],
        ]
        .rename(
            columns={
                "county_name_left": "county_name",
                "state_abb_left": "state_abb",
                "county_fips_left": "county_fips",
                "county_name_right": "county_name_neighbor",
                "state_abb_right": "state_abb_neighbor",
                "county_fips_right": "county_fips_neighbor",
            }
        )
        .sort_values("county_fips")
    )

    if as_pandas:
        return pd.DataFrame(neighbors)

    return pl.from_pandas(neighbors).sort("county_fips", "county_fips_neighbor").lazy()
