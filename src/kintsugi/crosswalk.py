from typing import Literal, overload

import pandas as pd
import polars as pl

from ._data import get_dataset

# num_county_subs = 169


@overload
def puma_2010_2020(as_pandas: Literal[False] = ...) -> pl.LazyFrame: ...


@overload
def puma_2010_2020(as_pandas: Literal[True]) -> pd.DataFrame: ...


def puma_2010_2020(as_pandas: bool = False) -> pl.LazyFrame | pd.DataFrame:
    """
    Crosswalk data between 2010 (effective 2012) and 2020 (effective 2022) PUMAs.

    `wt_PUMA_2010_to_2020_MCDC` describes the proportion of the 2010 PUMA's population
    that lives in the 2020 PUMA. Similarly, `wt_PUMA_2020_to_2010_MCDC` describes the
    proportion of the 2020 PUMA's population that lives in the 2010 PUMA.

    Sourced from Missouri Census Data Center (MCDC). Alternative source exists from
    Integrated Public Use Microdata Series (IPUMS).

    - Source (MCDC): https://mcdc.missouri.edu/geography/PUMAs.html
        - See: https://mcdc.missouri.edu/data/corrlst/puma2010-to-puma2020.csv
    - Source (IPUMS): https://usa.ipums.org/usa/volii/pumas20.shtml
        - See: https://usa.ipums.org/usa/resources/volii/PUMA2010_PUMA2020_crosswalk.xls
    """
    data = get_dataset("crosswalk/PUMA/puma2010-to-puma2020.csv")
    lf = (
        pl.scan_csv(
            data,
            skip_rows_after_header=1,
            schema_overrides={
                "state": pl.String,
                "puma12": pl.String,
                "puma22": pl.String,
                "afact": pl.Float64,
                "AFACT2": pl.Float64,
            },
        )
        .select(
            "state",
            "puma22",
            "puma12",
            "afact",  # portion of earlier PUMA's population living in later PUMA
            "AFACT2",  # portion of later PUMA's population living in the earlier PUMA
        )
        .rename(
            {
                "afact": "wt_PUMA_2010_to_2020_MCDC",
                "AFACT2": "wt_PUMA_2020_to_2010_MCDC",
            }
        )
        .with_columns(
            (pl.col(col).str.zfill(5).alias(col) for col in ["puma12", "puma22"]),
            state=pl.col("state").str.zfill(2),
        )
        .filter(pl.col("state").is_between(pl.lit("01"), pl.lit("56")))
        .with_columns(
            puma_geoid_2020=pl.col("state") + pl.col("puma22"),
            puma_geoid_2010=pl.col("state") + pl.col("puma12"),
        )
        .select(
            "puma_geoid_2010",
            "puma_geoid_2020",
            "wt_PUMA_2010_to_2020_MCDC",
            "wt_PUMA_2020_to_2010_MCDC",
        )
        .sort("puma_geoid_2010", "puma_geoid_2020")
    )

    # NOTE: implementation using alternate data source
    # lf_IPUMS = (
    #     pl.read_excel(
    #         PUMS_DATA / "PUMA2010_PUMA2020_crosswalk.xls",
    #         # NOTE: pPUMA20_Pop20 = "Estimated percent of the 2020 PUMA's 2020 population that lies in the area of intersection"
    #         columns=[
    #             "GEOID10",
    #             "GEOID20",
    #             "pPUMA20_Pop20",
    #             "pPUMA10_Pop20",
    #         ],
    #     )
    #     .lazy()
    #     .rename(
    #         {
    #             "GEOID20": "puma_geoid_2020",
    #             "GEOID10": "puma_geoid_2010",
    #             "pPUMA20_Pop20": "wt_PUMA_2020_to_2010",
    #             "pPUMA10_Pop20": "wt_PUMA_2010_to_2020",
    #         }
    #     )
    #     .with_columns(
    #         (pl.col(col) / 100.0).alias(col)
    #         for col in ["wt_PUMA_2020_to_2010", "wt_PUMA_2010_to_2020"]
    #     )
    #     .select(
    #         "puma_geoid_2020",
    #         "puma_geoid_2010",
    #         "wt_PUMA_2020_to_2010",
    #         "wt_PUMA_2010_to_2020",
    #     )
    #     .sort("puma_geoid_2020", "puma_geoid_2010")
    # )

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


# @overload
# def crosswalk_puma_2010_county_2020(
#     as_pandas: Literal[False] = ...,
# ) -> pl.LazyFrame: ...
#
#
# @overload
# def crosswalk_puma_2010_county_2020(as_pandas: Literal[True]) -> pd.DataFrame: ...
#
#
# def crosswalk_puma_2010_county_2020(
#     as_pandas: bool = False,
# ) -> pl.LazyFrame | pd.DataFrame:
#     """
#     Crosswalk data between 2010 PUMAs and 2020 counties.
#
#     Note: uses new CT counties
#
#     Source: MCDC 2022 Geocorr
#         - Form query: https://mcdc.missouri.edu/cgi-bin/broker?_PROGRAM=apps.geocorr2022.sas&_SERVICE=MCDC_long&_debug=0&state=Mo29&state=Al01&state=Ak02&state=Az04&state=Ar05&state=Ca06&state=Co08&state=Ct09&state=De10&state=Dc11&state=Fl12&state=Ga13&state=Hi15&state=Id16&state=Il17&state=In18&state=Ia19&state=Ks20&state=Ky21&state=La22&state=Me23&state=Md24&state=Ma25&state=Mi26&state=Mn27&state=Ms28&state=Mt30&state=Ne31&state=Nv32&state=Nh33&state=Nj34&state=Nm35&state=Ny36&state=Nc37&state=Nd38&state=Oh39&state=Ok40&state=Or41&state=Pa42&state=Pr72&state=Ri44&state=Sc45&state=Sd46&state=Tn47&state=Tx48&state=Ut49&state=Vt50&state=Va51&state=Wa53&state=Wv54&state=Wi55&state=Wy56&g1_=puma12&g2_=county&wtvar=pop20&nozerob=1&fileout=1&filefmt=csv&lstfmt=html&title=&afacts2=on&counties=&metros=&places=&oropt=&latitude=&longitude=&distance=&kiloms=0&locname=
#     """
#     lf = (
#         pl.read_csv(
#             CROSSWALK_DATA / "PUMA/geocorr_puma_2010_to_county_2020_with_afact2.csv",
#             encoding="iso-8859-1",
#             skip_rows_after_header=1,
#             columns=[
#                 "state",
#                 "puma12",
#                 "county",
#                 "afact",
#                 "afact2",
#             ],
#             schema_overrides={
#                 "state": pl.String,
#                 "puma12": pl.String,
#                 "county": pl.String,
#                 "afact": pl.String,
#                 "afact2": pl.String,
#             },
#         )
#         .lazy()
#         .rename({"county": "county_fips"})
#         .with_columns(
#             puma_geoid=pl.col("state") + pl.col("puma12"),
#             wt_PUMA_2010_to_county=pl.col("afact").str.strip_chars().cast(pl.Float64),
#             wt_county_to_PUMA_2010=pl.col("afact2").str.strip_chars().cast(pl.Float64),
#         )
#         .filter(
#             pl.col("state").is_between(pl.lit("01"), pl.lit("56")),
#             # pl.col("wt_PUMA_2010_to_county") != 0
#         )
#         .select(
#             "puma_geoid",
#             "county_fips",
#             "wt_PUMA_2010_to_county",
#             "wt_county_to_PUMA_2010",
#         )
#         .sort("puma_geoid", "county_fips")
#     )
#
#     if as_pandas:
#         return lf.collect().to_pandas()
#
#     return lf


@overload
def county_to_zip(year: int, as_pandas: Literal[False] = ...) -> pl.LazyFrame: ...


@overload
def county_to_zip(year: int, as_pandas: Literal[True]) -> pd.DataFrame: ...


def county_to_zip(year: int, as_pandas: bool = False) -> pl.LazyFrame | pd.DataFrame:
    """
    County-to-zip residential ratio weights.

    Use these weights to crosswalk zip-to-county via a weighted mean.
    2012-2022 data use 2010 Census geographies. 2023-present data use
    2020 Census geographies. All years use quarter 4 data.

    Source: https://www.huduser.gov/portal/datasets/usps_crosswalk.html
    """
    if not (2016 <= year <= 2025):
        raise ValueError("Must choose a year between 2016 and 2025")

    data = get_dataset(f"crosswalk/county_to_zip/county_to_zip_{year}.parquet")
    lf = pl.scan_parquet(data)

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


@overload
def zip_to_county(year: int, as_pandas: Literal[False] = ...) -> pl.LazyFrame: ...


@overload
def zip_to_county(year: int, as_pandas: Literal[True]) -> pd.DataFrame: ...


def zip_to_county(year: int, as_pandas: bool = False) -> pl.LazyFrame | pd.DataFrame:
    """
    Zip-to-county residential ratio weights.

    Use these weights to crosswalk counts data from zip-to-county
    2012-2022 data use 2010 Census geographies. 2023-present data use
    2020 Census geographies. All years use quarter 4 data.

    Source: https://www.huduser.gov/portal/datasets/usps_crosswalk.html
    """
    if not (2016 <= year <= 2025):
        raise ValueError("Must choose a year between 2016 and 2025")

    data = get_dataset(f"crosswalk/zip_to_county/zip_to_county_{year}.parquet")
    lf = pl.scan_parquet(data)

    if as_pandas:
        return lf.collect().to_pandas()

    return lf


# @overload
# def crosswalk_CT_counties(as_pandas: Literal[False] = ...) -> pl.LazyFrame: ...
#
#
# @overload
# def crosswalk_CT_counties(as_pandas: Literal[True]) -> pd.DataFrame: ...
#
#
# def crosswalk_CT_counties(as_pandas: bool = False) -> pl.LazyFrame | pd.DataFrame:
#     """
#     Crosswalk CT counties between pre and post-2022 changes. Weights calculated based on county subdivision populations.
#
#     See:
#         - FIPS code changes: https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.html
#         - CT specific: https://www2.census.gov/geo/pdfs/reference/ct_county_equiv_change.pdf
#         - CT specific: https://www.federalregister.gov/documents/2022/06/06/2022-12063/change-to-county-equivalents-in-the-state-of-connecticut
#
#     Crosswalk: https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.January_2020.html
#     Source: https://www2.census.gov/geo/docs/reference/ct_change/ct_cou_to_cousub_crosswalk.txt
#
#     CT county subdivision populations source: https://www.census.gov/data/tables/time-series/demo/popest/2020s-total-cities-and-towns.html
#     FTP: https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/cities/totals/sub-est2024_9.csv
#
#     """
#     crosswalk = (
#         pl.scan_csv(
#             CROSSWALK_DATA / "county/ct_cou_to_cousub_crosswalk.txt",
#             separator="|",
#             n_rows=174,
#             infer_schema=False,
#         )
#         .select(
#             "STATEFP\n(INCITS38)",
#             "OLD_COUNTYFP\n(INCITS31)",
#             "OLD_COUNTY_NAMELSAD",
#             "NEW_COUNTYFP\n(INCITS31)",
#             "NEW_COUNTY_NAMELSAD",
#             "COUSUBFP",
#             "OLD_COUSUB_GEOID",
#             "NEW_COUSUB_GEOID",
#             "COUSUB_NAMELSAD",
#         )
#         .rename(
#             {
#                 "STATEFP\n(INCITS38)": "state_fips",
#                 "OLD_COUNTYFP\n(INCITS31)": "county_fips_old",
#                 "OLD_COUNTY_NAMELSAD": "county_name_old",
#                 "NEW_COUNTYFP\n(INCITS31)": "county_fips_new",
#                 "NEW_COUNTY_NAMELSAD": "county_name_new",
#                 "COUSUBFP": "county_sub_fips",
#                 "OLD_COUSUB_GEOID": "county_sub_geoid_old",
#                 "NEW_COUSUB_GEOID": "county_sub_geoid_new",
#                 "COUSUB_NAMELSAD": "county_sub_name",
#             }
#         )
#         .filter(
#             # NOTE: 5 rows labeled with "County subdivisions not defined" but their GEOID/FIPS
#             # doesn't make sense anyway
#             pl.col("county_sub_fips") != "00000"
#         )
#         .with_columns(
#             (pl.col("state_fips") + pl.col(col)).alias(col)
#             for col in ["county_fips_old", "county_fips_new"]
#         )
#         .drop("state_fips")
#     )
#     assert crosswalk.select(pl.len()).collect().item() == num_county_subs
#
#     subcounty = (
#         pl.scan_csv(
#             CROSSWALK_DATA / "county/sub-est2024_9.csv",
#             schema_overrides={
#                 "SUMLEV": pl.String,
#                 "STATE": pl.String,
#                 "COUNTY": pl.String,
#                 "COUSUB": pl.String,
#                 "NAME": pl.String,
#                 "POPESTIMATE2024": pl.Int64,
#             },
#         )
#         .select(
#             "SUMLEV",
#             "STATE",
#             "COUNTY",
#             "COUSUB",
#             "NAME",
#             "POPESTIMATE2024",
#         )
#         .rename(
#             {
#                 "SUMLEV": "sumlev",
#                 "STATE": "state_fips",
#                 "COUNTY": "county_fips",
#                 "COUSUB": "county_sub_fips",
#                 "NAME": "county_sub_name",
#                 "POPESTIMATE2024": "pop_2024",
#             }
#         )
#         .filter(
#             # only minor civil divisions
#             pl.col("sumlev") == "061"
#         )
#         .with_columns(county_fips=pl.col("state_fips") + pl.col("county_fips"))
#         .drop("state_fips", "county_fips", "sumlev")
#         .sort("county_sub_fips")
#     )
#     assert subcounty.select(pl.len()).collect().item() == num_county_subs
#
#     lf = (
#         crosswalk.join(
#             subcounty,
#             on="county_sub_fips",
#             how="inner",
#             validate="1:1",
#         )
#         .select(
#             "county_sub_fips",
#             "county_sub_name",
#             "county_fips_new",
#             "county_name_new",
#             "county_fips_old",
#             "county_name_old",
#             "pop_2024",
#         )
#         .with_columns(
#             pop_old=pl.col("pop_2024").sum().over("county_fips_old"),
#             pop_new=pl.col("pop_2024").sum().over("county_fips_new"),
#         )
#         .group_by(
#             [
#                 "county_fips_new",
#                 "county_name_new",
#                 "county_fips_old",
#                 "county_name_old",
#                 "pop_old",
#                 "pop_new",
#             ]
#         )
#         .agg(pop_agg=pl.col("pop_2024").sum())  # agg county sub to county pairs
#         # NOTE: want weights to be expected prop. of origin FIPS that are located in dest. FIPS
#         # Aka prop. of origin FIPS that is located in dest. FIPS
#         # Ex: wt_new_to_old should give expected prop. of new FIPS that is located in old FIPS.
#         .with_columns(
#             wt_new_to_old=pl.col("pop_agg") / pl.col("pop_new"),
#             wt_old_to_new=pl.col("pop_agg") / pl.col("pop_old"),
#         )
#         .select(
#             "county_fips_old",
#             "county_name_old",
#             "county_fips_new",
#             "county_name_new",
#             "wt_new_to_old",
#             "wt_old_to_new",
#         )
#         .sort("county_fips_old")
#     )
#     assert (
#         lf.select(pl.len()).collect().item()
#         == lf.unique(["county_fips_old", "county_fips_new"])
#         .select(pl.len())
#         .collect()
#         .item()
#     )
#
#     if as_pandas:
#         return lf.collect().to_pandas()
#
#     return lf
