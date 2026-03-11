# kintsugi

![Tests](https://github.com/bansallab/kintsugi/actions/workflows/tests.yml/badge.svg)

Collection of commonly used datasets and functions. Datasets are stored in [kintsugi-data](https://github.com/bansallab/kintsugi-data).

## Installation

```bash
# with uv via HTTPS
uv add git+https://github.com/winter-again/kintsugi
# with uv via SSH
uv add git+ssh://git@github.com/winter-again/kintsugi

# with pip via HTTPS
pip install git+https://github.com/winter-again/kintsugi
# with pip via SSH
pip install git+ssh://git@github.com/winter-again/kintsugi
```

## Datasets

Currently supported datasets:

County neighbors

```python
from kintsugi.county_neighbors import county_neighbors

neighbors = county_neighbors()
```

State and county shapefiles

```python
from kintsugi.geo import county_geo, state_geo

counties = county_geo(2024)
states = state_geo(2024)
```

State and county population data, stratified by several different variables:

```python
from kintsugi.population import county_pop, state_age_pop

lf_county_pop = county_pop(2024)
lf_state_age_pop = state_age_pop(2024)
```

Low-population county groups

```python
from kintsugi.county_groups import county_groups

lf_county_grps = county_groups()
```

State FIPS codes, names, and abbreviations

```python
from kintsugi.states import states

lf_states = states()
```
