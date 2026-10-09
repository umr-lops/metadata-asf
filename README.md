# metadata-asf

[![CI](https://github.com/umr-lops/metadata-asf/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/umr-lops/metadata-asf/actions/workflows/ci.yml?branch=develop)
[![PyPI](https://img.shields.io/pypi/v/metadata-asf)](https://pypi.org/project/metadata-asf/)
[![PyPI — Python versions](https://img.shields.io/pypi/pyversions/metadata-asf)](https://pypi.org/project/metadata-asf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Python library and CLI to collect SAR acquisition **metadata** from the
[ASF API](https://alaska.sciopssearch.asf.alaska.edu) (`asf_search`) and export it as
**daily Parquet catalogs**.

`metadata-asf` is read-only by design: it never downloads products, stores no Earthdata
credentials, and ships without any heavy I/O dependency. The daily Parquet files it
produces are the interface contract with the companion repository [`fetch-asf`]
(planned), which performs the actual product transfer.

[fetch-asf]: https://github.com/umr-lops/fetch-asf

## Scope

- Provider: **ASF**, through `asf_search`.
- Multi-mission by construction; first target mission is **NISAR**.
- NISAR products collected: **L1 — RSLC**, **L2 — GSLC**.
- Geographic filter configurable via WKT (oceans by default).
- Output: one Parquet file per acquisition day, `{mission}_ocean_YYYYMMDD.parquet`.

Out of scope (see `AGENTS.md`): downloading products, Earthdata authentication, and any
scientific SAR processing. These live in `fetch-asf`.

## Installation (development)

```bash
pip install -e ".[dev]"
pre-commit install
```

Python ≥ 3.10 is required. The project uses a `src/` layout with Hatchling as build
backend, version derived from the Git history via `hatch-vcs`, and an entry point
`metadata-asf` registering `metadata_asf.cli:main`.

## Usage

### CLI — `harvest`

Query the ASF API and write the daily Parquet catalogs:

```bash
metadata-asf harvest \
  --mission NISAR \
  --outputdir ./out \
  --log-verbosity INFO \
  --date 2025-01-01:2025-01-31 \
  --conf config.yaml
```

| Option            | Required | Description                                                        |
|-------------------|----------|--------------------------------------------------------------------|
| `--mission`       | no       | Target mission (default: `NISAR`).                                  |
| `--outputdir`     | yes      | Directory where daily Parquet files are written.                    |
| `--log-verbosity` | no       | `DEBUG`, `INFO`, `WARNING` or `ERROR` (default: `INFO`).            |
| `--date`          | no       | `YYYY-MM-DD` or range `YYYY-MM-DD:YYYY-MM-DD`.                      |
| `--conf`          | no       | YAML configuration file (see `config.example.yaml`).                |

Precedence is **CLI > `--conf` file > mission profile defaults**. See `config.example.yaml`
for a complete working example.

### CLI — `report`

Render a self-contained HTML report on a directory of daily Parquet catalogs
(read-only, no API calls, no JavaScript in the output):

```bash
metadata-asf report \
  --catalogdir ./out \
  --outputfile ./out/report.html
```

The report covers volume & completeness (records, daily files, span, missing/empty
days), the product/instrument mix (platform, product type, level, beam, polarization)
and footprint geometry quality (invalid / antimeridian / near-polar).

| Option            | Required | Description                                                        |
|-------------------|----------|--------------------------------------------------------------------|
| `--catalogdir`    | yes      | Directory of daily Parquet files to report on.                     |
| `--outputfile`    | no       | HTML report path (default: `catalog_report.html`).                 |
| `--mission`       | no       | Mission label for the header (default: inferred from the data).    |
| `--log-verbosity` | no       | `DEBUG`, `INFO`, `WARNING` or `ERROR` (default: `INFO`).            |

### Python API

The public, versioned API consumed by `fetch-asf`:

```python
from pathlib import Path
import datetime as dt

from metadata_asf import search, extract, export

products = search.search(
    mission="NISAR",
    start=dt.date(2025, 1, 1),
    end=dt.date(2025, 1, 31),
    intersects_with=None,          # falls back to the profile's ocean WKT
    max_results=10_000,
)

df = extract.to_dataframe(products, mission="NISAR")
written = export.write_daily_parquet(df, output_dir=Path("./out"))
```

### Parquet schema (guaranteed)

| Column             | Type        | Content                          |
|--------------------|-------------|----------------------------------|
| `granule_id`       | str         | `file_name` / granuleName        |
| `platform`         | str         | e.g. `NISAR`, `SENTINEL-1A`      |
| `geometry`         | str (WKT)   | acquisition footprint            |
| `start_time`       | datetime    | UTC                              |
| `stop_time`        | datetime    | UTC                              |
| `polarization`     | list[str]   | e.g. `["HH", "HV"]`              |
| `beam_mode`        | str         | acquisition mode                 |
| `product_type`     | str         | e.g. `RSLC`, `GSLC`              |
| `processing_level` | str         | `L1`, `L2`                       |

Files are written with the PyArrow engine, Snappy compression and no index; every
timestamp is UTC. Any schema change implies a major version bump plus migration notes
(see `AGENTS.md`, section 7).

## Adding a mission

A mission is fully described by a profile in `src/metadata_asf/profiles/`. Adding one
amounts to: creating the module, registering it in the `MISSIONS` dict and adding tests.
The core code (search, extract, export) never changes — see `AGENTS.md`, section 3.

## Development

```bash
make install          # editable install + dev dependencies
make test             # pytest with branch coverage (>= 80 % required)
make check-codestyle  # ruff + black --check
make mypy             # mypy --strict on src/ and tests/
```

Integration tests performing real ASF API calls opt in explicitly: `pytest -m integration`.

## License

MIT — see [LICENSE](LICENSE).
