metadata-asf
============

Python library and CLI to collect SAR acquisition **metadata** from the
`ASF API <https://alaska.sciopssearch.asf.alaska.edu>`_ (``asf_search``) and
export it as **daily Parquet catalogs**.

``metadata-asf`` is read-only by design: it never downloads products, stores no
Earthdata credentials, and ships without any heavy I/O dependency. The daily
Parquet files it produces are the interface contract with the companion
repository `fetch-asf <https://github.com/umr-lops/fetch-asf>`_, which
performs the actual product transfer.

Scope
-----

- Provider: **ASF**, through ``asf_search``.
- Multi-mission by construction; first target mission is **NISAR**.
- NISAR products collected: **L1 — RSLC**, **L2 — GSLC**.
- Geographic filter configurable via WKT (oceans by default).
- Output: one Parquet file per acquisition day, ``{mission}_ocean_YYYYMMDD.parquet``.

Installation
------------

.. code-block:: bash

   pip install metadata-asf

For development:

.. code-block:: bash

   git clone https://github.com/umr-lops/metadata-asf.git
   cd metadata-asf
   pip install -e ".[dev,docs]"
   pre-commit install

Python >= 3.10 is required.

CLI usage
---------

The ``metadata-asf`` command has two subcommands: ``harvest`` (query the ASF API
and write the daily Parquet catalogs) and ``report`` (render an HTML report on an
existing catalog).

Harvest
~~~~~~~

.. code-block:: bash

    metadata-asf harvest \
      --mission NISAR \
      --outputdir ./out \
      --log-verbosity INFO \
      --start 2025-01-01 \
      --stop 2025-01-31 \
      --conf config.yaml

The window is harvested **sequentially, one day at a time**: each day is
searched, extracted and written to its own Parquet file. A day whose file
already exists is skipped, so a partially finished span can be re-run to
completion.

===============  ========  =====================================================
Option           Required  Description
===============  ========  =====================================================
--mission         no       Target mission (default: NISAR)
--outputdir       yes      Directory where daily Parquet files are written
--log-verbosity   no       DEBUG, INFO, WARNING or ERROR (default INFO)
--start           yes*     First day of the window, YYYY-MM-DD
--stop            no       Last day, YYYY-MM-DD (inclusive; default: --start)
--conf            no       YAML configuration file (see config.example.yaml)
===============  ========  =====================================================

.. * ``--start`` can be omitted only if a window is provided by ``--conf``.

Precedence is **CLI > ``--conf`` file > mission profile defaults**.

Report
~~~~~~

.. code-block:: bash

   metadata-asf report \
     --catalogdir ./out \
     --outputfile ./out/report.html \
     --log-verbosity INFO

===============  ========  =====================================================
Option           Required  Description
===============  ========  =====================================================
--catalogdir      yes      Directory of daily Parquet files to report on
--outputfile      no       HTML report path (default: catalog_report.html)
--mission         no       Mission label for the header (default: inferred)
--log-verbosity   no       DEBUG, INFO, WARNING or ERROR (default INFO)
===============  ========  =====================================================

The report is a single self-contained HTML file (inline CSS, no JavaScript, no
external assets) covering volume and completeness, the product/instrument mix, and
footprint geometry quality. It is read-only and makes no API calls. Figures (daily
volume, cumulative records, mix panels, footprint map, geometry issues) are rendered
server-side with matplotlib and inlined as base64 PNGs.

Python API
----------

.. code-block:: python

   from pathlib import Path
   import datetime as dt

   from metadata_asf import search, extract, export

   products = search.search(
       mission="NISAR",
       start=dt.date(2025, 1, 1),
       end=dt.date(2025, 1, 31),
       intersects_with=None,  # falls back to the profile's ocean WKT
       max_results=10_000,
   )

   df = extract.to_dataframe(products, mission="NISAR")
   written = export.write_daily_parquet(df, output_dir=Path("./out"))

Parquet schema
--------------

================  ==========  ===============================
Column            Type        Content
================  ==========  ===============================
granule_id        str         file_name / granuleName
platform          str         e.g. NISAR, SENTINEL-1A
geometry          str (WKT)   acquisition footprint
start_time        datetime    UTC
stop_time         datetime    UTC
polarization      list[str]   e.g. ["HH", "HV"]
beam_mode         str         acquisition mode
product_type      str         e.g. RSLC, GSLC
processing_level  str         L1, L2
================  ==========  ===============================

Files are written with the PyArrow engine, Snappy compression and no index;
every timestamp is UTC. Any schema change implies a major version bump plus
migration notes.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api
   contributing
