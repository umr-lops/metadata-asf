Contributing
============

Development
-----------

.. code-block:: bash

   make install          # editable install + dev dependencies
   make test             # pytest with branch coverage (>= 80 % required)
   make check-codestyle  # ruff + black --check
   make mypy             # mypy --strict on src/ and tests/
   make docs             # build the HTML documentation into docs/_build/html

Integration tests performing real ASF API calls are deselected by default; opt in
explicitly with:

.. code-block:: bash

   pytest -m integration

Adding a mission
----------------

A mission is fully described by a profile in ``src/metadata_asf/profiles/``.
Adding one amounts to:

1. Creating a module (e.g. ``profiles/sentinel1.py``) exposing a
   :class:`~metadata_asf.profiles.base.MissionProfile`.
2. Registering it in the :data:`metadata_asf.profiles.MISSIONS` dict.
3. Adding tests for the new profile.

The core code (search, extract, export) never changes.

Code conventions
----------------

- Style: ``black`` (line length 100) + ``ruff``.
- Annotations mandatory on all public functions; ``mypy --strict`` must pass.
- Logging via ``logger = logging.getLogger(__name__)`` per module; no ``print()``
  outside ``cli.py``.
- Docstrings in Google style, in English.
- All timestamps in UTC.

Boundaries
----------

This repository is strictly metadata-only. Anything requiring Earthdata
credentials or transferring more than a few MB belongs to `fetch-asf
<https://github.com/umr-lops/fetch-asf>`_.
