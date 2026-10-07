"""Smoke tests: the public API (AGENTS.md section 7) stays importable and coherent."""

from __future__ import annotations


def test_public_api_importable() -> None:
    """Every symbol of the public API resolves on import.

    This guards against accidental renames in ``__init__.py`` that would break
    downstream consumers such as fetch-asf before any behavior is tested. Note
    ``metadata_asf.search`` is a re-exported *module*: its entry point lives one
    attribute deeper (``metadata_asf.search.search``).
    """
    from metadata_asf import __version__, export, extract
    from metadata_asf.profiles import get_profile
    from metadata_asf.search import search as search_function

    assert callable(search_function)
    assert callable(extract.to_dataframe)
    assert callable(export.write_daily_parquet)
    assert callable(get_profile)
    assert isinstance(__version__, str)


def test_version_matches_packaging_import() -> None:
    """Re-raise version via normal import mechanism, even without build metadata."""
    import metadata_asf

    assert metadata_asf.__version__ is not None
