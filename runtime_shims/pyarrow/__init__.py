"""Disable optional pyarrow loading inside the movement runtime.

The project stores and reads movement evidence as CSV files. Apache Arrow is
therefore not required by this FastAPI phase. The audited macOS environment
segmented inside the installed pyarrow native extension while pandas was being
imported. Raising ``ModuleNotFoundError`` here makes pandas use its normal non-Arrow
path before the unsafe native extension can load.

This shim is placed first on ``PYTHONPATH`` only for FastAPI installation tests
and the isolated movement worker. It does not uninstall or modify pyarrow.
"""

raise ModuleNotFoundError(
    "pyarrow is intentionally disabled for the isolated FastAPI movement runtime."
)
