"""gridcast: GB grid carbon intensity forecasting service."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gridcast")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
