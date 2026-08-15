"""Shadowmire package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shadowmire")
except PackageNotFoundError:
    __version__ = "2.0.0.dev0"

__all__ = ["__version__"]
