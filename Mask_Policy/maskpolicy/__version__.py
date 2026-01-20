#!/usr/bin/env python

"""To enable `maskpolicy.__version__`"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("maskpolicy")
except PackageNotFoundError:
    __version__ = "0.1.0"

