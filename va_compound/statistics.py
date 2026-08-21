"""Backward-compatible shim: see va_compound.utils.statistics."""
import sys as _sys
from .utils import statistics as _real
_sys.modules[__name__] = _real
