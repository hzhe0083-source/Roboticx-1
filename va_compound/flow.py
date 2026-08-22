"""Backward-compatible shim: see va_compound.utils.flow."""
import sys as _sys
from .utils import flow as _real
_sys.modules[__name__] = _real
