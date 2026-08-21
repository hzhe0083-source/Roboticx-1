"""Backward-compatible shim: see va_compound.utils.exact_resume."""
import sys as _sys
from .utils import exact_resume as _real
_sys.modules[__name__] = _real
