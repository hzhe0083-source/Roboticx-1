"""Backward-compatible shim: see va_compound.control.servo."""
import sys as _sys
from .control import servo as _real
_sys.modules[__name__] = _real
