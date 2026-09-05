"""Backward-compatible shim: see va_compound.world.wmrm."""
import sys as _sys
from .world import wmrm as _real
_sys.modules[__name__] = _real
