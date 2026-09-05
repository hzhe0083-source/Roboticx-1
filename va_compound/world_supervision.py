"""Backward-compatible shim: see va_compound.world.world_supervision."""
import sys as _sys
from .world import world_supervision as _real
_sys.modules[__name__] = _real
