"""Backward-compatible shim: see va_compound.world.world_contract."""
import sys as _sys
from .world import world_contract as _real
_sys.modules[__name__] = _real
