"""Backward-compatible shim: see va_compound.control.local_control_slots."""
import sys as _sys
from .control import local_control_slots as _real
_sys.modules[__name__] = _real
