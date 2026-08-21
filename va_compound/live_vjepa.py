"""Backward-compatible shim: see va_compound.vision.live_vjepa."""
import sys as _sys
from .vision import live_vjepa as _real
_sys.modules[__name__] = _real
