"""Backward-compatible shim: see va_compound.vision.fovea."""
import sys as _sys
from .vision import fovea as _real
_sys.modules[__name__] = _real
