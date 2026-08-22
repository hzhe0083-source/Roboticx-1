"""Backward-compatible shim: see va_compound.vision.longtraj_frames."""
import sys as _sys
from .vision import longtraj_frames as _real
_sys.modules[__name__] = _real
