"""Backward-compatible shim: see va_compound.vision.backbones."""
import sys as _sys
from .vision import backbones as _real
_sys.modules[__name__] = _real
