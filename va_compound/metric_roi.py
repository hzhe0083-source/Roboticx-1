"""Backward-compatible shim: see va_compound.vision.metric_roi."""
import sys as _sys
from .vision import metric_roi as _real
_sys.modules[__name__] = _real
