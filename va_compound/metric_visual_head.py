"""Backward-compatible shim: see va_compound.vision.metric_visual_head."""
import sys as _sys
from .vision import metric_visual_head as _real
_sys.modules[__name__] = _real
