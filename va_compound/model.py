"""Backward-compatible shim: see va_compound.policy.model."""
import sys as _sys
from .policy import model as _real
_sys.modules[__name__] = _real
