"""Backward-compatible shim: see va_compound.policy.end_to_end."""
import sys as _sys
from .policy import end_to_end as _real
_sys.modules[__name__] = _real
