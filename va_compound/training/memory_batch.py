"""Batch compatible recurrent snapshots without mixing stream ownership."""
from dataclasses import fields

import torch


def memory_signature(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape[1:]), value.dtype, value.device)
    if isinstance(value, tuple):
        return tuple(memory_signature(x) for x in value)
    if hasattr(value, '__dataclass_fields__'):
        return (type(value), tuple((f.name, memory_signature(getattr(value, f.name))) for f in fields(value)))
    return value


def stack_memories(values):
    first = values[0]
    if first is None:
        return None
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, tuple):
        return tuple(stack_memories([v[i] for v in values]) for i in range(len(first)))
    if hasattr(first, '__dataclass_fields__'):
        return type(first)(**{f.name: stack_memories([getattr(v, f.name) for v in values]) for f in fields(first)})
    if any(v != first for v in values):
        raise ValueError('incompatible memory scalar')
    return first
