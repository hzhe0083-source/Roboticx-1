"""Episode-owned detached memory, committed only with an optimizer update."""
from __future__ import annotations

from dataclasses import fields

import torch

from va_compound.policy.model import VisualMemory
from va_compound.world.wmrm import WAMState


def _pack(memory):
    values = {}
    for field in fields(VisualMemory):
        value = getattr(memory, field.name)
        if field.name == "world_state" and value is not None:
            value = {f.name: None if getattr(value, f.name) is None else getattr(value, f.name).detach().cpu() for f in fields(WAMState)}
        elif isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        elif isinstance(value, tuple):
            value = tuple(x.detach().cpu() for x in value)
        values[field.name] = value
    return values


def _unpack(values):
    values = dict(values)
    if values["world_state"] is not None:
        values["world_state"] = WAMState(**values["world_state"])
    return VisualMemory(**values)


class EpisodeMemoryBank:
    def __init__(self):
        self.entries = {}
        self.pending = {}

    def begin(self, stream, episode, start, is_start, *, device, dtype):
        if stream in self.pending:
            raise ValueError("episode stream was forwarded twice before commit")
        prior = self.entries.get(stream)
        if is_start:
            if start != 0 or prior is not None:
                raise ValueError("episode start would overwrite live memory")
            return None
        if prior is None or prior[:2] != (episode, start):
            raise ValueError("episode memory is missing or decision sequence is discontinuous")
        return prior[2].to(device=device, dtype=dtype)

    def finish(self, stream, episode, next_start, is_end, memory):
        self.pending[stream] = None if is_end else (episode, next_start, memory.detach())

    def commit(self):
        for stream, entry in self.pending.items():
            if entry is None:
                self.entries.pop(stream, None)
            else:
                self.entries[stream] = entry
        self.pending.clear()

    def state_dict(self):
        if self.pending:
            raise ValueError("checkpoint requires committed episode memory")
        return {"contract": "episode_tbptt8_v1", "entries": {
            stream: {"episode": episode, "next_start": start, "memory": _pack(memory)}
            for stream, (episode, start, memory) in self.entries.items()
        }}

    def load_state_dict(self, state):
        if state.get("contract") != "episode_tbptt8_v1":
            raise ValueError("episode memory contract mismatch")
        self.entries = {int(stream): (int(entry["episode"]), int(entry["next_start"]), _unpack(entry["memory"]).detach())
                        for stream, entry in state["entries"].items()}
        self.pending.clear()
