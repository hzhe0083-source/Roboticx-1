#!/usr/bin/env python
"""Read global_step from a task35 torch zip checkpoint without loading tensors.

Used by the archiver so a missing SHA never recopies a later live checkpoint
over a 6k/9k/12k filename. CPU-only; does not touch the GPU.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

BININT1 = 0x4B
BININT = 0x4A
BININT2 = 0x4D
BINPUT = 0x71
LONG_BINPUT = 0x72
LONG1 = 0x8A


def peek_task35_checkpoint_step(path: Path) -> int:
    resolved = Path(path).expanduser().resolve(strict=True)
    with zipfile.ZipFile(resolved) as archive:
        names = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if not names:
            raise ValueError(f"checkpoint has no data.pkl: {resolved}")
        raw = archive.read(names[0])
    marker = b"global_step"
    index = raw.find(marker)
    if index < 0:
        raise ValueError(f"checkpoint pickle has no global_step: {resolved}")
    cursor = index + len(marker)
    while cursor < len(raw):
        opcode = raw[cursor]
        if opcode == BINPUT:
            cursor += 2
            continue
        if opcode == LONG_BINPUT:
            cursor += 5
            continue
        if opcode == BININT1:
            return int(raw[cursor + 1])
        if opcode == BININT2:
            return int.from_bytes(raw[cursor + 1 : cursor + 3], "little")
        if opcode == BININT:
            return int.from_bytes(raw[cursor + 1 : cursor + 5], "little", signed=True)
        if opcode == LONG1:
            nbytes = raw[cursor + 1]
            return int.from_bytes(
                raw[cursor + 2 : cursor + 2 + nbytes], "little", signed=True
            )
        raise ValueError(
            f"unexpected pickle opcode {opcode:#x} after global_step in {resolved}"
        )
    raise ValueError(f"truncated global_step in {resolved}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    print(peek_task35_checkpoint_step(parse_args().checkpoint))


if __name__ == "__main__":
    main()
