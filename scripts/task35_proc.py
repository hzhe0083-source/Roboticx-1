#!/usr/bin/env python
"""Exact /proc lookups for the task35 FM trainer and waiters.

pgrep -f matches the inspector's own command line. These helpers only accept
the real python trainer / named bash scripts.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


TRAINER_NEEDLE = "train.py --task35-precision-contract"
TRAINER_MARKERS = ("python", "train.py")
CONTINUE_20K_NEEDLE = "continue_task35_h6_to_20k.sh"
ARCHIVER_NEEDLE = "archive_task35_fm_milestones.sh"
INSPECTOR_MARKERS = (
    "pgrep",
    "python3 - <<",
    "needles =",
    "task35_proc.py",
    "check_task35_readiness.py",
)


def iter_cmdlines() -> list[tuple[int, str]]:
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        if cmd:
            rows.append((int(entry.name), cmd))
    return rows


def is_inspector(cmd: str) -> bool:
    return any(marker in cmd for marker in INSPECTOR_MARKERS)


def find_processes(needle: str, *, require: tuple[str, ...] = ()) -> list[dict]:
    matches = []
    for pid, cmd in iter_cmdlines():
        if needle not in cmd or is_inspector(cmd):
            continue
        if require and not all(token in cmd for token in require):
            continue
        elapsed_s = None
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().split()
            start_ticks = int(stat[21])
            hertz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            boot = float(Path("/proc/uptime").read_text().split()[0])
            elapsed_s = max(0.0, boot - start_ticks / hertz)
        except OSError:
            pass
        matches.append({"pid": pid, "cmd": cmd, "elapsed_s": elapsed_s})
    return matches


def trainer_processes() -> list[dict]:
    return find_processes(TRAINER_NEEDLE, require=TRAINER_MARKERS)


def trainer_alive() -> bool:
    return bool(trainer_processes())


def continue_20k_processes() -> list[dict]:
    return find_processes(CONTINUE_20K_NEEDLE)


def archiver_processes() -> list[dict]:
    return find_processes(ARCHIVER_NEEDLE)


def training_alive() -> bool:
    """True while the FM trainer or the 6k→20k exact-resume wrapper is live."""
    return trainer_alive() or bool(continue_20k_processes())


def pipeline_alive() -> bool:
    """True while training or the milestone archiver may still write archives."""
    return training_alive() or bool(archiver_processes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        choices=("trainer", "pipeline", "continue", "training", "archive"),
        default="trainer",
        help="process class to test",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check == "trainer":
        rows = trainer_processes()
    elif args.check == "continue":
        rows = continue_20k_processes()
    elif args.check == "archive":
        rows = archiver_processes()
    elif args.check == "training":
        rows = trainer_processes() + continue_20k_processes()
    elif args.check == "pipeline":
        rows = trainer_processes() + continue_20k_processes() + archiver_processes()
    else:
        raise SystemExit(f"unknown check {args.check}")
    for row in rows:
        print(f"{row['pid']}\t{row['cmd'][:200]}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
