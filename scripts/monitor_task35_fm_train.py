#!/usr/bin/env python
"""Read-only monitor for the task35 FM H6 training run.

Parses the live training log, writes a human report plus JSON snapshot, and
appends one history row. This process never touches the trainer, GPU, or
checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_task35_fm_train import ARCHIVE_MILESTONES, summarize_task35_fm_log
from scripts.task35_proc import find_processes, trainer_processes

BJ = timezone(timedelta(hours=8))
STEP_RE = re.compile(
    r"^step=(?P<step>\d+)\s+mode=(?P<mode>\S+)\s+contract=(?P<contract>\S+)\s+"
    r"task=(?P<task>\S+)\s+.*?loss=(?P<loss>[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"
    r".*?grad=(?P<grad>[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"
)
AUX_RE = re.compile(r"aux_rmse=(?P<rmse>[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?)px")
SAVE_RE = re.compile(r"global_step=(?P<step>\d+)\s+periodic checkpoint saved")
TRAINER_NEEDLE = "train.py --task35-precision-contract"
DEFAULT_TOTAL_STEPS = 15000
WAITER_NEEDLES = {
    "archiver": "archive_task35_fm_milestones.sh",
    "tail": "tail -n 0 -F",
    "wait_6000": "wait_validate_task35_fm_milestone.sh 6000",
    "wait_9000": "wait_validate_task35_fm_milestone.sh 9000",
    "wait_12000": "wait_validate_task35_fm_milestone.sh 12000",
    "wait_15000": "wait_task35_fm_finished_eval.sh",
}
LOW_RAM_KB = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=root / "logs" / "task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1.log",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "logs" / "task35_fm_train_monitor.md",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=root / "logs" / "task35_fm_train_monitor.json",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=root / "logs" / "task35_fm_train_monitor.history.jsonl",
    )
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--trainer-needle", default=TRAINER_NEEDLE)
    return parser.parse_args()


def parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        return value
    return value


def trainer_alive(needle: str) -> dict:
    del needle
    matches = trainer_processes()
    return {
        "alive": bool(matches),
        "count": len(matches),
        "processes": matches,
    }


def mem_available_kb(meminfo: str | None = None) -> int | None:
    text = Path("/proc/meminfo").read_text() if meminfo is None else meminfo
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return None


def pipeline_health(
    *,
    trainer: dict,
    checkpoint_stem: Path,
    meminfo: str | None = None,
    gpu_compute_pids: list[int] | None = None,
) -> dict:
    """Read-only health for waiters/RAM/GPU. Never starts or kills jobs."""
    alerts: list[str] = []
    present = {
        name: bool(find_processes(needle))
        for name, needle in WAITER_NEEDLES.items()
    }
    archived = {
        step: Path(f"{checkpoint_stem}_step{step}.pt").is_file()
        for step in ARCHIVE_MILESTONES
    }
    if trainer["alive"]:
        if trainer["count"] != 1:
            alerts.append(f"trainer_count_{trainer['count']}")
        if not present["archiver"] or not present["tail"]:
            alerts.append("archiver_missing")
        if not present["wait_15000"]:
            alerts.append("eval_waiter_missing")
        for step, key in ((6000, "wait_6000"), (9000, "wait_9000"), (12000, "wait_12000")):
            if not archived[step] and not present[key]:
                alerts.append(f"waiter_{step}_missing")
    available = mem_available_kb(meminfo)
    if available is not None and available < LOW_RAM_KB:
        alerts.append("low_ram")
    trainer_pids = {int(row["pid"]) for row in trainer.get("processes") or []}
    if gpu_compute_pids is None:
        gpu_compute_pids = []
        try:
            import subprocess

            raw = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
            )
            gpu_compute_pids = [
                int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()
            ]
        except (OSError, subprocess.CalledProcessError):
            gpu_compute_pids = []
    extra_gpu = sorted(pid for pid in gpu_compute_pids if pid not in trainer_pids)
    if trainer["alive"] and extra_gpu:
        alerts.append("gpu_not_exclusive")
    return {
        "waiters": present,
        "archived": archived,
        "mem_available_kb": available,
        "gpu_compute_pids": gpu_compute_pids,
        "alerts": alerts,
    }


def window_stats(values: list[float]) -> dict | None:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None
    return {
        "n": len(finite),
        "mean": float(mean(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
        "last": float(finite[-1]),
    }


def parse_log(path: Path, total_steps: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"training log missing: {path}")
    rows: list[dict] = []
    aux_rows: list[dict] = []
    saves: list[int] = []
    alerts: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            save = SAVE_RE.search(line)
            if save:
                saves.append(int(save.group("step")))
            if "OutOfMemoryError" in line or "CUDA out of memory" in line:
                alerts.append("oom")
            match = STEP_RE.match(line)
            if not match:
                continue
            loss = parse_float(match.group("loss"))
            grad = parse_float(match.group("grad"))
            row = {
                "step": int(match.group("step")),
                "mode": match.group("mode"),
                "contract": match.group("contract"),
                "task": match.group("task"),
                "loss": loss,
                "grad": grad,
            }
            aux = AUX_RE.search(line)
            if aux:
                row["aux_rmse_px"] = float(aux.group("rmse"))
                aux_rows.append(row)
            if not math.isfinite(loss) or not math.isfinite(grad):
                alerts.append(f"non_finite_step_{row['step']}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no training steps parsed from {path}")
    latest = rows[-1]
    losses = [row["loss"] for row in rows]
    snapshot = {
        "generated_at": datetime.now(BJ).isoformat(timespec="seconds"),
        "log": str(path.resolve()),
        "task": latest["task"],
        "mode": latest["mode"],
        "contract": latest["contract"],
        "latest_step": latest["step"],
        "total_steps": int(total_steps),
        "progress": latest["step"] / float(total_steps),
        "latest_loss": latest["loss"],
        "latest_grad": latest["grad"],
        "latest_aux_rmse_px": aux_rows[-1].get("aux_rmse_px") if aux_rows else None,
        "latest_aux_step": aux_rows[-1]["step"] if aux_rows else None,
        "windows": {
            "last_10": window_stats(losses[-10:]),
            "last_50": window_stats(losses[-50:]),
            "last_100": window_stats(losses[-100:]),
            "since_1001": window_stats(
                [row["loss"] for row in rows if row["step"] >= 1001]
            ),
        },
        "aux_last_10": window_stats(
            [row["aux_rmse_px"] for row in aux_rows[-10:] if "aux_rmse_px" in row]
        ),
        "checkpoints_saved": saves,
        "alerts": sorted(set(alerts)),
        "n_logged_steps": len(rows),
    }
    return snapshot


def eta(snapshot: dict, trainer: dict) -> dict:
    elapsed = None
    if trainer["processes"] and trainer["processes"][0]["elapsed_s"] is not None:
        elapsed = float(trainer["processes"][0]["elapsed_s"])
    step = int(snapshot["latest_step"])
    total = int(snapshot["total_steps"])
    remain = max(0, total - step)
    sec_per_step = (elapsed / step) if elapsed and step > 0 else None
    finish = None
    if sec_per_step is not None:
        finish = datetime.now(BJ) + timedelta(seconds=remain * sec_per_step)
    return {
        "elapsed_s": elapsed,
        "sec_per_step": sec_per_step,
        "remain_steps": remain,
        "eta": None if finish is None else finish.isoformat(timespec="minutes"),
    }


def render_md(snapshot: dict, trainer: dict, timing: dict) -> str:
    def fmt_window(name: str) -> str:
        stats = snapshot["windows"].get(name)
        if not stats:
            return f"- {name}: n/a"
        return (
            f"- {name}: mean={stats['mean']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f} n={stats['n']}"
        )

    aux = snapshot["aux_last_10"]
    aux_line = "n/a"
    if snapshot["latest_aux_rmse_px"] is not None:
        aux_line = (
            f"{snapshot['latest_aux_rmse_px']:.1f} px "
            f"(step {snapshot['latest_aux_step']})"
        )
        if aux:
            aux_line += f"; last10 mean={aux['mean']:.1f} px"
    status = "RUNNING" if trainer["alive"] else "NOT RUNNING"
    eta_line = "n/a"
    if timing["eta"] is not None and timing["sec_per_step"] is not None:
        eta_line = (
            f"{timing['eta']} "
            f"({timing['sec_per_step']:.2f} s/step, "
            f"{timing['remain_steps']} steps left)"
        )
    health = snapshot.get("pipeline") or {}
    alerts = ", ".join(snapshot["alerts"]) if snapshot["alerts"] else "none"
    saves = ", ".join(str(s) for s in snapshot["checkpoints_saved"]) or "none"
    waiters = health.get("waiters") or {}
    waiter_line = ", ".join(
        f"{name}={'up' if ok else 'down'}" for name, ok in waiters.items()
    ) or "n/a"
    ram_kb = health.get("mem_available_kb")
    ram_line = "n/a" if ram_kb is None else f"{ram_kb / (1024 * 1024):.1f} GiB"
    return "\n".join(
        [
            f"# task35 FM train monitor",
            "",
            f"- time: {snapshot['generated_at']}",
            f"- status: {status}",
            f"- task: {snapshot['task']}",
            f"- step: {snapshot['latest_step']} / {snapshot['total_steps']} "
            f"({100.0 * snapshot['progress']:.1f}%)",
            f"- latest loss/flow: {snapshot['latest_loss']:.6f}",
            f"- latest grad: {snapshot['latest_grad']:.6f}",
            f"- latest aux RMSE: {aux_line}",
            f"- ETA: {eta_line}",
            f"- periodic saves: {saves}",
            f"- waiters: {waiter_line}",
            f"- MemAvailable: {ram_line}",
            f"- alerts: {alerts}",
            "",
            "## loss windows",
            fmt_window("last_10"),
            fmt_window("last_50"),
            fmt_window("last_100"),
            fmt_window("since_1001"),
            "",
            "## milestones",
            *milestone_lines(snapshot.get("milestones") or []),
            "",
            "This is FM flow-matching loss on peg-insert-side-v3, not closed-loop success.",
            "",
        ]
    )


def milestone_lines(windows: list[dict]) -> list[str]:
    lines = []
    for item in windows:
        loss = item.get("loss") or {}
        aux = item.get("aux_rmse_px") or {}
        archived = item.get("archived")
        flag = ""
        if archived is True:
            flag = " archived"
        elif archived is False:
            flag = " missing-archive"
        elif item.get("periodic_save_only"):
            flag = " periodic-save"
        aux_txt = "n/a" if not aux else f"{aux['mean']:.1f}px"
        if not loss:
            continue
        lines.append(
            f"- {item['start']}-{item['end']}: loss_mean={loss['mean']:.4f} "
            f"aux_mean={aux_txt}{flag}"
        )
    return lines or ["- n/a"]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    snapshot = parse_log(args.log, args.total_steps)
    summary = summarize_task35_fm_log(
        args.log.read_text(encoding="utf-8", errors="replace"),
        total_steps=args.total_steps,
        checkpoint_stem=args.log.parent.parent
        / "checkpoints"
        / "task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1",
    )
    snapshot["milestones"] = summary["windows"]
    trainer = trainer_alive(args.trainer_needle)
    timing = eta(snapshot, trainer)
    checkpoint_stem = (
        args.log.parent.parent
        / "checkpoints"
        / "task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1"
    )
    health = pipeline_health(trainer=trainer, checkpoint_stem=checkpoint_stem)
    snapshot["pipeline"] = health
    if health["alerts"]:
        snapshot["alerts"] = sorted(set(snapshot["alerts"] + health["alerts"]))
    payload = {
        "contract": "task35_fm_train_monitor_v1",
        **snapshot,
        "trainer": {
            "alive": trainer["alive"],
            "count": trainer["count"],
            "pids": [row["pid"] for row in trainer["processes"]],
            "elapsed_s": trainer["processes"][0]["elapsed_s"] if trainer["processes"] else None,
        },
        "timing": timing,
        "pipeline": health,
    }
    atomic_write(args.json, json.dumps(payload, indent=2) + "\n")
    atomic_write(args.report, render_md(snapshot, trainer, timing))
    try:
        import subprocess

        python = sys.executable
        for script in (
            ROOT / "scripts" / "list_task35_fm_candidates.py",
            ROOT / "scripts" / "report_task35_fm_status.py",
        ):
            subprocess.run(
                [python, "-B", str(script)],
                check=True,
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
            )
    except Exception as exc:  # cron must keep writing the live loss report
        print(f"status ledger refresh skipped: {exc}", flush=True)
    history_row = {
        "generated_at": payload["generated_at"],
        "step": payload["latest_step"],
        "loss": payload["latest_loss"],
        "grad": payload["latest_grad"],
        "aux_rmse_px": payload["latest_aux_rmse_px"],
        "alive": payload["trainer"]["alive"],
        "eta": timing["eta"],
    }
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_row) + "\n")
    print(render_md(snapshot, trainer, timing), end="")


if __name__ == "__main__":
    main()
