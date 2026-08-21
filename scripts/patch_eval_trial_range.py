"""给服务器那份已分叉的 eval_metaworld.py 加上 --trial-range 分片支持。

本地仓库是新代码（子包布局），服务器那份是未入 git 的旧副本，无法直接同步整树。
本脚本只做 3 处等价插入，每处都要求锚点唯一命中，否则报错退出。

分片的正确性依据：每个 trial 开头都用 evaluation_episode_seed(task, trial) 重新
播种 numpy/env/torch，跨 trial 无 RNG 依赖，因此分片结果与串行逐 trial 完全一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

HELPER = '''

def parse_trial_range(spec: str | None, trials_per_task: int) -> tuple[int, int]:
    """Parse ``START:END`` into a half-open trial slice for sharded evaluation.

    Seeds come from ``evaluation_episode_seed(task, trial)`` alone, so a shard
    reproduces exactly the trials a serial run would produce for those indices.
    """
    if spec is None:
        return 0, int(trials_per_task)
    text = spec.strip()
    if text.count(":") != 1:
        raise ValueError(f"--trial-range must be START:END, got {spec!r}")
    start_text, stop_text = (part.strip() for part in text.split(":"))
    start = 0 if not start_text else int(start_text)
    stop = int(trials_per_task) if not stop_text else int(stop_text)
    if not 0 <= start < stop <= int(trials_per_task):
        raise ValueError(
            f"--trial-range {spec!r} must satisfy "
            f"0 <= start < stop <= --trials-per-task ({trials_per_task})"
        )
    return start, stop
'''

ARG = '''    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument(
        "--trial-range",
        type=str,
        default=None,
        help="分片评测的 trial 半开区间 START:END（缺省 = 全部）；"
        "跨 trial 无 RNG 依赖，分片结果与串行逐 trial 完全一致",
    )'''

SETUP = '''    completed_trials = 0
    trial_start, trial_stop = parse_trial_range(
        args.trial_range, args.trials_per_task
    )
    if (trial_start, trial_stop) != (0, args.trials_per_task):
        print(
            f"eval: trial shard [{trial_start}, {trial_stop}) of "
            f"{args.trials_per_task} per task",
            flush=True,
        )'''

EDITS: list[tuple[str, str, str]] = [
    (
        "parse_trial_range helper",
        '    return 1000 * int(global_task_id) + int(trial)',
        '    return 1000 * int(global_task_id) + int(trial)\n' + HELPER.rstrip("\n"),
    ),
    (
        "--trial-range argument",
        '    parser.add_argument("--trials-per-task", type=int, default=10)',
        ARG,
    ),
    (
        "shard bounds setup",
        '    completed_trials = 0',
        SETUP,
    ),
    (
        "sharded trial loop",
        '        for trial in range(args.trials_per_task):',
        '        for trial in range(trial_start, trial_stop):',
    ),
]


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_metaworld.py")
    source = target.read_text(encoding="utf-8")
    if "parse_trial_range" in source:
        print(f"{target}: already patched, nothing to do")
        return 0
    for label, old, new in EDITS:
        found = source.count(old)
        if found != 1:
            print(
                f"FAIL {label}: expected exactly 1 anchor, found {found}",
                file=sys.stderr,
            )
            return 1
        source = source.replace(old, new)
        print(f"patched: {label}")
    target.write_text(source, encoding="utf-8")
    print(f"OK {target}: --trial-range available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
