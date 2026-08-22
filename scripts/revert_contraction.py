"""把 fixv1 的前向收缩精确还原为原始的纯加法残差。

判定实验（同一 checkpoint、同样 seed、只换前向）测得收缩把闭环成功率从
3/20=15% 打到 1/20=5%，而它要治的长 horizon NaN 早已被评测侧的
``--world-reset-every 4`` 解决（默认值即 4，零训练代价的分布对齐）。
straight-through 版本的前向与 fixv1 逐位相同，因此同样继承这个损伤。

本脚本只改 4 处前向表达式，不动任何 nn.Parameter/buffer，checkpoint 的
strict load 不受影响。每处都要求命中恰好一次，否则报错退出。
"""
from __future__ import annotations

import sys
from pathlib import Path

# (描述, 原文, 替换) —— 原文取自收缩版，替换取自收缩前的纯加法残差。
EDITS: list[tuple[str, str, str]] = [
    (
        "world_map stage0 anchor (real DINO last frame)",
        "            return base + _MAP_DELTA_GAIN * delta",
        "            return base + delta",
    ),
    (
        "world_map recurrent stage",
        "        return _MAP_RETENTION * base + _MAP_DELTA_GAIN * delta",
        "        return base + delta",
    ),
    (
        "belief write from innovation",
        "        belief = (\n"
        "            _BELIEF_RETENTION * belief\n"
        "            + (1.0 - _BELIEF_RETENTION) * belief_update\n"
        "        )",
        "        belief = belief + belief_update",
    ),
    (
        "belief write from world tokens",
        "            belief = (\n"
        "                _BELIEF_RETENTION * belief\n"
        "                + (1.0 - _BELIEF_RETENTION) * belief_update\n"
        "            )",
        "            belief = belief + belief_update",
    ),
]


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "va_compound/wmrm.py")
    source = target.read_text(encoding="utf-8")
    for label, old, new in EDITS:
        found = source.count(old)
        if found != 1:
            print(
                f"FAIL {label}: expected exactly 1 occurrence, found {found}",
                file=sys.stderr,
            )
            return 1
        source = source.replace(old, new)
        print(f"reverted: {label}")
    target.write_text(source, encoding="utf-8")
    leftover = [
        name
        for name in ("_BELIEF_RETENTION *", "_MAP_RETENTION *", "_MAP_DELTA_GAIN *")
        if name in source
    ]
    if leftover:
        print(f"FAIL: contraction constants still applied: {leftover}", file=sys.stderr)
        return 1
    print(f"OK {target}: forward is pure additive residual again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
