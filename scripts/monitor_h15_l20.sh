#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
LOG=${1:-logs/mw_hard2_l20_h15_p15_prefix_tail_from_s1752.to_equiv_s5037.log}
[[ -f "$LOG" ]] || { echo "missing training log: $LOG" >&2; exit 1; }

# Print recent steps, then follow forever. awk's fflush keeps one compact line
# per optimizer step while the training process is still writing the log.
tail -n "${HISTORY:-5}" -F "$LOG" | awk -W interactive '
/^step=[0-9]+ mode=/ {
    step = task = fm = p2 = tail = world = gain = rel = guard = 0
    spread = 1
    for (i = 1; i <= NF; i++) {
        value = $i
        sub(/^[^=]*=/, "", value)
        if ($i ~ /^step=/) step = value
        else if ($i ~ /^flow=/) fm = value
        else if ($i ~ /^flow_first2=/) p2 = value
        else if ($i ~ /^flow_tail13=/) tail = value
        else if ($i ~ /^world_objective=/) world = value
        else if ($i ~ /^gain=/) gain = value
        else if ($i ~ /^rel=/) rel = value
        else if ($i ~ /^world_guard=/) guard = value
        else if ($i ~ /^world_task\[/) {
            task = $i
            sub(/^world_task\[/, "", task)
            sub(/:all=.*/, "", task)
        } else if ($i ~ /^stages=/) {
            sub(/\]$/, "", value)
            count = split(value, stages, ",")
            low = high = stages[1] + 0
            for (j = 2; j <= count; j++) {
                if (stages[j] < low) low = stages[j]
                if (stages[j] > high) high = stages[j]
            }
            if (low > 0) spread = high / low
        }
    }
    printf "step=%-5d %-14s FM=%.3f P2=%.3f T13=%.3f World=%.3f gain=%+.4f rel=%4.1f%% Sx=%.3f guard=%.6f\n", \
        step, task, fm, p2, tail, world, gain, 100 * rel, spread, guard
    fflush()
}'
