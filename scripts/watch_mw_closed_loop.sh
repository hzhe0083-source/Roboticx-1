#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
shopt -s nullglob

tag="${1:-taskpar10_seed4042_h15_v1}"
interval="${INTERVAL:-5}"
trap 'exit 0' INT TERM

while true; do
    clear
    date '+%F %T %Z'
    logs=(logs/*"${tag}"_k*.log)

    if ((${#logs[@]} == 0)); then
        echo "waiting for logs matching tag: ${tag}"
    else
        grep -h '^trial task=' "${logs[@]}" 2>/dev/null | awk '
            {
                split($2, parsed_task, "=")
                split($5, success, "=")
                id = parsed_task[2]
                total++
                wins += success[2]
                count[id]++
                task_wins[id] += success[2]
            }
            END {
                complete_tasks = 0
                complete_wins = 0
                for (id in count) {
                    if (count[id] == 10) {
                        complete_tasks++
                        complete_wins += task_wins[id]
                    }
                }
                printf "episodes: %d/500 | wins: %d | live: %.1f%%\n", total, wins, total ? 100 * wins / total : 0
                printf "complete tasks: %d/50 | rate: %.1f%%\n", complete_tasks, complete_tasks ? 10 * complete_wins / complete_tasks : 0
                print "note: early live rate is biased high because successes finish sooner"
                print ""
                print "active/finished task rows:"
                for (id in count)
                    printf "task %02d: %d/%d\n", id, task_wins[id], count[id]
            }
        ' | sort -V
    fi

    echo
    printf 'workers: '
    pgrep -fc '[e]val_metaworld.py' || true
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format='csv,noheader,nounits' 2>/dev/null | awk -F, '{printf "GPU %s: %s MiB, util %s%%\n", $1, $2, $3}'

    errors=$(grep -H -E 'Traceback|Error|out of memory|Killed' "${logs[@]}" 2>/dev/null || true)
    if [[ -n "$errors" ]]; then
        echo
        echo "ERRORS:"
        echo "$errors"
    fi

    final=$(grep -E 'CLOSED-LOOP SUCCESS|EVOMIND FOUR-TIER AVERAGE|wall clock' "logs/${tag}_launcher.log" 2>/dev/null || true)
    if [[ -n "$final" ]]; then
        echo
        echo "$final"
    fi

    sleep "$interval"
done
