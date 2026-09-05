#!/usr/bin/env bash
set -euo pipefail

cd /root/private_data/ORA0_next
log=${RLT_LOG:-logs/rlt_residual_success50_seed7_pilot100_v4_launcher.log}
total=${RLT_TOTAL:-100}

summarize() {
  local pattern=$1 label=$2
  awk -v pattern="$pattern" -v label="$label" -v total="$total" '
    index($0, pattern) {
      n++
      hit[n] = 0
      for (i = 1; i <= NF; i++)
        if ($i ~ /^success=/) { split($i, a, "="); hit[n] = a[2]; wins += a[2] }
    }
    END {
      start = n - 19; if (start < 1) start = 1
      for (i = start; i <= n; i++) recent += hit[i]
      count = n ? n - start + 1 : 0
      printf "%s %d/%d  success %d/%d (%.1f%%)  recent20 %d/%d (%.1f%%)\n",
        label, n, total, wins, n, n ? 100*wins/n : 0,
        recent, count, count ? 100*recent/count : 0
    }
  ' "$log"
}

echo "RLT v4 seed7 pilot"
echo "prefill $(grep -c 'rlt recovery prefill task=' "$log" || true)/50"
summarize "rlt warmup episode=" "warmup"
summarize "rlt online episode=" "online "
grep 'rlt eval overall' "$log" | tail -n 1 || true
echo "log age $(( $(date +%s) - $(stat -c %Y "$log") ))s"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits |
  awk -F, '{printf "gpu%s  util%s%%  memory%s MiB\n", $1, $2, $3}'
