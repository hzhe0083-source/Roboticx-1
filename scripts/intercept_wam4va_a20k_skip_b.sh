#!/usr/bin/env bash
# Intercept the v14 research20k runner after A/final's 20k held-out starts,
# so B/cycle does not launch unconditionally.
#
# Safe by construction:
#   - never signals a process group (no negative PID, no pkill -g)
#   - never STOPs until A 20k evaluator is live, or A 20k JSON is complete
#     and B has not started
#   - re-checks parent cmdline + starttime before every signal
#   - does not touch train.py / evaluator / runner source
set -euo pipefail

ROOT=/home/ryan/Documents/robot/ORA0
cd "$ROOT"

A_FAMILY=mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k
B_FAMILY=mw_hard2_wam4va_visualmotion_oraclestgapcycle_v14.research20k
A_CKPT="$ROOT/checkpoints/${A_FAMILY}.pt"
A_JSON="$ROOT/diagnostics/${A_FAMILY}.gate_step20000.json"
A_GATE_LOG="$ROOT/logs/${A_FAMILY}.gate_step20000.log"
A_TRAIN_LOG="$ROOT/logs/${A_FAMILY}.train_step1000_to_step20000.log"
STATE="$ROOT/diagnostics/intercept_wam4va_a20k_skip_b.state"
LOG="$ROOT/logs/intercept_wam4va_a20k_skip_b.log"
LOCK=/tmp/ora0_wam4va_a20k_intercept.lock
PY=/home/ryan/.venvs/openvla/bin/python

EXPECTED_PARENT_CMDLINE='bash scripts/run_mw_hard2_wam4va_visualmotion_gap_ab_v1.sh 20k'
HINT_PARENT_PID=${HINT_PARENT_PID:-3727416}
HINT_PARENT_STARTTIME=${HINT_PARENT_STARTTIME:-104112637}

mkdir -p "$ROOT/logs" "$ROOT/diagnostics"
exec >>"$LOG" 2>&1

ts() { date '+%F %T %z'; }
log() { printf '%s %s\n' "$(ts)" "$*"; }

exec 8>"$LOCK"
if ! flock -n 8; then
  log "EXIT another intercept watcher already holds $LOCK"
  exit 0
fi

read_cmdline() {
  local pid=$1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" | sed 's/[[:space:]]*$//'
}

read_starttime() {
  local pid=$1 stat rest
  [[ -r "/proc/$pid/stat" ]] || return 1
  stat=$(<"/proc/$pid/stat")
  rest=${stat##*) }
  set -- $rest
  printf '%s\n' "${20}"
}

read_state() {
  local pid=$1 stat rest
  [[ -r "/proc/$pid/stat" ]] || return 1
  stat=$(<"/proc/$pid/stat")
  rest=${stat##*) }
  set -- $rest
  printf '%s\n' "$1"
}

find_parent() {
  local pid cmd st
  if [[ -d "/proc/$HINT_PARENT_PID" ]]; then
    cmd=$(read_cmdline "$HINT_PARENT_PID" || true)
    st=$(read_starttime "$HINT_PARENT_PID" || true)
    if [[ "$cmd" == "$EXPECTED_PARENT_CMDLINE" && "$st" == "$HINT_PARENT_STARTTIME" ]]; then
      printf '%s\n' "$HINT_PARENT_PID"
      return 0
    fi
  fi
  local cand
  for cand in /proc/[0-9]*; do
    pid=${cand#/proc/}
    cmd=$(read_cmdline "$pid" 2>/dev/null || true)
    if [[ "$cmd" == "$EXPECTED_PARENT_CMDLINE" ]]; then
      printf '%s\n' "$pid"
      return 0
    fi
  done
  return 1
}

parent_ok() {
  local pid=$1 cmd st
  cmd=$(read_cmdline "$pid" || true)
  st=$(read_starttime "$pid" || true)
  [[ "$cmd" == "$EXPECTED_PARENT_CMDLINE" ]] || return 1
  if [[ -n "${BOUND_STARTTIME:-}" ]]; then
    [[ "$st" == "$BOUND_STARTTIME" ]] || return 1
  fi
  return 0
}

scan_procs() {
  "$PY" -B - "$A_FAMILY" "$B_FAMILY" "$A_CKPT" "$A_JSON" "$A_GATE_LOG" <<'PY'
import sys
from pathlib import Path

a_family, b_family, a_ckpt, a_json, a_gate_log = sys.argv[1:6]
a_ckpt_name = Path(a_ckpt).name
a_json_name = Path(a_json).name
a_gate_name = Path(a_gate_log).name
found = {"a_train": [], "b_train": [], "a_eval": [], "a_tee": []}

def flag_value(args, name):
    if name in args:
        idx = args.index(name)
        if idx + 1 < len(args):
            return args[idx + 1]
    return None

for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        raw = (entry / "cmdline").read_bytes()
    except OSError:
        continue
    args = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
    if not args:
        continue
    base = Path(args[0]).name
    tokens = [Path(arg).name for arg in args]
    is_python = base.startswith("python")
    if is_python and "train.py" in tokens:
        stage = flag_value(args, "--world-action-rank-stage")
        if stage == "final" and any(a_family in arg for arg in args):
            found["a_train"].append(entry.name)
        elif stage == "cycle" and any(b_family in arg for arg in args):
            found["b_train"].append(entry.name)
    elif is_python and "eval_wam4va_world_action.py" in tokens:
        ckpt = flag_value(args, "--checkpoint") or ""
        out = flag_value(args, "--output-json") or ""
        if (
            Path(ckpt).name == a_ckpt_name
            and Path(out).name == a_json_name
            and "oraclestgapfinal" in ckpt
            and "gate_step20000" in out
        ):
            found["a_eval"].append(entry.name)
    elif base == "tee" and any(Path(arg).name == a_gate_name for arg in args):
        found["a_tee"].append(entry.name)

for kind in ("a_train", "b_train", "a_eval", "a_tee"):
    print(kind + "=" + ",".join(found[kind]))
PY
}

json_complete() {
  [[ -s "$A_JSON" ]] || return 1
  "$PY" -B - "$A_JSON" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    report = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
gate = report.get("gate") or {}
ckpt = report.get("checkpoint") or {}
ok = (
    report.get("contract") == "wam4va_world_action_heldout_v1"
    and gate.get("decision") in {"GO", "NO-GO"}
    and isinstance(gate.get("passed"), bool)
    and int(ckpt.get("global_step", -1)) == 20000
    and str(ckpt.get("path", "")).endswith(
        "mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.pt"
    )
    and isinstance(ckpt.get("sha256"), str)
    and len(ckpt["sha256"]) == 64
)
raise SystemExit(0 if ok else 1)
PY
}

write_state() {
  printf '%s\n' "$*" >"$STATE"
}

stop_parent_only() {
  local parent=$1 state cmd st
  parent_ok "$parent" || { log "REFUSE stop: parent identity mismatch pid=$parent"; return 1; }
  cmd=$(read_cmdline "$parent")
  st=$(read_starttime "$parent")
  state=$(read_state "$parent")
  if [[ "$state" == T ]]; then
    log "parent already STOPPED pid=$parent starttime=$st"
    return 0
  fi
  log "STOP parent pid=$parent starttime=$st cmdline=$cmd"
  kill -STOP "$parent"
  sleep 0.2
  state=$(read_state "$parent" || true)
  if [[ "$state" != T ]]; then
    log "ERROR parent not T after STOP state=$state"
    return 1
  fi
  local child state_c scan eval_ids tee_ids
  scan=$(scan_procs)
  eval_ids=$(printf '%s\n' "$scan" | sed -n 's/^a_eval=//p' | tr ',' ' ')
  tee_ids=$(printf '%s\n' "$scan" | sed -n 's/^a_tee=//p' | tr ',' ' ')
  for child in $eval_ids $tee_ids; do
    [[ -n "$child" ]] || continue
    state_c=$(read_state "$child" || true)
    if [[ "$state_c" == T ]]; then
      log "ERROR child $child also STOPPED; CONT parent to recover"
      kill -CONT "$parent" || true
      return 1
    fi
    log "child still running pid=$child state=$state_c"
  done
  write_state "STOPPED parent=$parent starttime=$st at=$(ts)"
  return 0
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

decide_after_eval() {
  local parent=$1 decision passed step sha_json sha_disk
  if ! json_complete; then
    log "BAD_JSON after evaluator exit; CONT parent so runner can fail closed"
    if parent_ok "$parent" && [[ "$(read_state "$parent")" == T ]]; then
      kill -CONT "$parent"
      write_state "CONT_BAD_JSON parent=$parent at=$(ts)"
    fi
    return 2
  fi
  decision=$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate"]["decision"])' "$A_JSON")
  passed=$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate"]["passed"])' "$A_JSON")
  step=$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["global_step"])' "$A_JSON")
  sha_json=$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["sha256"])' "$A_JSON")
  sha_disk=$(sha256_file "$A_CKPT")
  log "REPORT decision=$decision passed=$passed step=$step json_sha=$sha_json disk_sha=$sha_disk"
  if [[ ! -s "$A_GATE_LOG" ]] || ! rg -q 'GO: report written|NO-GO: report written' "$A_GATE_LOG"; then
    log "BAD_LOG missing written marker; CONT parent"
    if parent_ok "$parent" && [[ "$(read_state "$parent")" == T ]]; then
      kill -CONT "$parent"
      write_state "CONT_BAD_LOG parent=$parent at=$(ts)"
    fi
    return 2
  fi
  if [[ "$sha_json" != "$sha_disk" || "$step" != 20000 ]]; then
    log "BAD_SHA_OR_STEP; CONT parent"
    if parent_ok "$parent" && [[ "$(read_state "$parent")" == T ]]; then
      kill -CONT "$parent"
      write_state "CONT_BAD_SHA parent=$parent at=$(ts)"
    fi
    return 2
  fi
  if [[ "$decision" == GO && "$passed" == True ]]; then
    log "GO: KILL parent pid=$parent to skip B"
    if parent_ok "$parent"; then
      kill -KILL "$parent"
    fi
    sleep 1
    if [[ -d "/proc/$parent" ]]; then
      log "ERROR parent still alive after KILL"
      return 3
    fi
    local b
    b=$(printf '%s\n' "$(scan_procs)" | sed -n 's/^b_train=//p')
    if [[ -n "${b:-}" ]]; then
      log "ERROR cycle trainer appeared after GO kill: $b"
      return 3
    fi
    write_state "KILLED_PARENT_GO parent=$parent at=$(ts) sha=$sha_disk"
    log "DONE skipped B after A GO"
    return 0
  fi
  if [[ "$decision" == NO-GO && "$passed" == False ]]; then
    log "NO-GO: CONT parent so B can start from scratch"
    if parent_ok "$parent" && [[ "$(read_state "$parent")" == T ]]; then
      kill -CONT "$parent"
    elif parent_ok "$parent"; then
      log "parent not STOPPED; leaving it running for B"
    fi
    write_state "CONT_NOGO parent=$parent at=$(ts) sha=$sha_disk"
    log "DONE released parent after A NO-GO"
    return 0
  fi
  log "AMBIGUOUS gate decision=$decision passed=$passed; CONT parent"
  if parent_ok "$parent" && [[ "$(read_state "$parent")" == T ]]; then
    kill -CONT "$parent"
  fi
  write_state "CONT_AMBIGUOUS parent=$parent at=$(ts)"
  return 2
}

wait_eval_exit() {
  local eval_pid=$1 tee_pid=${2:-}
  log "waiting for evaluator pid=$eval_pid tee=${tee_pid:-none}"
  while [[ -d "/proc/$eval_pid" ]]; do
    local st
    st=$(read_state "$eval_pid" || true)
    if [[ "$st" == Z ]]; then
      break
    fi
    sleep 5
  done
  if [[ -n "$tee_pid" ]]; then
    while [[ -d "/proc/$tee_pid" ]]; do
      local st
      st=$(read_state "$tee_pid" || true)
      if [[ "$st" == Z ]]; then
        break
      fi
      sleep 2
    done
  fi
  # JSON is opened with 'x' then dumped in place. Wait until parseable.
  local i
  for i in $(seq 1 60); do
    if json_complete; then
      return 0
    fi
    sleep 2
  done
  return 1
}

log "START intercept watcher pid=$$ pgid=$(ps -o pgid= -p $$ | tr -d ' ') sid=$(ps -o sid= -p $$ | tr -d ' ')"
PARENT=$(find_parent || true)
if [[ -z "${PARENT:-}" ]]; then
  log "WARN no live 20k parent yet; will keep scanning"
else
  BOUND_STARTTIME=$(read_starttime "$PARENT")
  log "bound parent=$PARENT starttime=$BOUND_STARTTIME state=$(read_state "$PARENT") cmdline=$(read_cmdline "$PARENT")"
  if [[ "$PARENT" == "$HINT_PARENT_PID" && "$BOUND_STARTTIME" != "$HINT_PARENT_STARTTIME" ]]; then
    log "REFUSE unexpected starttime change on hinted PID"
    exit 4
  fi
fi

write_state "WATCHING parent=${PARENT:-none} starttime=${BOUND_STARTTIME:-none} at=$(ts)"

while true; do
  PARENT=$(find_parent || true)
  SCAN=$(scan_procs)
  A_TRAIN=$(printf '%s\n' "$SCAN" | sed -n 's/^a_train=//p' | awk -F, '{print $1}')
  B_TRAIN=$(printf '%s\n' "$SCAN" | sed -n 's/^b_train=//p' | awk -F, '{print $1}')
  A_EVAL=$(printf '%s\n' "$SCAN" | sed -n 's/^a_eval=//p' | awk -F, '{print $1}')
  A_TEE=$(printf '%s\n' "$SCAN" | sed -n 's/^a_tee=//p' | awk -F, '{print $1}')

  if [[ -n "${B_TRAIN:-}" ]]; then
    if json_complete; then
      decision=$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate"]["decision"])' "$A_JSON")
      if [[ "$decision" == GO && -z "${A_TRAIN:-}" ]]; then
        log "LATE GO: verified cycle trainer $B_TRAIN; TERM B then KILL parent"
        kill -TERM "$B_TRAIN" || true
        if [[ -n "${PARENT:-}" ]] && parent_ok "$PARENT"; then
          kill -KILL "$PARENT" || true
        fi
        write_state "LATE_KILL_B b=$B_TRAIN parent=${PARENT:-none} at=$(ts)"
        exit 0
      fi
      log "B already started after A $decision; intercept window closed"
      write_state "B_STARTED b=$B_TRAIN decision=$decision at=$(ts)"
      exit 0
    fi
    log "WARN ignoring unmatched-or-early cycle pid=$B_TRAIN until A 20k JSON exists"
  fi

  if [[ -z "${PARENT:-}" ]]; then
    if json_complete; then
      log "parent gone and A 20k JSON complete; nothing left to intercept"
      write_state "PARENT_GONE_JSON_READY at=$(ts)"
      exit 0
    fi
    log "parent missing; sleep and rescan"
    sleep 15
    continue
  fi

  if [[ -n "${A_EVAL:-}" ]]; then
    log "A 20k evaluator live pid=$A_EVAL tee=${A_TEE:-none} parent=$PARENT"
    if ! stop_parent_only "$PARENT"; then
      sleep 2
      continue
    fi
    wait_eval_exit "$A_EVAL" "${A_TEE:-}" || true
    decide_after_eval "$PARENT"
    exit $?
  fi

  if json_complete && [[ -z "${A_TRAIN:-}" ]]; then
    log "A 20k JSON already complete and trainer gone; STOP then decide"
    if ! stop_parent_only "$PARENT"; then
      sleep 2
      continue
    fi
    decide_after_eval "$PARENT"
    exit $?
  fi

  if [[ -n "${A_TRAIN:-}" ]]; then
    sleep 15
  else
    # trainer finished; evaluator should appear soon
    sleep 2
  fi
done
