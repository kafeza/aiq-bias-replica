#!/bin/bash
# =====================================================================
# Security rules for EVERY job on Dhabi. Used by each job file:
#     source dhabi/check_rules.sh        # rules 1 and 2: stops the job if broken
#     ... start the servers ...
#     check_localhost_only               # rule 3: stops the job if broken
#
# Rule 1  runs only on node 18 (t01pdscgpu18), and only inside a SLURM job
# Rule 2  GPU limit: at most 8 GPUs until Sunday 15 Nov 2026 (whole node
#         reserved), at most 2 GPUs after that. The limit follows the date
#         and cannot be raised by a setting; MAX_GPUS can only lower it.
# Rule 3  nobody outside node 18 can connect: every port opened by this job
#         listens on 127.0.0.1 (or ::1) only
# =====================================================================

NODE="t01pdscgpu18"
RESERVED_UNTIL=20261115          # last day of the whole-node reservation

stop_job() {
    echo "STOP: $*" >&2
    # stop everything this job started, then end the job
    pkill -P $$ 2>/dev/null
    exit 1
}

# ---- Rule 1: node 18, inside SLURM ----------------------------------
[ "$(hostname -s)" = "$NODE" ] || stop_job "rule 1: this is $(hostname -s), not $NODE"
[ -n "$SLURM_JOB_ID" ]         || stop_job "rule 1: not started through SLURM (use sbatch)"

# ---- Rule 2: GPU limit ----------------------------------------------
if [ "$(date +%Y%m%d)" -le "$RESERVED_UNTIL" ]; then GPU_LIMIT=8; else GPU_LIMIT=2; fi
if [ -n "$MAX_GPUS" ] && [ "$MAX_GPUS" -lt "$GPU_LIMIT" ]; then GPU_LIMIT="$MAX_GPUS"; fi
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
if [ "$N_GPUS" -eq 0 ]; then
    # a job without GPU (e.g. downloads) must say so on purpose: NO_GPU_JOB=1
    [ "${NO_GPU_JOB:-0}" = "1" ] || stop_job "rule 2: SLURM gave this job no GPU (add #SBATCH --gres=gpu:N)"
fi
[ "$N_GPUS" -le "$GPU_LIMIT" ] || stop_job "rule 2: this job has $N_GPUS GPUs, the limit is $GPU_LIMIT"
# what the job can physically see must not exceed what it was given
N_VISIBLE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
[ "$N_VISIBLE" -le "$N_GPUS" ] || stop_job "rule 2: the job can see $N_VISIBLE GPUs but was given $N_GPUS"

echo "OK rules 1+2: node $(hostname -s), job $SLURM_JOB_ID, $N_GPUS GPU(s) [${CUDA_VISIBLE_DEVICES:-none}], visible $N_VISIBLE, limit $GPU_LIMIT"

# ---- Rule 3: call after starting the servers --------------------------
# Lists every listening port that belongs to a process of THIS job and
# stops the job if any listens on something other than localhost.
_descendants() { local p; for p in $(pgrep -P "$1"); do echo "$p"; _descendants "$p"; done; }

check_localhost_only() {
    local pids bad=0 line addr pid
    CHECKED_PORTS=0
    # this job's processes: SLURM's list, plus everything started by this script
    pids=" $(scontrol listpids "$SLURM_JOB_ID" 2>/dev/null | awk 'NR>1 {print $1}' | tr '\n' ' ') $$ $(_descendants $$ | tr '\n' ' ') "
    while read -r line; do
        addr=$(echo "$line" | awk '{print $4}')
        for pid in $(echo "$line" | grep -o 'pid=[0-9]*' | cut -d= -f2); do
            case "$pids" in *" $pid "*) ;; *) continue ;; esac
            CHECKED_PORTS=$((CHECKED_PORTS + 1))
            case "$addr" in
                127.0.0.1:*|"[::1]:"*|"[::ffff:127.0.0.1]:"*) echo "  ok   $addr  (pid $pid)" ;;
                *) echo "  OPEN $addr  (pid $pid)  <-- not localhost" >&2; bad=1 ;;
            esac
        done
    done < <(ss -Htlnp 2>/dev/null)
    [ "$bad" -eq 0 ] || stop_job "rule 3: a port of this job is open to other networks"
    echo "OK rule 3: all $CHECKED_PORTS port(s) of this job listen on localhost only"
}
