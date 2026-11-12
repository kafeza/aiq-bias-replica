#!/bin/bash
# =====================================================================
# Score finished runs with the judge (score_runs.py --judge) on node 18.
# One job = one judge (gpt-oss-120b in Ollama) on 1 GPU, on 127.0.0.1 only.
# To go faster, submit the same job several times, each with its own PART:
# part k of n scores every n-th run (runs 1..N sorted by folder name), so the
# parts share the work evenly and never score the same run.
#
# Settings (given when starting the job):
#   JOBS   the run jobs to score, e.g. "592825 592826" (their folders runs/<job>_L*/)
#   PART   which part this job does (1..PARTS)       default 1
#   PARTS  into how many parts the runs are split    default 1
#   REDO   1 = score again runs that already have judge scores   default 0
# Examples, from the repo folder on Dhabi:
#   sbatch --export=ALL,JOBS="592825" dhabi/score_runs.sbatch                    # one judge
#   for k in 1 2 3; do sbatch --export=ALL,JOBS="592825 592826",PART=$k,PARTS=3 dhabi/score_runs.sbatch; done
# Results: scores.json in each run folder.   Follow it: tail -f score_runs_<job number>.log
# =====================================================================
#SBATCH --job-name=score_runs
# node 18 only, in the security team's reservation; one GPU
#SBATCH --nodelist=t01pdscgpu18
#SBATCH --reservation=security
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --nice=1000
#SBATCH --time=12:00:00
#SBATCH --output=score_runs_%j.log

cd "$SLURM_SUBMIT_DIR" || exit 1
source dhabi/check_rules.sh                          # rules 1 and 2
REPO="$PWD"
PART="${PART:-1}"; PARTS="${PARTS:-1}"; REDO="${REDO:-0}"
[ -n "$JOBS" ] || stop_job "no JOBS given, e.g. --export=ALL,JOBS=\"592825 592826\""
[ "$PART" -ge 1 ] && [ "$PART" -le "$PARTS" ] || stop_job "PART must be between 1 and PARTS ($PART of $PARTS)"

OLLAMA_BIN="$HOME/ollama/bin/ollama"
MODELS="$REPO/ollama-models"
for f in "$OLLAMA_BIN" .venv-rag/bin/python nvidia_cases.json \
         "$MODELS/manifests/registry.ollama.ai/library/gpt-oss/120b"; do
    [ -e "$f" ] || stop_job "missing $f"
done

# ---- the runs of this part ----------------------------------------------
ALL=()
for j in $JOBS; do
    for d in runs/${j}_L*/*/; do
        [ -f "$d/response.json" ] && ALL+=("${d%/}")
    done
done
[ "${#ALL[@]}" -gt 0 ] || stop_job "no finished runs found for jobs: $JOBS"
SORTED=(); while IFS= read -r x; do SORTED+=("$x"); done < <(printf '%s\n' "${ALL[@]}" | sort -t/ -k3); ALL=("${SORTED[@]}")   # by run folder name (start time)
MINE=()
for i in "${!ALL[@]}"; do
    [ $(( i % PARTS )) -eq $(( PART - 1 )) ] && MINE+=("${ALL[$i]}")
done
echo "jobs $JOBS: ${#ALL[@]} finished runs; part $PART of $PARTS scores ${#MINE[@]}"

# ---- the judge, on a free port of 127.0.0.1 only -------------------------
cleanup() { trap - EXIT TERM INT; kill $(_descendants $$) 2>/dev/null; sleep 2; kill -9 $(_descendants $$) 2>/dev/null; }
trap cleanup EXIT
trap 'echo "job cancelled or out of time: stopping the judge"; exit 1' TERM INT
P=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
mkdir -p "logs/score_runs"
OLLAMA_HOST="127.0.0.1:$P" OLLAMA_MODELS="$MODELS" OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=1 \
    "$OLLAMA_BIN" serve > "logs/score_runs/ollama_$SLURM_JOB_ID.log" 2>&1 &
for i in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$P/api/version" && break; sleep 1; done
curl -s -o /dev/null "http://127.0.0.1:$P/api/version" || stop_job "the judge did not start (see logs/score_runs/ollama_$SLURM_JOB_ID.log)"
# load the judge now, with the scorer's own context (num_ctx 16384 in score_runs.py),
# so the first question does not wait and the model is not loaded twice
curl -s -m 1800 -o /dev/null "http://127.0.0.1:$P/api/generate" \
    -d '{"model": "gpt-oss:120b", "keep_alive": -1, "options": {"num_ctx": 16384}}' \
    || stop_job "Ollama could not load gpt-oss:120b"
check_localhost_only                                 # rule 3, once the judge is up
( while sleep 300; do check_localhost_only > /dev/null || exit 1; done ) &   # and every 5 minutes
curl -s "http://127.0.0.1:$P/api/tags" | python3 -c '
import json, sys
for m in json.load(sys.stdin)["models"]:
    if m["name"].startswith("gpt-oss:120b"): print("judge:", m["name"], m["digest"][:12])'

# ---- score ------------------------------------------------------------------
export OLLAMA_URL="http://127.0.0.1:$P/api/chat"
n=0; done_=0; skipped=0; failed=0
for run in "${MINE[@]}"; do
    n=$((n + 1))
    if [ "$REDO" != "1" ] && [ -f "$run/scores.json" ] && grep -q '"obeyed"' "$run/scores.json"; then
        skipped=$((skipped + 1)); echo "[$n/${#MINE[@]}] $run: already judged, skipped"; continue
    fi
    echo "[$n/${#MINE[@]}] $run  ($(date +%H:%M))"
    if RUNS_DIR="$REPO/$(dirname "$run")" .venv-rag/bin/python score_runs.py --judge --run "$(basename "$run")" \
        > "$run/score_log.txt" 2>&1; then
        done_=$((done_ + 1))
        grep -E '^(skip|note) ' "$run/score_log.txt" | sed 's/^/    /'
    else
        failed=$((failed + 1)); echo "    FAILED (see $run/score_log.txt)"
    fi
done
echo; echo "PART $PART OF $PARTS DONE: scored $done_, already judged $skipped, failed $failed"
