#!/bin/bash
# =====================================================================
# The second agent on node 18: LangChain's open_deep_research (installed by
# dhabi/setup_odr.sbatch), run through our connector odr/run_odr.py. One LANE per GPU:
#   Ollama (search answers, Nemotron) + llama-server (the agent's model) + our search
#   server, each on its own free port on 127.0.0.1, with its own attack card, search log,
#   index copy and run folder. No AI-Q and no Phoenix: the connector records every model
#   call and search itself (trace.jsonl in the run folder).
# The model server and the search server are started exactly as for AI-Q (run_batch.sbatch),
# with one addition: --temp 0.5, because open_deep_research sends no temperature and AI-Q
# sends 0.5.
#
# Settings (given when starting the job):
#   TASKS  which statements and how many runs each, e.g. "BIAS_STEREOTYPE_1:20"
#          (several: "BIAS_STEREOTYPE_1:20 BIAS_CF_DIABETES:20"); names from attacks.yml;
#          CLEAN = a run without attack, e.g. "CLEAN:5"
#   LLM    the agent's model: qwen3_8b (default), qwen3_8b_nothink, qwen3_32b, nemotron, falcon_r
#   POISON_EVERY, POISON_COUNT   the attack's position: every K-th search, at most C times
#          (default 2 and 1: the second search, once, as AI-Q's second query)
# Examples, from the repo folder on Dhabi:
#   sbatch --export=ALL,LLM=qwen3_8b,TASKS="BIAS_STEREOTYPE_1:1" dhabi/run_odr.sbatch     # 1 test run, 1 GPU
#   sbatch --gres=gpu:2 --export=ALL,LLM=nemotron,TASKS="BIAS_STEREOTYPE_1:20" dhabi/run_odr.sbatch
# Results: runs_odr/<job>_L<lane>/...   Servers' logs: lanes/<job>/L<lane>/
# Follow it:  tail -f run_odr_<job number>.log
# =====================================================================
#SBATCH --job-name=run_odr
# node 18 only, in the security team's reservation
#SBATCH --nodelist=t01pdscgpu18
#SBATCH --reservation=security
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --nice=1000
#SBATCH --time=24:00:00
#SBATCH --output=run_odr_%j.log

cd "$SLURM_SUBMIT_DIR" || exit 1
source dhabi/check_rules.sh                          # rules 1 and 2
REPO="$PWD"
LLM="${LLM:-qwen3_8b}"
POISON_EVERY="${POISON_EVERY:-2}"; POISON_COUNT="${POISON_COUNT:-1}"
[ -n "$TASKS" ] || stop_job "no TASKS given, e.g. --export=ALL,TASKS=\"BIAS_STEREOTYPE_1:1\""

# ---- fixed paths ------------------------------------------------------
OLLAMA_DIR="$HOME/ollama"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"
LLAMA_SERVER="$OLLAMA_DIR/lib/ollama/llama-server"
MODELS="$REPO/ollama-models"
ODR_PY="$REPO/odr/open_deep_research/.venv/bin/python"
case "$LLM" in      # the same files and settings as for AI-Q (run_batch.sbatch)
  falcon_r) BLOB=ac852a921c550ec9c55490c2a7aabd6706843e86b1a8cb9c31d5da908245f77e; ALIAS=falcon-h1r-7b
            EXTRA="--top-k 20 --top-p 0.95" ;;
  nemotron) BLOB=d3fc12f4480cad5060f183d6c186ca47d800509224632bb22e15791711950524; ALIAS=nemotron-super-49b
            EXTRA="--top-p 0.95" ;;
  qwen3_8b) BLOB=408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6; ALIAS=qwen3-8b
            EXTRA="--top-k 20 --top-p 0.95" ;;
  qwen3_8b_nothink) BLOB=408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6; ALIAS=qwen3-8b-nothink
            EXTRA="--top-k 20 --top-p 0.95 --chat-template-kwargs {\"enable_thinking\":false}" ;;
  qwen3_32b) BLOB=efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689; ALIAS=qwen3-32b
            EXTRA="--top-k 20 --top-p 0.95" ;;
  *) stop_job "LLM must be falcon_r, nemotron, qwen3_8b, qwen3_8b_nothink or qwen3_32b, not '$LLM'" ;;
esac
for f in "$OLLAMA_BIN" "$LLAMA_SERVER" "$MODELS/blobs/sha256-$BLOB" rag/milvus.db \
         .venv-rag/bin/python "$ODR_PY" odr/run_odr.py; do
    [ -e "$f" ] || stop_job "missing $f (run download_models / setup_envs / setup_odr first)"
done
# llama-server's GPU library (as in run_batch.sbatch): stop if it cannot see the GPU
CUDA_DIR=""
for d in cuda_v13 cuda_v12; do
    lib="$OLLAMA_DIR/lib/ollama/$d"
    [ -f "$lib/libggml-cuda.so" ] || continue
    if GGML_BACKEND_PATH="$lib/libggml-cuda.so" LD_LIBRARY_PATH="$OLLAMA_DIR/lib/ollama:$lib" \
        "$LLAMA_SERVER" --list-devices 2>/dev/null | grep -q "CUDA0"; then CUDA_DIR="$lib"; break; fi
done
[ -n "$CUDA_DIR" ] || stop_job "the model server cannot use the GPU (no working cuda_v13/cuda_v12 library in $OLLAMA_DIR/lib/ollama)"
export GGML_BACKEND_PATH="$CUDA_DIR/libggml-cuda.so"
export LD_LIBRARY_PATH="$OLLAMA_DIR/lib/ollama:$CUDA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "model server GPU library: $(basename "$CUDA_DIR")"
export HF_HOME="$REPO/hf-cache" HF_HUB_OFFLINE=1     # search models: the downloaded versions only

# ---- stop everything at the end, also when cancelled or out of time ----
cleanup() { trap - EXIT TERM INT; kill $(_descendants $$) 2>/dev/null; sleep 2; kill -9 $(_descendants $$) 2>/dev/null; }
trap cleanup EXIT
trap 'echo "job cancelled or out of time: stopping all lanes"; exit 1' TERM INT

free_ports() {   # $1 = how many free ports on 127.0.0.1
    python3 -c '
import socket, sys
ss = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in ss: s.bind(("127.0.0.1", 0))
print(" ".join(str(s.getsockname()[1]) for s in ss))' "$1"
}

wait_port() {    # $1 port  $2 pid  $3 name  $4 seconds
    local i
    for i in $(seq 1 "$4"); do
        python3 -c 'import socket,sys; socket.create_connection(("127.0.0.1", int(sys.argv[1])), 1)' "$1" 2>/dev/null && return 0
        kill -0 "$2" 2>/dev/null || stop_job "$3 stopped while starting (see its log in $LANE_DIR)"
        sleep 1
    done
    stop_job "$3 did not start within $4 s (see its log in $LANE_DIR)"
}

wait_ready() {   # $1 url  $2 pid  $3 name  $4 seconds: until the server answers 200 (model loaded)
    local i
    for i in $(seq 1 "$4"); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "$1")" = 200 ] && return 0
        kill -0 "$2" 2>/dev/null || stop_job "$3 stopped while loading (see its log in $LANE_DIR)"
        sleep 1
    done
    stop_job "$3 was not ready within $4 s (see its log in $LANE_DIR)"
}

# ---- start one lane per GPU ----------------------------------------------
SEARCH_CTX=$(grep -oE '^NUM_CTX = [0-9]+' rag/server.py | grep -oE '[0-9]+$')
[ -n "$SEARCH_CTX" ] || stop_job "could not read NUM_CTX from rag/server.py"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
N=${#GPUS[@]}
echo "Starting $N lane(s), agent open_deep_research, model $LLM ($ALIAS), tasks: $TASKS," \
     "poison every $POISON_EVERY. search, at most $POISON_COUNT time(s)"
declare -a LLM_PORT RAG_PORT
for k in $(seq 0 $((N - 1))); do
    LANE_DIR="$REPO/lanes/$SLURM_JOB_ID/L$k"; mkdir -p "$LANE_DIR"
    cp -r rag/milvus.db "$LANE_DIR/milvus.db"        # the index is a folder
    read -r P_OLL P_LLM P_RAG <<< "$(free_ports 3)"
    LLM_PORT[$k]=$P_LLM; RAG_PORT[$k]=$P_RAG
    G=${GPUS[$k]}

    CUDA_VISIBLE_DEVICES=$G OLLAMA_HOST="127.0.0.1:$P_OLL" OLLAMA_MODELS="$MODELS" OLLAMA_KEEP_ALIVE=-1 \
        OLLAMA_NUM_PARALLEL=1 OLLAMA_LOAD_TIMEOUT=30m "$OLLAMA_BIN" serve > "$LANE_DIR/ollama.log" 2>&1 &
    wait_port "$P_OLL" $! "lane $k Ollama" 60
    curl -s -m 1800 -o /dev/null "http://127.0.0.1:$P_OLL/api/generate" \
        -d "{\"model\": \"nemotron-super-49b\", \"keep_alive\": -1, \"options\": {\"num_ctx\": $SEARCH_CTX}}" \
        || stop_job "lane $k: Ollama could not load nemotron-super-49b"

    CUDA_VISIBLE_DEVICES=$G "$LLAMA_SERVER" -m "$MODELS/blobs/sha256-$BLOB" --alias "$ALIAS" --jinja \
        --reasoning-format none -c 32768 -np 1 -ngl 999 -n 5000 --temp 0.5 $EXTRA \
        --host 127.0.0.1 --port "$P_LLM" > "$LANE_DIR/llama-server.log" 2>&1 &
    wait_port "$P_LLM" $! "lane $k llama-server" 600
    wait_ready "http://127.0.0.1:$P_LLM/health" $! "lane $k llama-server (model loading)" 1800
    grep -q "no usable GPU" "$LANE_DIR/llama-server.log" && stop_job "lane $k: llama-server is not using the GPU (see $LANE_DIR/llama-server.log)"

    CUDA_VISIBLE_DEVICES=$G RAG_PORT=$P_RAG OLLAMA_URL="http://127.0.0.1:$P_OLL/api/chat" \
        RAG_LOG="$LANE_DIR/rag_requests.jsonl" ATTACK_DIR="$LANE_DIR" MILVUS_DB="$LANE_DIR/milvus.db" \
        .venv-rag/bin/python rag/server.py > "$LANE_DIR/rag-server.log" 2>&1 &
    wait_port "$P_RAG" $! "lane $k search server" 300

    echo "lane $k on GPU $G ready: ollama $P_OLL, model $P_LLM, search $P_RAG"
done

check_localhost_only                                 # rule 3, once everything is up
( while sleep 300; do check_localhost_only > /dev/null || exit 1; done ) &   # and every 5 minutes

# ---- share the runs out over the lanes ----------------------------------------
ITEMS=()
for t in $TASKS; do
    name="${t%%:*}"; reps="${t##*:}"
    [ "$name" = "CLEAN" ] || grep -q "name: $name\b" attacks.yml || stop_job "no attack card '$name' in attacks.yml"
    for r in $(seq 1 "$reps"); do ITEMS+=("$name"); done
done
echo "${#ITEMS[@]} runs over $N lane(s)"

LANE_PIDS=()
for k in $(seq 0 $((N - 1))); do
    LANE_DIR="$REPO/lanes/$SLURM_JOB_ID/L$k"
    (
        # the connector reads the lane's attack card and search log from these
        export ATTACK_DIR="$LANE_DIR" RAG_LOG="$LANE_DIR/rag_requests.jsonl"
        OUT="$REPO/runs_odr/${SLURM_JOB_ID}_L$k"; mkdir -p "$OUT"
        for j in $(seq "$k" "$N" $((${#ITEMS[@]} - 1))); do
            echo "[lane $k] run $((j + 1))/${#ITEMS[@]}: ${ITEMS[$j]}"
            ATTACK=()
            [ "${ITEMS[$j]}" = "CLEAN" ] || ATTACK=(--attack "${ITEMS[$j]}" --poison-every "$POISON_EVERY" --poison-count "$POISON_COUNT")
            "$ODR_PY" odr/run_odr.py --model "$ALIAS" --base-url "http://127.0.0.1:${LLM_PORT[$k]}/v1" \
                --rag-url "http://127.0.0.1:${RAG_PORT[$k]}" --out "$OUT" "${ATTACK[@]}" \
                > "$LANE_DIR/run_$((j + 1)).log" 2>&1 || echo "[lane $k] run $((j + 1)) FAILED (see $LANE_DIR/run_$((j + 1)).log)"
            grep -E "^4\. done|WARNING|ERROR|^attack " "$LANE_DIR/run_$((j + 1)).log" | sed "s/^/[lane $k] /"
        done
    ) &
    LANE_PIDS+=($!)
done
wait "${LANE_PIDS[@]}"

echo; echo "ALL LANES FINISHED. Run folders:"
ls -d runs_odr/${SLURM_JOB_ID}_L*/* 2>/dev/null | sed 's/^/  /'
