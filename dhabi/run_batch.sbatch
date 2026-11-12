#!/bin/bash
# =====================================================================
# Run the benchmark on node 18: one LANE per GPU, all lanes at the same time.
# A lane = one complete copy of the replica on one GPU, like one Mac:
#   Ollama (search answers, Nemotron) + llama-server (the agent's model)
#   + search server + Phoenix + AI-Q, each on its own free port on 127.0.0.1,
#   with its own attack card, search log, index copy and run folder.
# The runs are shared out over the lanes; each run folder ends up holding
# everything about the run (report, scores input, spans.json, rag_requests.jsonl).
#
# Settings (given when starting the job):
#   TASKS  which attacks and how many runs each, e.g. "BIAS_STEREOTYPE_1:20"
#          (several: "BIAS_STEREOTYPE_1:20 OTHER_CARD:10"); names from attacks.yml;
#          CLEAN = a run without attack (run_report.py), e.g. "CLEAN:10"
#   LLM    the agent's model: falcon_r (default) or nemotron
# Examples, from the repo folder on Dhabi:
#   sbatch --export=ALL,TASKS="BIAS_STEREOTYPE_1:1" dhabi/run_batch.sbatch            # 1 test run, 1 GPU
#   sbatch --gres=gpu:8 --export=ALL,TASKS="BIAS_STEREOTYPE_1:16" dhabi/run_batch.sbatch
# Results: runs/<job>_L<lane>/...   Servers' logs: lanes/<job>/L<lane>/
# Follow it:  tail -f run_batch_<job number>.log
# =====================================================================
#SBATCH --job-name=run_batch
# node 18 only, in the security team's reservation
#SBATCH --nodelist=t01pdscgpu18
#SBATCH --reservation=security
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --nice=1000
#SBATCH --time=08:00:00
#SBATCH --output=run_batch_%j.log

cd "$SLURM_SUBMIT_DIR" || exit 1
source dhabi/check_rules.sh                          # rules 1 and 2
REPO="$PWD"
LLM="${LLM:-falcon_r}"
[ -n "$TASKS" ] || stop_job "no TASKS given, e.g. --export=ALL,TASKS=\"BIAS_STEREOTYPE_1:1\""

# ---- fixed paths ------------------------------------------------------
OLLAMA_DIR="$HOME/ollama"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"
LLAMA_SERVER="$OLLAMA_DIR/lib/ollama/llama-server"
MODELS="$REPO/ollama-models"
case "$LLM" in
  falcon_r) BLOB=ac852a921c550ec9c55490c2a7aabd6706843e86b1a8cb9c31d5da908245f77e; ALIAS=falcon-h1r-7b
            EXTRA="--top-k 20 --top-p 0.95" ;;
  nemotron) BLOB=d3fc12f4480cad5060f183d6c186ca47d800509224632bb22e15791711950524; ALIAS=nemotron-super-49b
            EXTRA="--top-p 0.95" ;;
  # Qwen3 (official GGUF files from Qwen/Qwen3-8B-GGUF and Qwen/Qwen3-32B-GGUF): top-k 20 and
  # top-p 0.95, Qwen's values for thinking mode; AI-Q sends temperature 0.5 as for every model.
  # qwen3_8b_nothink: the same file and settings, thinking switched off by Qwen's own template
  # switch (enable_thinking=false) at the server; AI-Q's prompts are unchanged.
  qwen3_8b) BLOB=408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6; ALIAS=qwen3-8b
            EXTRA="--top-k 20 --top-p 0.95" ;;
  qwen3_8b_nothink) BLOB=408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6; ALIAS=qwen3-8b-nothink
            EXTRA="--top-k 20 --top-p 0.95 --chat-template-kwargs {\"enable_thinking\":false}" ;;
  qwen3_32b) BLOB=efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689; ALIAS=qwen3-32b
            EXTRA="--top-k 20 --top-p 0.95" ;;
  *) stop_job "LLM must be falcon_r, nemotron, qwen3_8b, qwen3_8b_nothink or qwen3_32b, not '$LLM'" ;;
esac
for f in "$OLLAMA_BIN" "$LLAMA_SERVER" "$MODELS/blobs/sha256-$BLOB" rag/milvus.db \
         .venv-rag/bin/python .venv-phoenix/bin/phoenix aiq/aira/.venv/bin/aiq; do
    [ -e "$f" ] || stop_job "missing $f (run download_models / setup_envs first; the index is rag/milvus.db)"
done
# llama-server bundled with Ollama has its GPU part in a separate library (cuda_v13 or
# cuda_v12); started directly, it must be told where it is, or it runs on the CPU only
# (hundreds of times slower). Pick the one that sees the GPU, else stop.
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
# the search server's context window (Ollama is preloaded with the same, see below)
SEARCH_CTX=$(grep -oE '^NUM_CTX = [0-9]+' rag/server.py | grep -oE '[0-9]+$')
[ -n "$SEARCH_CTX" ] || stop_job "could not read NUM_CTX from rag/server.py"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
N=${#GPUS[@]}
echo "Starting $N lane(s), agent model $LLM ($ALIAS), tasks: $TASKS"
declare -a AIQ_PORT PHX_PORT
for k in $(seq 0 $((N - 1))); do
    LANE_DIR="$REPO/lanes/$SLURM_JOB_ID/L$k"; mkdir -p "$LANE_DIR/phoenix"
    cp -r rag/milvus.db "$LANE_DIR/milvus.db"        # the index is a folder
    read -r P_OLL P_LLM P_RAG P_PHX P_GRPC P_AIQ <<< "$(free_ports 6)"
    AIQ_PORT[$k]=$P_AIQ; PHX_PORT[$k]=$P_PHX
    G=${GPUS[$k]}

    # this lane's copy of the agent's settings: every address -> this lane's ports
    sed -e "s#localhost:11434#127.0.0.1:$P_OLL#g" -e "s#localhost:8090#127.0.0.1:$P_LLM#g" \
        -e "s#localhost:8091#127.0.0.1:$P_LLM#g"  -e "s#localhost:8081#127.0.0.1:$P_RAG#g" \
        -e "s#localhost:4317#127.0.0.1:$P_GRPC#g" aiq_changes/config_local.yml > "$LANE_DIR/config.yml"
    grep -q "localhost" "$LANE_DIR/config.yml" && stop_job "lane $k: an address in config.yml was not replaced"

    # OLLAMA_NUM_PARALLEL=1: the search server sends one request at a time, so Ollama
    # reserves working memory for one request only
    CUDA_VISIBLE_DEVICES=$G OLLAMA_HOST="127.0.0.1:$P_OLL" OLLAMA_MODELS="$MODELS" OLLAMA_KEEP_ALIVE=-1 \
        OLLAMA_NUM_PARALLEL=1 "$OLLAMA_BIN" serve > "$LANE_DIR/ollama.log" 2>&1 &
    wait_port "$P_OLL" $! "lane $k Ollama" 60
    # load the search answers' model now (and keep it loaded), so no run waits for it.
    # Load it with the search server's own context (NUM_CTX in rag/server.py): without it,
    # Ollama takes the model's maximum (131,072 tokens = 25 GB for Nemotron), and with
    # Nemotron also as the agent on the same GPU that ran out of memory (job 592734).
    curl -s -m 900 -o /dev/null "http://127.0.0.1:$P_OLL/api/generate" \
        -d "{\"model\": \"nemotron-super-49b\", \"keep_alive\": -1, \"options\": {\"num_ctx\": $SEARCH_CTX}}" \
        || stop_job "lane $k: Ollama could not load nemotron-super-49b"

    CUDA_VISIBLE_DEVICES=$G "$LLAMA_SERVER" -m "$MODELS/blobs/sha256-$BLOB" --alias "$ALIAS" --jinja \
        --reasoning-format none -c 32768 -np 1 -ngl 999 -n 5000 $EXTRA \
        --host 127.0.0.1 --port "$P_LLM" > "$LANE_DIR/llama-server.log" 2>&1 &
    wait_port "$P_LLM" $! "lane $k llama-server" 600
    wait_ready "http://127.0.0.1:$P_LLM/health" $! "lane $k llama-server (model loading)" 900
    grep -q "no usable GPU" "$LANE_DIR/llama-server.log" && stop_job "lane $k: llama-server is not using the GPU (see $LANE_DIR/llama-server.log)"

    PHOENIX_HOST=127.0.0.1 PHOENIX_PORT=$P_PHX PHOENIX_GRPC_PORT=$P_GRPC PHOENIX_WORKING_DIR="$LANE_DIR/phoenix" \
        .venv-phoenix/bin/phoenix serve > "$LANE_DIR/phoenix.log" 2>&1 &
    wait_port "$P_PHX" $! "lane $k Phoenix" 120

    CUDA_VISIBLE_DEVICES=$G RAG_PORT=$P_RAG OLLAMA_URL="http://127.0.0.1:$P_OLL/api/chat" \
        RAG_LOG="$LANE_DIR/rag_requests.jsonl" ATTACK_DIR="$LANE_DIR" MILVUS_DB="$LANE_DIR/milvus.db" \
        .venv-rag/bin/python rag/server.py > "$LANE_DIR/rag-server.log" 2>&1 &
    wait_port "$P_RAG" $! "lane $k search server" 300

    (cd aiq/aira && exec .venv/bin/aiq serve --config_file "$LANE_DIR/config.yml" --host 127.0.0.1 --port "$P_AIQ") \
        > "$LANE_DIR/aiq.log" 2>&1 &
    wait_port "$P_AIQ" $! "lane $k AI-Q" 300

    echo "lane $k on GPU $G ready: ollama $P_OLL, model $P_LLM, search $P_RAG, phoenix $P_PHX/$P_GRPC, ai-q $P_AIQ"
done

check_localhost_only                                 # rule 3, once everything is up
# and again every 5 minutes while the runs go (programs started later, e.g. Ollama's runner)
( while sleep 300; do check_localhost_only > /dev/null || exit 1; done ) &

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
        export AIQ_URL="http://127.0.0.1:${AIQ_PORT[$k]}" PHOENIX_URL="http://127.0.0.1:${PHX_PORT[$k]}"
        export ATTACK_DIR="$LANE_DIR" RAG_LOG="$LANE_DIR/rag_requests.jsonl" RUNS_DIR="$REPO/runs/${SLURM_JOB_ID}_L$k"
        mkdir -p "$RUNS_DIR"
        for j in $(seq "$k" "$N" $((${#ITEMS[@]} - 1))); do
            echo "[lane $k] run $((j + 1))/${#ITEMS[@]}: ${ITEMS[$j]}"
            if [ "${ITEMS[$j]}" = "CLEAN" ]; then
                # no attack: clear any card left in this lane (e.g. by a failed attack run),
                # then ask for the report as NVIDIA's harness does (run_report.py)
                rm -f "$LANE_DIR/attack.json" "$LANE_DIR/attack.done.json"
                .venv-rag/bin/python run_report.py --llm "$LLM" \
                    > "$LANE_DIR/run_$((j + 1)).log" 2>&1 || echo "[lane $k] run $((j + 1)) FAILED (see $LANE_DIR/run_$((j + 1)).log)"
            else
                .venv-rag/bin/python run_attack.py --attack "${ITEMS[$j]}" --llm "$LLM" --reps 1 \
                    > "$LANE_DIR/run_$((j + 1)).log" 2>&1 || echo "[lane $k] run $((j + 1)) FAILED (see $LANE_DIR/run_$((j + 1)).log)"
            fi
            grep -E "^4\. done|WARNING" "$LANE_DIR/run_$((j + 1)).log" | sed "s/^/[lane $k] /"
        done
    ) &
    LANE_PIDS+=($!)
done
wait "${LANE_PIDS[@]}"

echo; echo "ALL LANES FINISHED. Run folders:"
ls -d runs/${SLURM_JOB_ID}_L*/* 2>/dev/null | sed 's/^/  /'
