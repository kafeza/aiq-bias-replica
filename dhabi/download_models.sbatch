#!/bin/bash
# =====================================================================
# Download the Ollama models used by the replica, the SAME files as on the
# Mac (checked by SHA-256 at the end). No GPU is used.
#   Falcon-H1R-7B BF16   (agent model, 15 GB)
#   Nemotron Super 49B Q4_K_M  (search answers, 30 GB; also named nemotron-super-49b)
#   gpt-oss 120b         (judge, 65 GB)
# Ollama runs only while downloading, on 127.0.0.1 with a free port (rule 3),
# and is stopped at the end. Models go to ollama-models/ in the repo folder
# (about 110 GB; listed in .gitignore, so never uploaded).
# Run from the repo folder on Dhabi:   sbatch dhabi/download_models.sbatch
# Follow it:                           tail -f download_models_<job number>.log
# =====================================================================
#SBATCH --job-name=download_models
# node 18 only, in the security team's reservation; no GPU
#SBATCH --nodelist=t01pdscgpu18
#SBATCH --reservation=security
#SBATCH --nice=1000
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=download_models_%j.log

cd "$SLURM_SUBMIT_DIR" || exit 1
export NO_GPU_JOB=1
source dhabi/check_rules.sh                          # rules 1 and 2 (0 GPUs)

OLLAMA_BIN="$HOME/ollama/bin/ollama"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$SLURM_SUBMIT_DIR/ollama-models}"   # inside the repo folder, never uploaded (.gitignore)
export CUDA_VISIBLE_DEVICES=-1                       # never a GPU in this job
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
export OLLAMA_HOST="127.0.0.1:$PORT"                 # rule 3: localhost only
mkdir -p "$OLLAMA_MODELS"
echo "Ollama $("$OLLAMA_BIN" --version 2>/dev/null | tail -1), models in $OLLAMA_MODELS, address $OLLAMA_HOST"

"$OLLAMA_BIN" serve > "ollama_download_$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT            # always stop Ollama at the end
for i in $(seq 1 60); do curl -s -o /dev/null "http://$OLLAMA_HOST/api/version" && break; sleep 1; done
check_localhost_only                                 # rule 3

# name on Dhabi | expected SHA-256 of the model file (from the Mac)
MODELS=(
  "hf.co/tiiuae/Falcon-H1R-7B-GGUF:BF16|ac852a921c550ec9c55490c2a7aabd6706843e86b1a8cb9c31d5da908245f77e"
  "hf.co/bartowski/nvidia_Llama-3_3-Nemotron-Super-49B-v1-GGUF:Q4_K_M|d3fc12f4480cad5060f183d6c186ca47d800509224632bb22e15791711950524"
  "gpt-oss:120b|6be6d66a3f546d8c19b130dc41dc24b2fc159f84ffbc76a0ee0676205083cf5a"
  # added 11 Nov: official Qwen files; expected SHA-256 = Hugging Face's checksum of the file
  "hf.co/Qwen/Qwen3-8B-GGUF:Q8_0|408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6"
  "hf.co/Qwen/Qwen3-32B-GGUF:Q4_K_M|efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"
)
# ONLY=<text>: download and check only the models whose name contains <text>
# (e.g. --export=ALL,ONLY=Qwen3 for the two Qwen files; the others are already there)
if [ -n "${ONLY:-}" ]; then
    KEEP=(); for e in "${MODELS[@]}"; do case "${e%%|*}" in *"$ONLY"*) KEEP+=("$e");; esac; done
    [ ${#KEEP[@]} -gt 0 ] || { echo "ONLY=$ONLY matches no model"; exit 1; }
    MODELS=("${KEEP[@]}")
fi
for entry in "${MODELS[@]}"; do
    name="${entry%%|*}"
    echo "=== downloading $name"
    "$OLLAMA_BIN" pull "$name" || { echo "FAILED: $name"; exit 1; }
done
# the search server asks Ollama for "nemotron-super-49b" (same file, second name, as on the Mac)
[ -z "${ONLY:-}" ] && "$OLLAMA_BIN" cp "hf.co/bartowski/nvidia_Llama-3_3-Nemotron-Super-49B-v1-GGUF:Q4_K_M" nemotron-super-49b

echo "=== checking that the files are identical to the Mac's (SHA-256; a few minutes)"
ok=1
for entry in "${MODELS[@]}"; do
    name="${entry%%|*}"; want="${entry##*|}"
    blob="$OLLAMA_MODELS/blobs/sha256-$want"
    if [ -f "$blob" ] && [ "$(sha256sum "$blob" | cut -d' ' -f1)" = "$want" ]; then
        echo "  checksum OK: $name"
    else
        echo "  DIFFERENT or missing: $name (expected $want)"; ok=0
    fi
done
"$OLLAMA_BIN" list
[ "$ok" -eq 1 ] && echo "DOWNLOAD DONE: all checksums OK" || { echo "DOWNLOAD PROBLEM: see above"; exit 1; }
