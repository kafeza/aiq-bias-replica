#!/bin/bash
# =====================================================================
# Install everything the replica needs on Dhabi, with the SAME versions as
# on the Mac. No GPU, no server, no open port. Safe to run again: steps
# already done are skipped.
#   1. uv 0.13.1 (the installer the Mac used), in .tools/
#   2. NVIDIA's AI-Q code at commit d843c18 + our patch and settings (aiq/)
#   3. NVIDIA's documents (2 zips, SHA-256 checked) unpacked in data/
#   4. three Python environments with the Mac's exact package lists
#      (dhabi/requirements/*.txt): .venv-rag, .venv-phoenix, aiq/aira/.venv
#   5. Phoenix patched to listen on 127.0.0.1 only (rule 3)
#   6. the two search models from Hugging Face, the Mac's exact versions (hf-cache/)
# Everything stays in the repo folder and is never uploaded (.gitignore).
# Run from the repo folder on Dhabi:   sbatch dhabi/setup_envs.sbatch
# Follow it:                           tail -f setup_envs_<job number>.log
# =====================================================================
#SBATCH --job-name=setup_envs
# node 18 only, in the security team's reservation; no GPU
#SBATCH --nodelist=t01pdscgpu18
#SBATCH --reservation=security
#SBATCH --nice=1000
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=setup_envs_%j.log

cd "$SLURM_SUBMIT_DIR" || exit 1
export NO_GPU_JOB=1
source dhabi/check_rules.sh                          # rules 1 and 2 (0 GPUs)
set -e                                               # from here on: stop at the first error
REPO="$PWD"

step() { echo; echo "=== $*"; }

# ---- 1. uv ------------------------------------------------------------
step "1. uv 0.13.1"
export UV_CACHE_DIR="$REPO/.tools/uv-cache" UV_PYTHON_INSTALL_DIR="$REPO/.tools/python"
UV="$REPO/.tools/uv/uv"
if [ ! -x "$UV" ]; then
    # official installer, version pinned, into .tools/uv only (changes no shell settings)
    curl -LsSf https://astral.sh/uv/0.13.1/install.sh | env UV_UNMANAGED_INSTALL="$REPO/.tools/uv" sh
fi
"$UV" --version

# ---- 2. NVIDIA's AI-Q code ----------------------------------------------
step "2. NVIDIA AI-Q at d843c18 + our patch"
AIQ_COMMIT=d843c1805c7af3e13af6482c6c452fa125f2a241
if [ ! -d aiq/.git ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent aiq
fi
git -C aiq checkout -q "$AIQ_COMMIT"
if git -C aiq apply --check ../aiq_changes/nvidia_changes.patch 2>/dev/null; then
    git -C aiq apply ../aiq_changes/nvidia_changes.patch && echo "patch applied"
else
    git -C aiq apply --reverse --check ../aiq_changes/nvidia_changes.patch && echo "patch already applied"
fi
cp aiq_changes/config_local.yml aiq/aira/configs/config_local.yml
git -C aiq log -1 --format="AI-Q at %h (%cd)"

# ---- 3. NVIDIA's documents ----------------------------------------------
step "3. NVIDIA's documents (SHA-256 checked)"
mkdir -p data
for entry in "Biomedical_Dataset|3495a369ff0109be5374395d5eca48095373ebc933fcc98bb97528e671b9f25c" \
             "Financial_Dataset|dcd525e05210afc6d6f9f8eb487a5b96d02d17f63617f6ecb1fd89963aa33279"; do
    name="${entry%%|*}"; want="${entry##*|}"; zip="data/$name.zip"
    if [ ! -f "$zip" ] || [ "$(sha256sum "$zip" | cut -d' ' -f1)" != "$want" ]; then
        curl -sSfL -o "$zip" "https://media.githubusercontent.com/media/NVIDIA-AI-Blueprints/deep-researcher-agent/$AIQ_COMMIT/data/files/$name.zip"
    fi
    [ "$(sha256sum "$zip" | cut -d' ' -f1)" = "$want" ] || { echo "STOP: $zip differs from NVIDIA's"; exit 1; }
    [ -d "data/$name" ] || python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip" "data/$name"
    echo "$name: $(find "data/$name" -name '*.pdf' | wc -l) PDFs"
done

# ---- 4. Python environments (exact Mac versions) -------------------------
step "4. Python environments"
make_env() {   # $1 folder  $2 Python version  $3 package list
    [ -x "$1/bin/python" ] || "$UV" venv -q --python "$2" "$1"
    "$UV" pip install -q --python "$1/bin/python" -r "$3"
    echo "$1: $("$1/bin/python" --version), $("$UV" pip list --python "$1/bin/python" 2>/dev/null | tail -n +3 | wc -l) packages"
}
make_env .venv-rag      3.12.15 dhabi/requirements/rag.txt
make_env .venv-phoenix  3.12.15 dhabi/requirements/phoenix.txt
make_env aiq/aira/.venv 3.13.12 dhabi/requirements/aiq.txt
"$UV" pip install -q --python aiq/aira/.venv/bin/python --no-deps -e aiq/aira

# ---- 5. Phoenix on localhost only (rule 3) --------------------------------
step "5. Phoenix patch (trace port on 127.0.0.1)"
SP=$(.venv-phoenix/bin/python -c 'import phoenix, os; print(os.path.dirname(os.path.dirname(phoenix.__file__)))')
if grep -q '"\[::\]:{self._port}"' "$SP/phoenix/server/grpc_server.py"; then
    patch -s -p1 -d "$SP" < aiq_changes/phoenix_localhost.patch && echo "patch applied"
fi
grep -q '"\[::\]:{self._port}"' "$SP/phoenix/server/grpc_server.py" \
    && { echo "STOP: Phoenix still listens on all networks"; exit 1; } \
    || echo "Phoenix listens on 127.0.0.1 only"

# ---- 6. search models (exact Mac versions) --------------------------------
step "6. search models from Hugging Face"
export HF_HOME="$REPO/hf-cache"
.venv-rag/bin/python - <<'EOF'
import os
from huggingface_hub import snapshot_download
for repo, rev in [("nvidia/llama-nemotron-embed-1b-v2", "113abe4acafa848e77ead9c0623205e511932348"),
                  ("nvidia/llama-nemotron-rerank-1b-v2", "828765652b05bd439c9789d2a6d093db1caa1443")]:
    path = snapshot_download(repo, revision=rev)
    # make "main" point at this exact version, so the code (which asks for main) gets it offline
    refs = os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"), "refs")
    os.makedirs(refs, exist_ok=True)
    open(os.path.join(refs, "main"), "w").write(rev)
    print(repo, "version", rev[:7], "->", path)
EOF

# ---- check ------------------------------------------------------------------
step "check: key versions (must match the Mac)"
.venv-rag/bin/python -c "import torch, transformers, pymilvus; print('rag:     torch', torch.__version__, '| transformers', transformers.__version__, '| pymilvus', pymilvus.__version__)"
.venv-phoenix/bin/python -c "import phoenix; print('phoenix:', phoenix.__version__)"
aiq/aira/.venv/bin/python -c "import aiq_aira, importlib.metadata as m; print('aiq:     agentiq', m.version('agentiq'))"
echo; echo "SETUP DONE"
