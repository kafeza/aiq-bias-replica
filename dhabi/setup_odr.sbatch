#!/bin/bash
# =====================================================================
# Install the second agent, LangChain's open_deep_research, on Dhabi, exactly as on the Mac:
#   1. its code from GitHub at the pinned final version (commit 1b7d2e8), into
#      odr/open_deep_research/ (never uploaded: .gitignore)
#   2. its packages at the versions of its own lock file (uv sync --frozen), Python 3.11.17
#      as on the Mac, with the replica's uv 0.13.1
#   3. a check: the agent loads, and our connector (odr/run_odr.py) compiles
# No GPU, no server, no open port. Safe to run again.
# Downloads go to the node's local disk (/tmp, one folder per job, deleted at the end);
# only the finished environment stays on the shared disk.
# Run from this repo folder on Dhabi:   sbatch dhabi/setup_odr.sbatch
# Follow it:                           tail -f setup_odr_<job number>.log
# =====================================================================
#SBATCH --job-name=setup_odr
# node 18 only, in the security team's reservation; no GPU
#SBATCH --nodelist=t01pdscgpu18
#SBATCH --reservation=security
#SBATCH --nice=1000
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=setup_odr_%j.log

cd "$SLURM_SUBMIT_DIR" || exit 1
export NO_GPU_JOB=1
source dhabi/check_rules.sh                          # rules 1 and 2 (0 GPUs); this job opens no port
set -e                                               # from here on: stop at the first error
REPO="$PWD"
ODR="$REPO/odr/open_deep_research"
COMMIT=1b7d2e8                                       # the final version (repository archived Aug 2026)
step() { echo; echo "=== $*"; }

# ---- 1. the agent's code, pinned --------------------------------------------
step "1. open_deep_research @ $COMMIT"
[ -d "$ODR/.git" ] || git clone -q https://github.com/langchain-ai/open_deep_research.git "$ODR"
git -C "$ODR" fetch -q origin
git -C "$ODR" checkout -q "$COMMIT"
echo "at $(git -C "$ODR" rev-parse --short HEAD): $(git -C "$ODR" log -1 --format=%s)"
[ -z "$(git -C "$ODR" status --porcelain)" ] || { echo "STOP: the agent's code was changed locally"; exit 1; }

# ---- 2. its packages, from its lock file ----------------------------------------
step "2. packages (uv sync --frozen, Python 3.11.17)"
UV="$REPO/.tools/uv/uv"
[ -x "$UV" ] || { echo "STOP: uv not found at $UV (run setup_envs first)"; exit 1; }
export UV_CACHE_DIR="/tmp/uv-cache-$SLURM_JOB_ID" UV_PYTHON_INSTALL_DIR="$REPO/.tools/python" UV_LINK_MODE=copy
trap 'rm -rf "$UV_CACHE_DIR"' EXIT
"$UV" --version
(cd "$ODR" && "$UV" sync --frozen --python 3.11.17)

# ---- 3. check ---------------------------------------------------------------
step "3. check"
"$ODR/.venv/bin/python" -c "
import sys
from importlib.metadata import version as v
sys.path.insert(0, '$ODR/src')
import open_deep_research.deep_researcher as d
print('python', sys.version.split()[0], '| langgraph', v('langgraph'), '| langchain-core', v('langchain-core'),
      '| langchain-openai', v('langchain-openai'))
print('graph:', list(d.deep_researcher.get_graph().nodes))"
"$ODR/.venv/bin/python" -m py_compile odr/run_odr.py && echo "connector odr/run_odr.py compiles"
echo; echo "SETUP ODR DONE"
