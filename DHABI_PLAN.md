# Running the replica on the Dhabi cluster: plan

Status: **plan only, 10 Nov 2026.** Nothing here has been done yet except step 1.
Each step is agreed before it is done. **All commands on Dhabi are run by Eleanna**;
the scripts are written and checked here first.

## Security rules (every script and command for Dhabi)

Every script, command and code change for Dhabi must meet all three rules, and is checked
against them before it is used.

| Rule | How it is enforced |
|---|---|
| **1. Runs only on node 18** (`t01pdscgpu18`) | `#SBATCH --nodelist=t01pdscgpu18`, **and** the script stops at once if `hostname` is not node 18 |
| **2. Uses at most 2 GPUs** | `#SBATCH --gres=gpu:1` (or `gpu:2` at most), **and** the script stops at once if it sees more than 2 GPUs |
| **3. Nobody outside node 18 can connect** | every server listens on `127.0.0.1` only (table below), **and** after starting them the script lists every open port of the job and stops everything if any port listens on anything other than `127.0.0.1`. No tunnels, no port forwarding, no public links |

Notes on rule 3:
- **Phoenix needs two fixes.** Its web page (port 6006) listens on all networks by default:
  fixed with the setting `PHOENIX_HOST=127.0.0.1`. Its trace port (4317) listens on all
  networks and this is written into Phoenix's code (`[::]` in `phoenix/server/grpc_server.py`):
  fixed with a one-line patch (`aiq_changes/phoenix_localhost.patch`, to be written).
- Other users logged in to node 18 itself can reach `127.0.0.1` ports; that is accepted.
- **Unusual port numbers**, checked to be free before each start, so that our runs never
  talk to another user's server on node 18 (for example someone else's Ollama on 11434),
  and theirs never talk to ours.
- Outgoing connections (downloads from Hugging Face, GitHub, Ollama) happen only during
  setup; they let nobody in.

## Fair use of node 18

- Dhabi has no low-priority level (only QOS `normal`, no preemption), so we use
  `--nice` (others go first in the queue), a **time limit**, a fixed number of processors
  (memory cannot be requested: SLURM does not count it on node 18), and **short jobs** (one batch of runs per job).
- If node 18 is busy, the job waits; it never moves to another node.

```bash
#SBATCH --nodelist=t01pdscgpu18
#SBATCH --gres=gpu:1            # at most gpu:2 (rule 2)
#SBATCH --nice=1000
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
# no --mem line: SLURM does not count memory on node 18 (RealMemory=1), any --mem is refused
```

## What runs where (all inside one SLURM job)

| Part | What it does | How it runs on Dhabi | Address (rule 3) |
|---|---|---|---|
| Ollama 0.40.15 | the search server's answer model (Nemotron) and, later, the judge (gpt-oss-120b) | `~/ollama` (installed, no container) | `OLLAMA_HOST=127.0.0.1:21434` |
| llama-server | the agent's model (Falcon, Nemotron, **or the new model**) | bundled with Ollama: `~/ollama/lib/ollama/llama-server` | `--host 127.0.0.1 --port 28090` |
| search server | NVIDIA's documents, the attack hook | conda environment `aiq-rag` | `127.0.0.1:28081` |
| Phoenix | records every step of the agent | conda environment `aiq-phoenix` | `PHOENIX_HOST=127.0.0.1`, web `26006`, traces `24317` (patched) |
| AI-Q | NVIDIA's research agent | its own environment, NVIDIA's pinned versions | `--host 127.0.0.1 --port 23838` |

Same versions as on the Mac, so the Mac runs and the Dhabi runs can be added together:
Ollama 0.40.15, the same model files, NVIDIA's AI-Q at commit `d843c18` with our patch.

**GPU memory (80 GB per GPU):** Nemotron Q4 for the search answers (30 GB) + the agent model
(Falcon 15 GB, or Nemotron 30 GB) + the search models (about 5 GB) = 50–65 GB, so 1 GPU is
enough. The judge (65 GB) runs afterwards, in its own job, with nothing else loaded.

## Steps

1. **Ollama 0.40.15 in `~/ollama`.** ✅ Done (10 Nov). It includes llama-server.
2. **Get the code** into `~/aiq-bias-replica` (clone this repo, branch `alterations`),
   then NVIDIA's AI-Q into `aiq/` with our patch (see `aiq_changes/README.md`).
3. **NVIDIA's documents:** they come with NVIDIA's AI-Q download
   (`aiq/data/files/*.zip`, stored with Git LFS: `git lfs pull`), 290 MB. Unzip into `data/`.
4. **Models** (about 110 GB, downloaded once into a folder on `/lustre1`):
   - with Ollama: Falcon-H1R-7B BF16, Nemotron Super 49B Q4_K_M, gpt-oss-120b
     (the same names as on the Mac);
   - the search models from Hugging Face: `nvidia/llama-nemotron-embed-1b-v2` and
     `nvidia/llama-nemotron-rerank-1b-v2`;
   - **the new model**: to be decided (name, file format, size).
5. **Conda environments** for the search server and Phoenix, with the Mac's versions
   (torch 2.14.1, transformers 5.19.0, pymilvus 3.0.2, milvus-lite 3.2.2, pymupdf 1.28.2,
   arize-phoenix 20.23.0; Python 3.12); AI-Q's own environment from NVIDIA's `uv.lock`
   (Python 3.13). Then the Phoenix patch.
6. **Build the search index** (one short SLURM job, 1 GPU): `rag/ingest.py` for both
   collections.
7. **SLURM job script** `dhabi/run_batch.sbatch`, in this order:
   1. **checks rules 1 and 2** (node 18, at most 2 GPUs); stops if either fails;
   2. checks that our ports are free; stops if one is taken;
   3. starts Ollama, llama-server, the search server, Phoenix and AI-Q, each on `127.0.0.1`;
   4. **checks rule 3**: lists every open port of the job (`ss -tlnp`); stops everything if
      any listens on anything other than `127.0.0.1`;
   5. runs a batch of attacks; runs are saved one by one, so a job that ends early loses
      at most the run in progress;
   6. stops everything at the end, also when the job is cancelled or times out.
8. **Test:** one run, compared with a Mac run of the same attack.
9. **Scoring job** (`dhabi/score.sbatch`): the judge, after the runs; the same checks
   (rules 1–3) as step 7.

## Code changes needed (each shown before it is made)

| File | Change | Why |
|---|---|---|
| `rag/retrieve.py`, `rag/ingest.py` | use the GPU (`cuda`) when there is one; now only the Mac GPU (`mps`) or the CPU | otherwise the search models run on the CPU |
| `rag/server.py` | port from a setting (now fixed at 8081); address stays `127.0.0.1` | unusual ports (rule 3) |
| `score_runs.py`, `run_attack.py` | addresses of Phoenix, Ollama and AI-Q from settings | unusual ports |
| `aiq_changes/config_local.yml` | a Dhabi version with the unusual ports, and an entry for the new model | so the agent finds our servers and can use the new model |
| new `aiq_changes/phoenix_localhost.patch` | Phoenix's trace port on `127.0.0.1` instead of all networks | rule 3 |
| new `dhabi/` folder | job scripts (steps 7 and 9) with the rule checks, and the setup commands | starting everything in one job, safely |
| `README.md` | Mac paths (`/Applications/Ollama.app/...`, `caffeinate`) explained for Dhabi too | they don't exist on Linux |

## Open questions

- **The new model:** name, where it comes from, file format (GGUF for llama-server?), size.
- **Personal disk limit** on `/lustre1` (the `quota` command said "ask your administrator");
  we need about 120 GB plus the new model.
- **Phoenix's data folder**: put on `/lustre1` (it records every run).
- **Who can read our files** on `/lustre1` (`ls -ld ~ ~/Projects`): close if open.
