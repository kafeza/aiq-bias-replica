# Our changes to NVIDIA's AI-Q code

NVIDIA's AI-Q research agent is not copied into this repo. Download it from NVIDIA and add
the files in this folder:

| File | What it is | Where it goes |
|---|---|---|
| `config_local.yml` | our model settings: which models the agent uses (names, ports, temperature 0.5, max_tokens 5000), the search server's address, tracing. **A new model is added here.** | `aiq/aira/configs/config_local.yml` |
| `nvidia_changes.patch` | two timeouts raised, because the Mac's search server answers one request at a time: `ASYNC_TIMEOUT` 120 → 900 s (`constants.py`) and the HTTP session timeout set to the same value (`search_utils.py`) | applied with `git apply` (below) |

## Set up (from the repo folder)

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent aiq
cd aiq
git checkout d843c1805c7af3e13af6482c6c452fa125f2a241   # the version we use (AI-Q v1.1.1, 18 Sep 2025)
git apply ../aiq_changes/nvidia_changes.patch
cp ../aiq_changes/config_local.yml aira/configs/
```

`aiq/` is listed in `.gitignore`, so NVIDIA's code is never uploaded with this repo.

If you change `config_local.yml` or NVIDIA's code on your machine, copy the change back
here (`cp aiq/aira/configs/config_local.yml aiq_changes/`, or
`git -C aiq diff > aiq_changes/nvidia_changes.patch`) so the team gets it.
