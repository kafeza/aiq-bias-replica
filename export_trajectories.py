# =====================================================================
# Export the replica's runs as trajectories in one JSON file, for sharing
# inside the organisation. Uses agent_safety's loader (aiq_replica.py) for the
# ordered steps and ground truth, and adds each LLM step's reasoning (<think>
# part) and the run's scores.
#
# The file contains NVIDIA dataset content (attack texts, NVIDIA's recorded
# poisoned answers) and harmful model outputs: NVIDIA Evaluation Dataset
# License, internal evaluation only; do not post it publicly or in a repo.
#
# Usage (from ~/llm-safety/aiq_local; Phoenix must be running):
#   python3 export_trajectories.py                 # all valid runs
#   python3 export_trajectories.py <run> [<run>]   # selected runs
# Output: exports/trajectories_<date>.json
# =====================================================================
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path("~/agent_safety").expanduser()))
import aiq_replica as ar  # noqa: E402

runs = [HERE / "runs" / r for r in sys.argv[1:]] or ar.valid_runs(HERE)
out = {"description": "AI-Q replica runs (NVIDIA's AI-Q v1.1.1 research agent with NVIDIA's "
                      "Nemotron-AIQ attacks), as ordered trajectories with ground truth and scores.",
       "notice": "Contains NVIDIA Evaluation Dataset content and harmful model outputs. "
                 "Internal evaluation only; do not redistribute or publish.",
       "exported": time.strftime("%Y-%m-%d %H:%M"),
       "fields": {"steps": "in the order the agent saw them; role user/assistant/tool; label says "
                           "which AI-Q step; text = answer without reasoning; calls = searches made; "
                           "called_by = [step, call index] for search answers; reasoning = the "
                           "model's <think> text for that step (assistant steps only)",
                  "truth": "poisoned_step = index of the poisoned search answer (null if none or "
                           "if the attack was in the request); clean_answer = that answer before "
                           "the attack was planted",
                  "scores": "per attack rule: draft (summarize_sources), revised "
                            "(reflect_on_summary), final (finalize_summary); obeyed; NVIDIA's "
                            "Nemotron means for comparison"},
       "runs": []}

for d in runs:
    run = ar.load_run(d)
    meta, start, end = ar.run_window(d)
    spans = ar.phoenix_spans(start, end)
    # reasoning per LLM step, in the same order as the loader's assistant steps
    thoughts = []
    for s in spans:
        if s.get("span_kind") != "LLM":
            continue
        prompt = str(s["attributes"].get("input.value") or "")
        kind = next((k for p, k in ar.STEP_KINDS if p in prompt), None)
        if kind in ("plan", "draft", "reflect", "extend", "finalize"):
            raw = str(ar._json(s["attributes"].get("output.value")) or "")
            if "</think>" in raw:                                   # reasoning, then the answer
                thoughts.append(raw.split("</think>")[0].replace("<think>", "").strip())
            elif "<think>" in raw:                                  # cut off while reasoning
                thoughts.append(raw.replace("<think>", "").strip())
            else:                                                   # step without reasoning
                thoughts.append(None)
    steps, k = [], 0
    for st, label in zip(run["steps"], run["labels"]):
        item = {"index": st.idx, "role": st.role, "label": label, "text": st.text,
                "calls": [{"tool": n, "args": a} for n, a in st.calls],
                "called_by": list(st.called_by) if st.called_by else None}
        if st.role == "assistant":
            item["reasoning"] = thoughts[k] if k < len(thoughts) else None
            k += 1
        steps.append(item)
    out["runs"].append({"id": run["id"], "model": run["llm"], "attack": run["attack"],
                        "mode": run["mode"], "steps": steps, "truth": run["truth"],
                        "scores": run["scores"]})
    print(f"{run['id']}: {len(steps)} steps")

(HERE / "exports").mkdir(exist_ok=True)
path = HERE / "exports" / f"trajectories_{time.strftime('%Y%m%d')}.json"
path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"-> {path}  ({path.stat().st_size / 1e6:.1f} MB, {len(out['runs'])} runs)")
