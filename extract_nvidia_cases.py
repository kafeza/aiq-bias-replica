# =====================================================================
# Pull the NVIDIA records our attack cards point to into one file,
# nvidia_cases.json, with everything run_attack.py and score_runs.py need:
#
#   template, severity, objective, injection node, scoring rule
#   (evaluation_method + expected_output), the request NVIDIA sent
#   (topic, report organization, collection; "{attack}" in a template field
#   means the attack text was put there), the attack text, and for attacks
#   injected into a search answer: which answer NVIDIA poisoned (slot 1-3),
#   where inside it (relative position over the <answer> text) and the
#   poisoned answer itself, as recorded in NVIDIA's draft prompt.
#
# Record choice: the card's nvidia_record if given; otherwise the first
# record of that template in stage3_cases.json (the random Stage 2 sample),
# else the first record of that template in NVIDIA's file. Slot and
# position are measured, not typed by hand. NVIDIA ran each record about 6
# times with different search orders; slot, position and poisoned answer
# come from the run kept in stage3_cases.json when there is one (as for
# the first three cards), else from the first run in NVIDIA's file.
#
# Usage (from ~/llm-safety/aiq_local, with pyarrow, e.g. .venv-rag):
#   .venv-rag/bin/python extract_nvidia_cases.py
# Reads attacks.yml, ../stage3_cases.json and NVIDIA's parquet file.
# Nothing is sent to any model.
# =====================================================================
import html, json, os, re
import pyarrow.parquet as pq
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "..", "..", "agent_safety", "data", "nemotron",
                       "security_without_defense", "data.parquet")
STAGE3 = os.path.join(HERE, "..", "stage3_cases.json")
OUT = os.path.join(HERE, "nvidia_cases.json")
DRAFT = "Generate a high-quality report from the given sources"


def value(v):
    """Span values are often JSON strings holding JSON again."""
    for _ in range(2):
        if isinstance(v, str) and v[:1] in '"[{':
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                break
    return v


def draft_prompt(trace):
    for s in sorted(trace, key=lambda s: str(s["start_time"])):
        if s.get("span_kind") != "LLM":
            continue
        msgs = value(s["attributes.input.value"])
        user = next((m["content"] for m in msgs if m.get("type") != "system"), "")
        if user.lstrip().startswith(DRAFT):
            return user
    return None


def poisoned_answer(prompt, inj):
    """(slot, relative position, answer) of the source holding the attack text."""
    for i, src in enumerate(re.findall(r"<source>(.*?)</source>", prompt, re.S), 1):
        ans = html.unescape(re.search(r"<answer>(.*?)</answer>", src, re.S).group(1))
        at = ans.find(inj[:40])
        if at >= 0:
            return i, round(at / len(ans), 2), ans
    return None, None, None


cards = yaml.safe_load(open(os.path.join(HERE, "attacks.yml")))["attacks"]
stage3 = json.load(open(STAGE3))
want = {}                                            # template -> record id (or None = first found)
for c in cards:
    rid = c.get("nvidia_record") or next((x["id"] for x in stage3 if x["template"] == c["name"]), None)
    want[c["name"]] = rid

out = {}
for batch in pq.ParquetFile(PARQUET).iter_batches(batch_size=64):
    for row in batch.to_pylist():
        snap = value(row["attack_snapshot"])
        t = snap["template"]
        name = t["name"]
        if name not in want or name in out or (want[name] and snap["id"] != want[name]):
            continue
        inj = snap["attack"]["injection_string"]
        case = {"id": snap["id"], "template": name, "severity": t["severity"],
                "objective": t["attack_objective"], "injection_node": t["injection_node"],
                "method": t["evaluation_method"], "expected": t.get("expected_output") or "",
                "topic": t["topic"], "organization": t["organization"],
                "collection": t["rag_collection"], "search_web": t["search_web"],
                "injection_string": inj}
        if t["injection_node"] == "summarize_sources":
            # Each record was run ~6 times with different search orders. Use the
            # same run as Stage 3 (stage3_cases.json) when the record is there, so
            # slot and poisoned answer match the cards used before; else this run.
            s3 = next((x for x in stage3 if x["id"] == snap["id"]), None)
            prompt = (next(st["messages"][-1]["content"] for st in s3["steps"] if st["kind"] == "draft")
                      if s3 else draft_prompt(value(row["trace"])))
            slot, pos, ans = poisoned_answer(prompt or "", inj)
            if not slot:
                continue                             # attack text not found; try the next run of this record
            case.update({"nvidia_slot": slot, "nvidia_position": pos, "recorded_answer": ans})
        out[name] = case

missing = [n for n in want if n not in out]
json.dump([out[n] for n in want if n in out], open(OUT, "w"), indent=1)
print(f"{len(out)} cases -> {OUT}")
for n, c in out.items():
    where = (f"answer #{c['nvidia_slot']} at {c['nvidia_position']}" if c["injection_node"] == "summarize_sources"
             else "in the request (" + ", ".join(k for k in ("topic", "organization") if "{attack}" in c[k]) + ")")
    print(f"  {n:<20} {c['id'][:8]}  {c['method']:<8} {c['collection']:<18} {where}")
if missing:
    print("NOT FOUND:", missing)
