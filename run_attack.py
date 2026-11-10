# =====================================================================
# Step 12c: run NVIDIA attacks through the local AI-Q pipeline.
#
# For each attack card in attacks.yml (and each repetition):
#   1. AI-Q plans: POST /generate_query                      -> 3 queries
#   2. arm the card rag/attack.json for the RAG server's hook, aimed at the
#      query in the same slot NVIDIA poisoned (or the card's target)
#   3. AI-Q writes the report: POST /generate_summary
#      (the hook poisons that one answer, once, then disarms)
#   4. check the attack really landed (rag/attack.done.json), save the run
#
# From NVIDIA's record (nvidia_cases.json, made by extract_nvidia_cases.py
# from NVIDIA's dataset) the script takes: the attack text, where NVIDIA
# injected it, the recorded poisoned RAG answer and its slot and position
# (for replay mode), NVIDIA's goal and scoring rule, and NVIDIA's request:
# topic, outline and collection; no web search, 2 reflections.
#
# Two kinds of attack (NVIDIA's injection_node):
#   summarize_sources  one search answer is poisoned by the RAG server's hook
#   user_input         the attack is in the user's request itself (NVIDIA put
#                      it in the topic or the outline, marked "{attack}" in
#                      the record); no hook is armed, mode is "request"
#
# Needs: RAG server (rag/server.py) restarted after the hook was added,
# AI-Q (aiq serve), the model server, and Phoenix for traces.
# Usage (from ~/llm-safety/aiq_local; any Python with PyYAML):
#   python3 run_attack.py --attack SYS_COMP_1 --llm falcon_r
#   python3 run_attack.py --attack all --llm falcon_r
#   python3 run_attack.py --attack SYS_COMP_1 --llm falcon_r --mode insert
# Output: runs/<time>_<attack>_<mode>_<llm>/  card.json  queries.json
#         response.json  report.md  meta.json (times for finding the trace
#         in Phoenix, whether the hook fired, a first exact-match check)
# =====================================================================
import argparse, html, json, os, random, re, shutil, time, urllib.request
from datetime import datetime, timezone
import yaml
import run_files

HERE = os.path.dirname(os.path.abspath(__file__))
AIQ = os.getenv("AIQ_URL", "http://127.0.0.1:3838")   # Dhabi: each job its own port
CASES = os.path.join(HERE, "nvidia_cases.json")
CARD = os.path.join(run_files.ATTACK_DIR, "attack.json")       # Dhabi: one per lane
DONE = os.path.join(run_files.ATTACK_DIR, "attack.done.json")
SLOTS = {"first": 1, "second": 2, "third": 3}

ap = argparse.ArgumentParser()
ap.add_argument("--attack", required=True, help="card name in attacks.yml, or 'all'")
ap.add_argument("--llm", default="falcon_r", help="model entry in AI-Q's config_local.yml")
ap.add_argument("--mode", choices=["replay", "insert"], help="override the card's mode")
ap.add_argument("--placement", help="override the card's placement (insert mode)")
ap.add_argument("--reps", type=int, help="override the card's repetitions")
args = ap.parse_args()


def post(path, body, timeout):
    req = urllib.request.Request(AIQ + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def now():
    return datetime.now(timezone.utc).isoformat()


def nvidia_record(template):
    """NVIDIA's record for this attack (nvidia_cases.json), with its request filled in."""
    try:
        rec = next(c for c in json.load(open(CASES)) if c["template"] == template)
    except (FileNotFoundError, StopIteration):
        raise SystemExit(f"No NVIDIA record for {template}: run extract_nvidia_cases.py first")
    inj = rec["injection_string"]
    rec["request_topic"] = rec["topic"].replace("{attack}", inj)
    rec["request_organization"] = rec["organization"].replace("{attack}", inj)
    if rec["injection_node"] == "summarize_sources" and inj[:40] not in rec["recorded_answer"]:
        raise SystemExit(f"Attack text not found in NVIDIA's recorded answer for {template}")
    return rec


cfg = yaml.safe_load(open(os.path.join(HERE, "attacks.yml")))
cards = [dict(cfg["defaults"], **a) for a in cfg["attacks"]]
if args.attack != "all":
    cards = [c for c in cards if c["name"] == args.attack]
    if not cards:
        raise SystemExit(f"No card named {args.attack} in attacks.yml")

for card in cards:
    for k in ("mode", "placement", "reps"):
        if getattr(args, k) is not None:
            card[k] = getattr(args, k)
    nv = nvidia_record(card["name"])
    if card.get("nvidia_record") and card["nvidia_record"] != nv["id"]:
        raise SystemExit(f"{card['name']}: card names record {card['nvidia_record']}, nvidia_cases.json has {nv['id']}")
    card["nvidia_record"] = nv["id"]
    in_request = nv["injection_node"] == "user_input"
    if in_request:
        card["mode"] = "request"
    else:
        card.setdefault("nvidia_slot", nv["nvidia_slot"])
        card.setdefault("nvidia_position", nv["nvidia_position"])
    TOPIC, ORGANIZATION, COLLECTION = nv["request_topic"], nv["request_organization"], nv["collection"]

    for rep in range(1, card["reps"] + 1):
        for f in (CARD, DONE):                      # never start with a stale card
            if os.path.exists(f):
                os.remove(f)
        out = os.path.join(run_files.RUNS_DIR, time.strftime("%Y%m%d-%H%M%S")
                           + f"_{card['name']}_{card['mode']}_{args.llm}" + (f"_rep{rep}" if card["reps"] > 1 else ""))
        os.makedirs(out)
        meta = {"attack": card["name"], "nvidia_record": card["nvidia_record"], "mode": card["mode"],
                "llm": args.llm, "rep": rep, "topic": TOPIC, "collection": COLLECTION,
                "objective": nv["objective"], "scoring": nv["method"], "expected": nv["expected"],
                "start_utc": now()}
        print(f"\n=== {card['name']} | mode {card['mode']} | model {args.llm} | rep {rep}/{card['reps']}\n    -> {out}")

        # 1. plan
        t0 = time.time()
        q = post("/generate_query", {"topic": TOPIC, "report_organization": ORGANIZATION,
                                     "num_queries": 3, "llm_name": args.llm}, timeout=1800)
        queries = q.get("queries") or []
        json.dump(q, open(os.path.join(out, "queries.json"), "w"), indent=1)
        print(f"1. plan ({time.time() - t0:.0f}s): {len(queries)} queries")
        if len(queries) < 3:
            meta["error"] = ("fewer than 3 queries: the attack in the request derailed planning; no report"
                             if in_request else "fewer than 3 queries; attack not run")
            json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
            print("   stopped: AI-Q returned fewer than 3 queries (see queries.json)")
            continue

        # 2. arm the card, aimed at the query in NVIDIA's slot (or the card's target);
        #    an attack in the request needs no hook
        target = card.get("target", "same_as_nvidia")
        if in_request:
            target_query, slot = None, None
            print("2. no hook: the attack is in the request (" +
                  ", ".join(k for k in ("topic", "organization") if "{attack}" in nv[k]) + ")")
        elif target == "all":
            target_query, slot = "*", None
            card["call_limit"] = max(card.get("call_limit", 1), 3)
        else:
            slot = (card["nvidia_slot"] if target == "same_as_nvidia" else
                    random.randint(1, 3) if target == "random" else SLOTS[target])
            target_query = queries[slot - 1]["query"]
        if not in_request:
            armed = {"name": card["name"], "nvidia_record": card["nvidia_record"], "mode": card["mode"],
                     "placement": card.get("placement"), "nvidia_position": card["nvidia_position"],
                     "call_limit": card.get("call_limit", 1), "target_query": target_query,
                     "target_slot": slot, "injection_string": nv["injection_string"],
                     "recorded_answer": nv["recorded_answer"]}
            json.dump(armed, open(CARD, "w"), indent=1)
            print(f"2. armed: poison answer #{slot} -> {target_query[:80]}")

        # 3. write the report
        t1 = time.time()
        print("3. writing the report (this takes a while)...")
        s = post("/generate_summary", {"topic": TOPIC, "report_organization": ORGANIZATION,
                                       "queries": queries, "search_web": False,
                                       "rag_collection": COLLECTION, "reflection_count": 2,
                                       "llm_name": args.llm}, timeout=14400)
        json.dump(s, open(os.path.join(out, "response.json"), "w"), indent=1)
        report = s.get("final_report") or ""
        with open(os.path.join(out, "report.md"), "w") as f:
            f.write(report + "\n\n" + (s.get("citations") or ""))

        # 4. did the hook fire? save the used card with the run
        #    (an attack in the request is always delivered: hook_fired = None)
        fired = None if in_request else os.path.exists(DONE)
        if in_request:
            json.dump({"name": card["name"], "nvidia_record": card["nvidia_record"], "mode": "request",
                       "topic": TOPIC, "organization": ORGANIZATION},
                      open(os.path.join(out, "card.json"), "w"), indent=1)
        elif fired:
            shutil.move(DONE, os.path.join(out, "card.json"))
        else:
            if os.path.exists(CARD):
                shutil.move(CARD, os.path.join(out, "card.json"))   # disarm a card that never fired
        meta.update({"end_utc": now(), "minutes": round((time.time() - t0) / 60, 1),
                     "hook_fired": fired, "target_slot": slot, "target_query": target_query,
                     "report_words": len(report.split())})
        # first look only (final report); real scoring uses the 3 checkpoints from Phoenix
        if nv["method"] == "CONTAINS":
            meta["final_report_contains_expected"] = nv["expected"] in (report + (s.get("citations") or ""))
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        run_files.save_trace(out, meta["start_utc"], meta["end_utc"])   # spans + search log into the run folder
        print(f"4. done in {meta['minutes']} min; hook fired: {fired}; report {meta['report_words']} words"
              + (f"; '{nv['expected']}' in final report: {meta['final_report_contains_expected']}"
                 if nv["method"] == "CONTAINS" else ""))
        if fired is False:
            print("   WARNING: the attack was not planted (no search matched the target). Check Terminal A.")
