# =====================================================================
# Step 9: ask AI-Q for one research report, as NVIDIA's web page (and their
# benchmark harness) would. Two calls to AI-Q (aiq serve, port 3838):
#   1. POST /generate_query   topic + outline           -> 3 search queries
#   2. POST /generate_summary  topic + outline + queries,
#                              collection, 2 reflections,
#                              no web search            -> final report
# The queries from call 1 are passed to call 2 unchanged (no human edits).
#
# Request values are NVIDIA's own (run_files.nvidia_request): by default the
# record of BIAS_STEREOTYPE_1 (cystic fibrosis, the topic, outline (typo
# included), collection and search_web=False shared by SYS_COMP_1,
# PHISHING_LINK_1 and BIAS_STEREOTYPE_1); --request finance takes NVIDIA's
# finance record (Amazon's 2023 performance, over the SEC filings).
#
# Needs: RAG server running (rag/server.py, port 8081) and AI-Q running
# (aiq serve --config_file configs/config_local.yml --port 3838).
# Uses only Python's standard library, so it runs in any environment.
# Usage (from ~/llm-safety/aiq_local):
#   python3 run_report.py                  # model under test: falcon
#   python3 run_report.py --llm nemotron
#   python3 run_report.py --request finance
# Output: runs/<time>_<llm>/ with request.json, queries.json, response.json, report.md
# =====================================================================
import argparse, json, os, time, urllib.request
from datetime import datetime, timezone
import run_files

AIQ = os.getenv("AIQ_URL", "http://127.0.0.1:3838")   # Dhabi: each job its own port
ap = argparse.ArgumentParser()
ap.add_argument("--llm", default="falcon", help="model entry in config_local.yml (falcon or nemotron)")
ap.add_argument("--request", default="biomedical", choices=sorted(run_files.REQUESTS),
                help="whose request to send (NVIDIA's record of that topic, run_files.REQUESTS)")
args = ap.parse_args()
REQUEST = run_files.nvidia_request(args.request)
TOPIC, ORGANIZATION, COLLECTION = REQUEST["topic"], REQUEST["organization"], REQUEST["collection"]

def post(path, body, timeout):
    req = urllib.request.Request(AIQ + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

out = os.path.join(run_files.RUNS_DIR, time.strftime("%Y%m%d-%H%M%S") + f"_{args.llm}")
start_utc = datetime.now(timezone.utc).isoformat()
os.makedirs(out)
json.dump(REQUEST, open(os.path.join(out, "request.json"), "w"), indent=1)   # which topic was asked
print(f"Model under test: {args.llm}   Topic: {TOPIC}   Collection: {COLLECTION}\nSaving to {out}\n")

# ---- call 1: plan (3 search queries)
t0 = time.time()
q = post("/generate_query", {"topic": TOPIC, "report_organization": ORGANIZATION,
                             "num_queries": 3, "llm_name": args.llm}, timeout=900)
queries = q.get("queries") or []
json.dump(q, open(os.path.join(out, "queries.json"), "w"), indent=1)
print(f"1. queries ({time.time() - t0:.0f}s):")
for x in queries:
    print("   -", x.get("query", x))
if not queries:
    raise SystemExit("No queries came back; see queries.json")

# ---- call 2: search, draft, reflect x2, extend x2, finalize
t1 = time.time()
print("\n2. writing the report (each search goes to the RAG server; this takes a while)...")
s = post("/generate_summary", {"topic": TOPIC, "report_organization": ORGANIZATION,
                               "queries": queries, "search_web": False,
                               "rag_collection": COLLECTION, "reflection_count": 2,
                               "llm_name": args.llm}, timeout=7200)
json.dump(s, open(os.path.join(out, "response.json"), "w"), indent=1)
report = s.get("final_report") or ""
with open(os.path.join(out, "report.md"), "w") as f:
    f.write(report + "\n\n" + (s.get("citations") or ""))
print(f"   done in {(time.time() - t1) / 60:.1f} min; report {len(report.split())} words\n")
run_files.save_trace(out, start_utc, datetime.now(timezone.utc).isoformat())   # spans + search log into the run folder
print(report[:1500] + ("\n[...]" if len(report) > 1500 else ""))

# ---- a clean run must be clean: no search answer may have been poisoned
log = os.path.join(out, run_files.RAG_FILE)
entries = [json.loads(x) for x in open(log) if x.strip()] if os.path.exists(log) else []
poisoned = [r for r in entries if r.get("injected")]
if not entries:
    print("\nWARNING: no search requests saved with this run (rag_requests.jsonl); cleanliness not checked")
elif poisoned:
    open(os.path.join(out, "INVALID"), "w").write(
        f"clean run, but {len(poisoned)} search answer(s) were poisoned by an armed attack card\n")
    print(f"\nWARNING: {len(poisoned)} poisoned search answer(s): run marked INVALID")
print(f"\n4. done: clean run, {len(entries)} search answers, none poisoned" if entries and not poisoned else "")
