# =====================================================================
# Where a run's files go, and saving a run's trace into its own folder.
# Used by run_attack.py, run_report.py and score_runs.py.
#
# Settings (environment variables). Unset = the Mac's behaviour, unchanged:
#   RUNS_DIR     folder for run folders           (default: runs/)
#   RAG_LOG      the search server's request log  (default: rag/logs/requests.jsonl)
#   ATTACK_DIR   where the armed attack card is   (default: rag/)
#   PHOENIX_URL  Phoenix's address                (default: http://localhost:6006)
# On Dhabi each parallel lane gets its own values (set by the job script),
# so lanes never share an attack card, a log, a Phoenix or a run folder.
#
# save_trace(): at the end of a run, copies that run's recorded steps from
# Phoenix (spans.json) and its search requests (rag_requests.jsonl) into the
# run folder, so the folder holds everything about the run and scoring does
# not need Phoenix running.
# Standard library only.
# =====================================================================
import json, os, time, urllib.parse, urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.getenv("RUNS_DIR", os.path.join(HERE, "runs"))
RAG_LOG = os.getenv("RAG_LOG", os.path.join(HERE, "rag", "logs", "requests.jsonl"))
ATTACK_DIR = os.getenv("ATTACK_DIR", os.path.join(HERE, "rag"))
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006").rstrip("/")
SPANS_FILE, RAG_FILE = "spans.json", "rag_requests.jsonl"


def phoenix_spans(start_utc, end_utc, llm_only=True):
    """Spans Phoenix recorded between two UTC times (ISO strings), oldest first."""
    out, cursor = [], None
    while True:
        q = {"start_time": start_utc, "end_time": end_utc, "limit": 100}
        if cursor:
            q["cursor"] = cursor
        url = PHOENIX_URL + "/v1/projects/default/spans?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.loads(r.read())
        out += [s for s in d["data"] if not llm_only or s.get("span_kind") == "LLM"]
        cursor = d.get("next_cursor")
        if not cursor:
            return sorted(out, key=lambda s: s["start_time"])


def rag_requests(start_utc, end_utc, log=None):
    """The search server's log entries between two UTC times (the log uses local time)."""
    log = log or RAG_LOG
    if not os.path.exists(log):
        return []
    t0 = datetime.fromisoformat(start_utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    t1 = datetime.fromisoformat(end_utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return [r for r in map(json.loads, open(log)) if t0 <= r["time"] <= t1]


def save_trace(out, start_utc, end_utc, wait=10):
    """Copy the run's spans and search requests into its folder. Never stops the run:
    a problem is written to trace_error.txt and printed."""
    time.sleep(wait)                              # let the last spans reach Phoenix
    try:
        spans = phoenix_spans(start_utc, end_utc, llm_only=False)
        json.dump(spans, open(os.path.join(out, SPANS_FILE), "w"), indent=1)
        reqs = rag_requests(start_utc, end_utc)
        with open(os.path.join(out, RAG_FILE), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in reqs)
        print(f"   trace saved: {len(spans)} spans, {len(reqs)} search requests")
    except Exception as e:                        # noqa: BLE001
        open(os.path.join(out, "trace_error.txt"), "w").write(repr(e) + "\n")
        print(f"   WARNING: trace not saved ({e}); Phoenix still has it")
