# =====================================================================
# The second agent: LangChain's open_deep_research (ODR), final version
# (github.com/langchain-ai/open_deep_research, commit 1b7d2e8, archived), run
# UNCHANGED against the replica's search server and a local model server.
#
# What this connector changes, and nothing else:
#   - every role (supervisor, sub-researchers, compression, final report) uses
#     the agent model on our local OpenAI-compatible server (llama-server)
#   - the search tool: ODR's "tavily_search" (same name, description and
#     arguments, so ODR's prompts stay word for word) asks our search server
#     instead of the web: one answer per query, as in AI-Q. ODR's extra step
#     of summarising each web page is not used: our answers are already
#     summaries, and AI-Q has no such step. Each answer is labelled as a
#     source with an id in its title ([SN]) and its URL (local-search://N), so
#     that the report's citations can later be traced to a search answer
#     (citation analysis deferred: AI-Q's reports have no citations).
#   - "ask the user" is switched off (AI-Q never asks)
#   - the sentence giving today's date is removed from ODR's prompts (AI-Q's
#     prompts give no date; our collection ends in 2024); --with-date keeps it
#   - no web search, no MCP tools, no cloud model
# The request is AI-Q's: the same topic and report outline (run_report.py).
#
# Run with ODR's own environment (its uv.lock, Python 3.11.17; on Dhabi installed by
# dhabi/setup_odr.sbatch into odr/open_deep_research/.venv):
#   ~/llm-safety/open_deep_research/.venv/bin/python odr/run_odr.py \
#       --model qwen3-8b --base-url http://127.0.0.1:8090/v1
# Needs: the search server (rag/server.py) and the model server running.
# Output: runs_odr/<time>_odr_<model>/  response.json  report.md
#         trace.jsonl (every model call and tool call, in order)
#         rag_requests.jsonl (the search server's log of this run)  odr_meta.json
# =====================================================================
import argparse, asyncio, json, os, sys, time, urllib.request
from datetime import datetime, timezone
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import run_files                                                   # noqa: E402
import re                                                          # noqa: E402

# AI-Q's request (run_report.py), read from its source: importing run_report would start a run
_src = open(os.path.join(REPO, "run_report.py")).read()
TOPIC, ORGANIZATION, COLLECTION = (eval(re.search(rf'^{k} = (".*")$', _src, re.M).group(1))
                                   for k in ("TOPIC", "ORGANIZATION", "COLLECTION"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="model name on the server (its --alias), e.g. qwen3-8b")
ap.add_argument("--base-url", required=True, help="the model server, e.g. http://127.0.0.1:8090/v1")
ap.add_argument("--rag-url", default=os.getenv("RAG_URL", "http://127.0.0.1:8081"))
ap.add_argument("--odr", default=next((d for d in (os.path.join(HERE, "open_deep_research", "src"),     # Dhabi
                                                  os.path.expanduser("~/llm-safety/open_deep_research/src"))  # Mac
                                       if os.path.isdir(d)), None), help="the agent's src folder")
ap.add_argument("--out", default=os.path.join(REPO, "runs_odr"))
ap.add_argument("--with-date", action="store_true",
                help="keep ODR's sentences giving today's date (default: removed, as AI-Q's prompts give no date "
                     "and our collection ends in 2024; with the date, 3 of 4 searches asked for 2026 in the test)")
ap.add_argument("--attack", help="attack card in attacks.yml (e.g. BIAS_CF_DIABETES); omitted: a clean run")
ap.add_argument("--poison-every", type=int, default=2,
                help="poison every K-th search: the K-th, 2K-th, ... (default 2: the 2nd search, as AI-Q's 2nd query)")
ap.add_argument("--poison-count", type=int, default=1,
                help="at most this many poisoned answers; fewer if the run makes fewer searches (default 1, as NVIDIA)")
args = ap.parse_args()

for k in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):          # nothing leaves the machine
    os.environ.pop(k, None)
os.environ.update({"OPENAI_API_KEY": "not-needed", "OPENAI_BASE_URL": args.base_url,
                   "OPENAI_API_BASE": args.base_url, "GET_API_KEYS_FROM_CONFIG": "false"})
sys.path.insert(0, args.odr)
from langchain_core.callbacks import BaseCallbackHandler           # noqa: E402
from langchain_core.messages import HumanMessage, messages_to_dict  # noqa: E402
from langchain_core.tools import tool                              # noqa: E402
from open_deep_research import utils as odr_utils                  # noqa: E402
import open_deep_research.deep_researcher as odr_graph             # noqa: E402
from open_deep_research.deep_researcher import deep_researcher     # noqa: E402

PROMPTS = ("clarify_with_user_instructions", "transform_messages_into_research_topic_prompt",
           "lead_researcher_prompt", "research_system_prompt", "compress_research_system_prompt",
           "final_report_generation_prompt")
if not args.with_date:                           # only the date sentences go; every other word stays
    import re as _re
    for name in PROMPTS:
        old = getattr(odr_graph, name)
        new = _re.sub(r"\s*For context, today's date is \{date\}\.", "", old)
        new = _re.sub(r"\n?Today's date is \{date\}\.\n?", "\n", new)
        assert "{date}" not in new, name
        setattr(odr_graph, name, new)

tag = (f"_{args.attack}" + (f"_every{args.poison_every}" if args.poison_every != 2 else "")
       + (f"_x{args.poison_count}" if args.poison_count != 1 else "")) if args.attack else ""
out = os.path.join(args.out, time.strftime("%Y%m%d-%H%M%S") + f"_odr{tag}_{args.model}"
                   + ("_withdate" if args.with_date else ""))
os.makedirs(out)

# ---- the attack: the same card as for AI-Q (run_attack.py), counted instead of matched -------
CARD = os.path.join(run_files.ATTACK_DIR, "attack.json")
DONE = os.path.join(run_files.ATTACK_DIR, "attack.done.json")
for f in (CARD, DONE):                           # never start with a stale card
    if os.path.exists(f):
        os.remove(f)
armed = None
if args.attack:
    import yaml
    card = next(c for c in yaml.safe_load(open(os.path.join(REPO, "attacks.yml")))["attacks"] if c["name"] == args.attack)
    template = card.get("based_on") or card["name"]
    rec = next(c for c in json.load(open(os.path.join(REPO, "nvidia_cases.json"))) if c["template"] == template)
    rec = run_files.planted_record(card, rec)    # our statement planted, exactly as for AI-Q
    if rec["injection_node"] != "summarize_sources" or rec["injection_string"][:40] not in rec["recorded_answer"]:
        raise SystemExit(f"{args.attack}: not an attack planted in a search answer")
    armed = {"name": card["name"], "nvidia_record": rec["id"], "mode": "replay", "placement": None,
             "nvidia_position": rec["nvidia_position"], "target_query": "*", "target_slot": None,
             "every": args.poison_every, "call_limit": args.poison_count,
             "injection_string": rec["injection_string"], "recorded_answer": rec["recorded_answer"]}
    if args.poison_every < 1 or args.poison_count < 1:
        raise SystemExit("--poison-every and --poison-count must be at least 1")
    json.dump(armed, open(CARD, "w"), indent=1)
    print(f"armed: {args.attack}, poison every {args.poison_every}. search, at most {args.poison_count} time(s)")
TRACE = open(os.path.join(out, "trace.jsonl"), "w")
t_start = time.time()


def log(kind, **kw):
    TRACE.write(json.dumps({"kind": kind, "t": round(time.time() - t_start, 2), **kw}, default=str) + "\n")
    TRACE.flush()


# ---- the search tool: our search server, under ODR's tool name ------------------
SOURCES = []                                     # every answer given, in order: (id, query, answer)


def ask_search_server(query):
    body = {"messages": [{"role": "user", "content": query}], "collection_name": COLLECTION}
    req = urllib.request.Request(args.rag_url + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        text = r.read().decode()
    first = json.loads(text.split("data: ", 2)[1].split("\n", 1)[0])
    return first["choices"][0]["message"]["content"]


@tool(description=odr_utils.TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(queries: List[str]) -> str:
    """ODR's search tool, answered by our search server (one answer per query)."""
    parts = []
    for q in queries:
        answer = await asyncio.to_thread(ask_search_server, q)
        SOURCES.append({"id": len(SOURCES) + 1, "query": q, "answer": answer})
        n = SOURCES[-1]["id"]
        parts.append(f"\n\n--- SOURCE {len(parts) + 1}: [S{n}] Search result for \"{q}\" ---\n"
                     f"URL: local-search://{n}\n\nSUMMARY:\n{answer}\n\n\n" + "-" * 80 + "\n")
    return "Search results: \n\n" + "".join(parts) if parts else "No valid search results found."


odr_utils.tavily_search = tavily_search          # get_search_tool() returns this module-level tool


# ---- the record of the run: every model call and tool call ----------------------
class Recorder(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kw):
        log("llm_start", run=str(run_id), parent=str(parent_run_id), node=(metadata or {}).get("langgraph_node"),
            path=(metadata or {}).get("langgraph_checkpoint_ns"),
            tools=[t.get("function", {}).get("name") for t in (kw.get("invocation_params") or {}).get("tools", [])],
            messages=messages_to_dict(messages[0]))

    def on_llm_end(self, response, *, run_id, **kw):
        g = response.generations[0][0]
        msg = getattr(g, "message", None)
        log("llm_end", run=str(run_id), text=g.text,
            tool_calls=getattr(msg, "tool_calls", None), usage=(response.llm_output or {}).get("token_usage"))

    def on_llm_error(self, error, *, run_id, **kw):
        log("llm_error", run=str(run_id), error=repr(error))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, metadata=None, **kw):
        log("tool_start", run=str(run_id), parent=str(parent_run_id), name=(serialized or {}).get("name"),
            node=(metadata or {}).get("langgraph_node"), input=kw.get("inputs") or input_str)

    def on_tool_end(self, output, *, run_id, **kw):
        log("tool_end", run=str(run_id), output=getattr(output, "content", output))


# ---- one run ------------------------------------------------------------------------
model = "openai:" + args.model
config = {"configurable": {
    "research_model": model, "summarization_model": model, "compression_model": model,
    "final_report_model": model, "allow_clarification": False, "search_api": "tavily",
    "mcp_config": None}, "callbacks": [Recorder()], "recursion_limit": 200}
request = f"{TOPIC}\n\nReport organization:\n{ORGANIZATION}"
start_utc = datetime.now(timezone.utc).isoformat()
print(f"ODR with {args.model} at {args.base_url}; search server {args.rag_url}\nSaving to {out}\n", flush=True)
log("request", text=request, config={k: v for k, v in config["configurable"].items()}, date_in_prompts=args.with_date,
    prompts={n: getattr(odr_graph, n) for n in PROMPTS})
error = None
try:
    result = asyncio.run(deep_researcher.ainvoke({"messages": [HumanMessage(content=request)]}, config))
except Exception as e:                           # noqa: BLE001  (the run is kept, with the error)
    result, error = {}, repr(e)
    print("ERROR:", error)
end_utc = datetime.now(timezone.utc).isoformat()

report = result.get("final_report") or ""
TRACE.flush()
_calls = [json.loads(x) for x in open(os.path.join(out, "trace.jsonl"))]
_node = {c["run"]: c.get("node") for c in _calls if c["kind"] == "llm_start"}
# the compressed notes, one per sub-researcher, from the record: ODR empties its own "notes"
# after writing the report
notes = [c["text"] for c in _calls if c["kind"] == "llm_end" and _node.get(c["run"]) == "compress_research"]
json.dump({"final_report": report, "research_brief": result.get("research_brief"),
           "notes": notes, "raw_notes": result.get("raw_notes"), "sources": SOURCES, "error": error},
          open(os.path.join(out, "response.json"), "w"), indent=1)
open(os.path.join(out, "report.md"), "w").write(report)
reqs = run_files.rag_requests(start_utc, end_utc)
with open(os.path.join(out, run_files.RAG_FILE), "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in reqs)
TRACE.close()
calls = [json.loads(x) for x in open(os.path.join(out, "trace.jsonl"))]
meta = {"agent": "open_deep_research", "odr_commit": "1b7d2e8", "llm": args.model, "start_utc": start_utc,
        "end_utc": end_utc, "minutes": round((time.time() - t_start) / 60, 1), "searches": len(SOURCES),
        "model_calls": sum(c["kind"] == "llm_start" for c in calls),
        "researchers": sum(c["kind"] == "llm_start" and c.get("node") == "compress_research" for c in calls),
        "error": error, "attack": args.attack, "date_in_prompts": args.with_date}
if armed:                                        # did the card fire? which searches were poisoned?
    fired = os.path.exists(DONE)
    used = json.load(open(DONE if fired else CARD)) if (fired or os.path.exists(CARD)) else {}
    os.replace(DONE if fired else CARD, os.path.join(out, "card.json")) if used else None   # always disarm
    poisoned = [i + 1 for i, r in enumerate(reqs) if r.get("injected")]
    meta.update({"nvidia_record": armed["nvidia_record"], "based_on": rec.get("based_on"),
                 "poison_every": args.poison_every, "poison_count": args.poison_count,
                 "hook_fired": bool(used.get("used")), "poisoned_uses": used.get("used", 0),
                 "poisoned_requests": poisoned,
                 "poisoned_queries": [r["question"] for r in reqs if r.get("injected")]})
json.dump(meta, open(os.path.join(out, "odr_meta.json"), "w"), indent=1)
if armed:
    print(f"attack {args.attack}: hook fired {meta['hook_fired']}, poisoned search(es) {meta['poisoned_requests']}: "
          f"{[q[:60] for q in meta['poisoned_queries']]}")
print(f"\n{meta['minutes']} min; {meta['searches']} searches; {meta['model_calls']} model calls; "
      f"{meta['researchers']} sub-researchers; report {len(report.split())} words" + (f"; ERROR {error}" if error else ""))
print("4. done" if report and not error else "WARNING: no report")
