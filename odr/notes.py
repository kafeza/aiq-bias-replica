"""The second agent without its notes (run_odr.py --notes raw): each sub-researcher passes on its
search answers in full, instead of compressed notes.

In open_deep_research each sub-researcher ends with a compression step (compress_research) that
writes its findings as notes; the supervisor and the final writer see only these notes. AI-Q has
no such step: its writer sees every search answer. With --notes raw the compression step is
replaced by one that passes on, unchanged and in order, every search result the sub-researcher
received (its tavily_search tool messages; its own reflections, think_tool, are left out, as AI-Q's
writer sees none). Everything else in the agent stays as it is: the supervisor, the sub-researchers'
search loop, the final report prompt. Combined with --writing aiq, the only remaining difference to
AI-Q's writing is the agent's structure of supervisor and sub-researchers.
"""
from langchain_core.messages import filter_messages
from langgraph.graph import END, START, StateGraph

PASSED = []                                      # what each sub-researcher passed on, in order


def search_results(messages):
    """The sub-researcher's search results, unchanged and in order."""
    found = [str(m.content) for m in filter_messages(messages, include_types=["tool"])
             if getattr(m, "name", None) == "tavily_search"]
    return "\n".join(found) if found else "No search results."


def apply(module):
    """Rebuild the agent's researcher subgraph (deep_researcher) with the pass-through step. The
    supervisor looks the subgraph up by its module name when it delegates, so the new one is used."""
    async def pass_through(state, config):
        text = search_results(state.get("researcher_messages", []))
        PASSED.append(text)
        return {"compressed_research": text, "raw_notes": [text]}

    b = StateGraph(module.ResearcherState, output=module.ResearcherOutputState, config_schema=module.Configuration)
    b.add_node("researcher", module.researcher)
    b.add_node("researcher_tools", module.researcher_tools)
    b.add_node("compress_research", pass_through)
    b.add_edge(START, "researcher")
    b.add_edge("compress_research", END)
    module.researcher_subgraph = b.compile()
