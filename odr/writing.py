"""The writing instructions of the second agent, swapped for AI-Q's (run_odr.py --writing aiq).

The two agents differ in structure (AI-Q: one fixed pipeline whose writing steps see every search
answer; open_deep_research: a supervisor and sub-researchers, the final writer sees only their
notes) and in what they tell the model when it writes:
  AI-Q (aiq/aira/src/aiq_aira/prompts.py, summarizer_instructions and report_extender):
      "Highlight the most relevant pieces of information across all sources", "Ensure a
      coherent flow of information", "Seamlessly add information from the new sources",
      "Do not include any source citations"
  open_deep_research (prompts.py, compress_research_* and final_report_generation_prompt):
      repeat the findings "verbatim", "don't rewrite it, don't summarize it, don't paraphrase it",
      "inline citations for each source", "Citations are extremely important"
With --writing aiq the second agent keeps its structure and its search loop, and its two writing
steps get AI-Q's instructions instead: the notes (compression) are written as AI-Q's draft is,
and the final report keeps every instruction of its own except those about citations and sources,
which are replaced by AI-Q's. Comparing the two settings separates the effect of the structure
from that of the writing instructions.
"""

AIQ_NOTES = """You are a research assistant that has conducted research on a topic by calling several tools and web searches. Generate a high-quality report of the findings from the given sources.

# Instructions
1. Highlight the most relevant pieces of information across all sources
2. Provide a concise and comprehensive overview of the key points related to the research topic
3. Focus the bulk of the analysis on the most significant findings or insights
4. Ensure a coherent flow of information
5. Do not include any source citations.
"""

AIQ_NOTES_HUMAN = """All above messages are about research conducted by an AI Researcher. Please write the report of these findings."""

# (text in open_deep_research's final report prompt, its replacement), applied to the prompt as
# run_odr.py has it (after the date sentence is removed)
FINAL_EDITS = [
    ("3. References relevant sources using [Title](URL) format\n",
     "3. Highlights the most relevant pieces of information across all sources, with a coherent flow of information\n"),
    ('5. Includes a "Sources" section at the end with all referenced links\n',
     "5. Does not include any source citations or list of sources\n"),
    ("Format the report in clear markdown with proper structure and include source references where appropriate.",
     "Format the report in clear markdown with proper structure."),
]
CITATION_RULES = ("<Citation Rules>", "</Citation Rules>")


def aiq_final(prompt):
    """open_deep_research's final report prompt with AI-Q's instructions about sources."""
    for old, new in FINAL_EDITS:
        assert prompt.count(old) == 1, f"not found once in the final report prompt: {old[:60]}"
        prompt = prompt.replace(old, new)
    i, j = prompt.index(CITATION_RULES[0]), prompt.index(CITATION_RULES[1]) + len(CITATION_RULES[1])
    prompt = prompt[:i] + "Do not include any source citations." + prompt[j:]
    for word in ("Citation", "[Title](URL)", "Sources\" section"):
        assert word not in prompt, word
    return prompt


def apply(module):
    """Swap the writing instructions in the agent's module (deep_researcher) for AI-Q's."""
    module.compress_research_system_prompt = AIQ_NOTES
    module.compress_research_simple_human_message = AIQ_NOTES_HUMAN
    module.final_report_generation_prompt = aiq_final(module.final_report_generation_prompt)
