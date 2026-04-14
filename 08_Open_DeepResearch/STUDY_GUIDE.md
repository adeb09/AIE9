# Session 8: Open Deep Research (Multi-Step Research Pipelines)
### Interview-Ready Study Guide — Senior / Staff AI Engineer

---

## A. Core Concept Summary

The **Deep Research pattern** is a multi-stage agentic pipeline that converts a high-level research question into a structured, citation-grounded report by decomposing it into subtopics, executing parallel retrieval across heterogeneous sources, and synthesizing results section-by-section before producing a final document. Unlike a standard RAG pipeline — which performs a single-shot retrieve-then-generate over a fixed corpus — Deep Research is fundamentally iterative and branching: the planner's output drives the topology of the downstream graph at runtime, making it a dynamically-shaped DAG rather than a static chain.

The architectural distinction matters because it shifts the primary engineering challenge from embedding quality and retrieval precision (RAG concerns) to **workflow orchestration, state management, and synthesis coherence across multiple independently-retrieved document sets**. A practitioner must hold the mental model of Deep Research as a distributed map-reduce over the web: the planner emits a work plan, a fan-out stage maps parallel search workers onto each subtopic, and a fan-in stage reduces those results into a coherent narrative with consistent provenance. Production viability depends on getting the quality/cost/latency triangle right — the number of queries, sources, and synthesis passes are the primary levers, and each has compounding effects on cost and output quality.

---

## B. Key Terms & Definitions

- **Deep Research Pattern**: An agentic pipeline architecture where a planner decomposes a query into subtopics, parallel retrieval workers collect evidence per subtopic, and a synthesizer produces a structured multi-section report. Differs from RAG by having dynamic graph topology and multi-hop, multi-source retrieval.

- **Fan-out / Fan-in**: The LangGraph orchestration pattern where a single node emits multiple `Send()` objects to dynamically spawn parallel subgraph invocations (fan-out), and a downstream aggregator node collects all parallel results back into shared state (fan-in). Critical for parallelizing independent research subtasks.

- **Send API (LangGraph)**: `langgraph.types.Send` — a primitive that allows a node to return a list of `(node_name, state_dict)` tuples, causing the graph to dynamically instantiate parallel branches at runtime. This is the mechanism that makes variable-topology research graphs possible.

- **Section Schema / Structured Output**: A Pydantic model defining the expected shape of each section in a research report (title, content, sources, confidence). Enforcing this via LLM structured output (`.with_structured_output()`) prevents hallucinated citations and malformed section boundaries.

- **Checkpoint / Persistence**: A LangGraph mechanism (via `SqliteSaver`, `PostgresSaver`, or custom checkpointers) that serializes the full graph state after each node execution. Enables resumption of long-running research tasks after crashes without replaying expensive LLM calls.

- **Multi-provider LLM Routing**: The practice of assigning different model tiers to different subtasks based on their cost/quality requirements — e.g., using `gpt-4o-mini` or `claude-haiku` for search query generation, and `gpt-4o` or `claude-opus` for final synthesis. Routing logic lives in the graph node definitions, not in a separate routing layer.

- **Source Normalization**: The process of converting heterogeneous retrieval results (Tavily JSON, Arxiv XML, Supabase rows, Azure AI Search responses) into a unified `SearchResult(url, title, snippet, score, source_type)` schema before passing them to synthesis nodes. Prevents downstream LLMs from being confused by inconsistent formatting.

- **Topic Drift**: A failure mode in long-running research pipelines where section writers, operating on loosely-related retrieved documents, gradually diverge from the original query intent. Detected by semantic similarity between section content and the original plan, and mitigated by injecting the original query into every synthesis prompt.

- **Grounded Citation**: A citation in a generated report that is directly traceable to a retrieved document in the pipeline's state. "Hallucinated citations" (plausible-sounding but unfabricated sources) are a critical failure mode unique to generative synthesis steps; they are prevented by requiring the LLM to emit only URLs/IDs present in the `sources` field of the current state.

- **Quality/Cost/Latency Triangle**: The three-way trade-off governing Deep Research configuration. Increasing search queries or sources per query improves coverage (quality) but multiplies cost and latency. Synthesis passes compound cost further. In production, you tune these parameters per query complexity class rather than using a single global config.

---

## C. How It Works — Technical Mechanics

### Architectural Contrast: RAG vs. Deep Research

| Dimension | Standard RAG | Deep Research |
|---|---|---|
| Graph topology | Static chain: retrieve → generate | Dynamic DAG: plan → fan-out → fan-in → synthesize |
| Retrieval | Single query against fixed corpus | Multi-query against heterogeneous live sources |
| State | Ephemeral (context window) | Persistent (checkpointed between nodes) |
| Parallelism | None | Explicit fan-out via Send API |
| Output | Single response | Structured multi-section report |
| Failure recovery | Restart from scratch | Resume from last checkpoint |

---

### Full Deep Research Graph — Text Diagram

```
                         ┌─────────────────────────────────┐
                         │         User Query Input        │
                         └──────────────┬──────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────────┐
                         │           PLANNER NODE          │
                         │  - Decomposes query into N      │
                         │    research subtopics           │
                         │  - Generates search queries     │
                         │    per subtopic (1–3 each)      │
                         │  - Emits: List[Section]         │
                         └──────────────┬──────────────────┘
                                        │
                              Send(section_1) ──────────────────────────────────┐
                              Send(section_2) ──────────────────────┐           │
                              Send(section_N) ──────┐               │           │
                                                    │               │           │
                                                    ▼               ▼           ▼
                              ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                              │  SEARCH NODE │  │  SEARCH NODE │  │  SEARCH NODE │
                              │  (section N) │  │  (section 2) │  │  (section 1) │
                              │ - Web search │  │ - Web search │  │ - Web search │
                              │ - Arxiv/DB   │  │ - Arxiv/DB   │  │ - Arxiv/DB   │
                              │ - Normalize  │  │ - Normalize  │  │ - Normalize  │
                              └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                                     │                 │                 │
                                     └─────────────────┴─────────────────┘
                                                        │  (fan-in: all sections complete)
                                                        ▼
                              ┌──────────────────────────────────────────────────┐
                              │             SECTION WRITER NODES                 │
                              │  (one per section, may run in parallel)          │
                              │  - Receives section plan + retrieved sources     │
                              │  - Synthesizes section content                   │
                              │  - Enforces structured output (Pydantic)         │
                              │  - Emits grounded citations from state sources   │
                              └──────────────────────┬───────────────────────────┘
                                                     │
                                                     ▼
                              ┌──────────────────────────────────────────────────┐
                              │              FINAL SYNTHESIZER NODE              │
                              │  - Assembles section drafts into coherent report │
                              │  - Resolves cross-section contradictions         │
                              │  - Produces executive summary + references       │
                              │  - Optional: reflection pass for gap detection   │
                              └──────────────────────┬───────────────────────────┘
                                                     │
                                                     ▼
                              ┌──────────────────────────────────────────────────┐
                              │             STRUCTURED REPORT OUTPUT             │
                              │  Report(title, sections[], references[], metadata)│
                              └──────────────────────────────────────────────────┘
```

---

### State Design

The shared state object must carry everything required for any node to execute independently (enabling checkpointing and resumption):

```python
from typing import Annotated, List, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel
import operator

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    score: float
    source_type: str  # "web" | "arxiv" | "database" | "azure_search"

class SectionDraft(BaseModel):
    section_id: str
    title: str
    plan: str                          # planner's intent for this section
    queries: List[str]                 # search queries used
    sources: List[SearchResult]        # all retrieved sources
    content: Optional[str] = None      # writer's output
    grounded_citations: List[str] = [] # URLs actually cited

class ResearchState(TypedDict):
    query: str
    plan: List[SectionDraft]           # planner output
    sections: Annotated[List[SectionDraft], operator.add]  # fan-in aggregator
    final_report: Optional[str]
    metadata: dict                     # run_id, model_config, cost_tracker, timestamps
```

Key design decision: `sections` uses `Annotated[..., operator.add]` so that parallel branch results are **appended** to the list rather than overwriting each other — this is the fan-in merge strategy for LangGraph parallel nodes.

---

### Multi-Provider Routing

Routing is expressed at node definition time, not in a separate router:

```python
# Fast/cheap model for query generation (high volume, low stakes)
QUERY_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Powerful model for section synthesis (low volume, high stakes)
SYNTHESIS_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Optional fallback chain
from langchain_core.runnables import RunnableWithFallbacks
synthesis_with_fallback = SYNTHESIS_MODEL.with_fallbacks(
    [ChatAnthropic(model="claude-3-5-sonnet-20241022")]
)
```

---

## D. Common Interview Questions (with Strong Answers)

---

### Q1: How does LangGraph's Send API differ from a static parallel edge, and why does that distinction matter for Deep Research?

**Q:** Walk me through how you'd implement dynamic parallelism in a research pipeline where the number of research subtopics isn't known until the planner runs.

**A:** A static parallel edge in LangGraph connects two named nodes at graph-construction time — the branching factor is fixed in the schema. The `Send` API is fundamentally different: it's a runtime primitive that lets a node return a list of `(target_node, state)` tuples, causing the graph engine to dynamically instantiate N independent branches, one per returned `Send`. In Deep Research, the planner decides at runtime how many subtopics to research based on query complexity — sometimes 3 sections, sometimes 12. Encoding that as static edges would require knowing the max branching factor ahead of time and wiring conditional logic around it, which breaks the clean separation between planning and execution. With `Send`, the planner just returns `[Send("search_node", section) for section in plan]` and LangGraph handles scheduling, parallelism, and fan-in. The critical production implication is that `Send`-based branches each receive their own **copy** of the state slice, so you need to be deliberate about what shared state they can mutate and how fan-in merges those mutations — that's where `Annotated[..., operator.add]` comes in for list accumulation.

**Staff-Level Extension:** How would you implement backpressure or rate limiting if you have 20 parallel search branches all hitting the same external API? The `Send` API doesn't have built-in concurrency caps. At the executor level you'd wrap the search node with an asyncio `Semaphore` or use a token bucket, and at the graph level you might batch `Send` calls into groups of N rather than firing all at once.

---

### Q2: How do you design checkpoint/resume for a Deep Research run that crashes mid-execution?

**Q:** A 30-section research report takes 8 minutes to generate. If it crashes at section 22, how do you resume rather than restart?

**A:** LangGraph's `SqliteSaver` or `PostgresSaver` checkpointers serialize the **full graph state** after every node execution to a persistent store, keyed by `(thread_id, checkpoint_id)`. On restart, you call `graph.get_state(config)` to retrieve the last checkpoint, then `graph.invoke(None, config)` to resume — LangGraph replays only the nodes that haven't executed yet. The key design requirement is that your state is fully serializable (no live connections, file handles, or runtime objects in state). For Deep Research specifically, you want fine-grained checkpointing: after each individual section is written, not just after all sections complete. This means using `interrupt_after=["section_writer"]` or structuring the subgraph so each section write is a discrete node execution. The operational implication is that expensive LLM calls (synthesis passes) are idempotent only if you deduplicate by section_id in state rather than by position — if you recover and re-run a partially-complete batch, you need to skip already-written sections, not re-write them.

**Staff-Level Extension:** How does checkpointing interact with external side effects — e.g., if a section writer wrote to a Supabase table before the crash? You need either idempotent writes (upsert by section_id) or a two-phase approach where state tracks which writes have been committed, so the resume pass can skip already-persisted sections.

---

### Q3: How do you prevent hallucinated citations in a generative synthesis step?

**Q:** Your research report cites 15 sources. How do you verify those citations actually exist in your retrieved document set?

**A:** The core issue is that LLMs will generate plausible-sounding URLs and paper titles even when not instructed to. There are three complementary defenses. First, enforce **closed-world citation**: in the synthesis prompt, include the explicit list of source URLs from state and instruct the model that it may only cite URLs from that list — no others. Second, use **structured output** via Pydantic where the `citations` field is typed as `List[str]` with a validator that checks each string against the known source URLs in state — any URL not in the retrieved set raises a validation error, triggering a retry. Third, do a **post-generation citation audit**: after synthesis, extract all cited URLs, verify each exists in `section.sources`, and flag or strip any that don't. The Pydantic approach catches this at generation time; the audit catches anything that slips through. In practice, the structured output approach reduces hallucinated citations to near zero because the model can see exactly what URLs it's allowed to cite — the problem is almost always one of opportunity (model doesn't know what it retrieved) rather than intent.

**Staff-Level Extension:** What happens when the retrieved sources themselves contain incorrect or contradictory information? Citation grounding ensures traceability, but not factual accuracy. You need a separate cross-source consistency check — compare claims across sections for logical contradictions, flag low-confidence sections based on source disagreement, and include provenance metadata in the output so humans can audit the full retrieval chain.

---

### Q4: How do you tune the quality/cost/latency triangle for a Deep Research system in production?

**Q:** You're deploying this for an enterprise where some queries are quick competitive scans and others are deep technical due diligence. How do you parameterize the system?

**A:** The three primary levers are `num_queries_per_section` (controls retrieval breadth), `num_sources_per_query` (controls retrieval depth), and `num_synthesis_passes` (controls output quality). These compound multiplicatively: 3 sections × 3 queries × 5 sources = 45 search API calls before a single synthesis token is spent. In production, I'd define **query complexity classes** — shallow (news scan: 2 sections, 2 queries, 3 sources, 1 synthesis pass), standard (technical overview: 5 sections, 3 queries, 5 sources, 1 pass), deep (due diligence: 10 sections, 4 queries, 10 sources, 2 passes with reflection) — and route queries to a complexity class based on a lightweight classifier or explicit user intent. The model routing also matters: using `gpt-4o-mini` for the planner and search query generation (maybe 20 calls at ~$0.00015/call) versus `gpt-4o` only for synthesis (3–5 calls at ~$0.01/call) can reduce cost by 60–70% with negligible quality loss on the synthesis side. The hardest production decision is the synthesis model — downgrading it degrades report coherence much more than downgrading the planner.

**Staff-Level Extension:** How do you measure quality to know whether a tuning change improved things? You need an automated eval harness: factual grounding score (claims traceable to sources), coherence score (section-to-section logical flow, measurable via LLM-as-judge), and user satisfaction (if you have feedback signals). Without evals, cost/latency optimization is flying blind.

---

### Q5: How do you handle conflicting information across sources in the synthesis step?

**Q:** Two of your retrieved sources directly contradict each other on a key claim. How should the synthesizer handle this?

**A:** A naive synthesis LLM will typically pick one source and ignore the other, or worse, blend them into an incoherent claim. The correct approach has two parts. First, surface contradictions **explicitly** in the synthesis prompt by including source metadata and flagging when sources disagree — the model is much better at reconciling conflicts when it knows they exist than when it's expected to discover them. Second, the synthesis node should have a structured output field like `conflict_flags: List[ConflictFlag]` where each flag captures the conflicting claim, the sources on each side, and the synthesizer's resolution strategy (e.g., "preferred more recent source", "flagged as unresolved", "presented both perspectives"). For due diligence use cases, "present both perspectives with attribution" is almost always the right strategy — the reader needs to know there's disagreement. For factual lookup use cases, you can implement a source credibility scoring step (PageRank-like authority signal, publication recency, domain reputation) to break ties. The worst outcome is silent contradiction burial, which produces confident-sounding but factually unreliable reports.

---

### Q6: How do you normalize results across heterogeneous search providers?

**Q:** Your pipeline uses Tavily for web, Arxiv for academic, Supabase for internal documents, and Azure AI Search for enterprise data. Each returns a different schema. How do you handle this before synthesis?

**A:** Each provider gets its own adapter that maps the native response to a canonical `SearchResult(url, title, snippet, score, source_type, metadata)` schema. The `score` field is particularly tricky because different providers use incompatible relevance signals: Tavily returns a float in [0,1], Arxiv returns recency/citation signals, and BM25-based systems return unnormalized term scores. I'd normalize all scores to [0,1] within each provider's batch using min-max scaling, then apply a per-provider **credibility weight** (e.g., peer-reviewed Arxiv results weight 1.3×, web results weight 1.0×, internal docs weight 1.1×) before global ranking. The synthesis prompt should include `source_type` so the model understands provenance — it should treat an Arxiv abstract differently than a marketing webpage. One production gotcha: snippet length varies wildly (Tavily gives 200 chars, Azure AI Search can return full document chunks). Truncate to a consistent max token budget per source before packing into the synthesis context window, or you'll overflow context on large source sets.

---

## E. Gotchas, Trade-offs & Best Practices

- **Fan-in ordering is non-deterministic by default.** When N parallel search branches complete and append their `SectionDraft` to the shared `sections` list, arrival order depends on network latency and LLM response time — not section order in the original plan. Always sort by `section_id` or `plan_index` in the final synthesizer node before assembling the report. Failing to do this produces coherent sections in random order, which breaks narrative flow in a way that's subtle and hard to debug in production.

- **Checkpointing LangGraph state with large payloads.** Each retrieved source set can be 10–50KB of text. Across 10 sections × 10 sources, your state blob is 1–5MB per checkpoint. `SqliteSaver` handles this fine locally, but `PostgresSaver` with naive JSONB storage will bloat quickly in production. Compress source snippets after the search node (store just enough for synthesis), or externalize large source blobs to object storage (S3) and store only references in state.

- **Topic drift in long runs.** Section writers operating on loosely-retrieved documents will drift from the original query intent if the only context they receive is their section's retrieved sources. Mitigation: always inject `state["query"]` (the original research question) and `section.plan` (the planner's stated intent for this section) into every section writer prompt. This acts as a relevance anchor. Additionally, run a **reflection pass** after all sections complete: have an LLM score each section's alignment with the original query and flag low-score sections for re-research.

- **Search API rate limits compound at scale.** With 10 sections × 3 queries per section, you fire 30 search API calls in parallel. Most search providers (Tavily: 1000 req/min on paid tiers, Exa: lower) can handle this, but if you're running multiple concurrent research jobs, you'll hit rate limits. Design the search node with exponential backoff + jitter, implement per-provider rate limit tracking in shared state metadata, and consider a local async rate limiter (e.g., `aiolimiter`) wrapping all search calls.

- **Structured output reliability degrades with complex schemas.** Using `.with_structured_output()` with a deeply nested Pydantic model (e.g., `Section` containing `List[Claim]` each containing `List[Citation]`) causes higher parse failure rates, especially with smaller models. In production, use **flat schemas with a single nesting level**, break complex structures into sequential structured calls, and always implement a fallback: catch `ValidationError`, log the raw output, and either retry with a more capable model or fall back to unstructured extraction via a parsing prompt.

---

## F. Code & Architecture Pattern

### LangGraph Fan-out / Fan-in with Send API

```python
import operator
from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Models ---

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    score: float
    source_type: str = "web"

class SectionDraft(BaseModel):
    section_id: str
    title: str
    plan: str
    queries: List[str]
    sources: List[SearchResult] = []
    content: Optional[str] = None

# --- State ---

class ResearchState(TypedDict):
    query: str
    plan: List[SectionDraft]
    # Annotated with operator.add: parallel branches append to this list
    sections: Annotated[List[SectionDraft], operator.add]
    final_report: Optional[str]

# --- Models with routing ---

PLANNER_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)
SYNTHESIS_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.2)

# --- Search tool ---

search_tool = TavilySearchResults(max_results=5)

# --- Node: Planner ---

def planner_node(state: ResearchState) -> dict:
    """Decomposes the query into a structured research plan."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_template(
        "You are a research planner. Decompose this query into 3-5 focused research subtopics.\n"
        "For each subtopic, produce: section_id (slug), title, plan (1 sentence), "
        "and 2-3 search queries.\n\n"
        "Query: {query}\n\n"
        "Return JSON: list of section objects."
    )

    chain = prompt | PLANNER_MODEL | JsonOutputParser()
    raw_sections = chain.invoke({"query": state["query"]})

    plan = [SectionDraft(**s) for s in raw_sections]
    return {"plan": plan, "sections": []}

# --- Fan-out: map plan → Send objects ---

def route_to_search(state: ResearchState) -> List[Send]:
    """Emit one Send per section — this is the fan-out."""
    return [
        Send("search_node", {"section": section, "query": state["query"]})
        for section in state["plan"]
    ]

# --- Node: Search (runs N times in parallel, one per section) ---

class SearchState(TypedDict):
    query: str
    section: SectionDraft

def search_node(state: SearchState) -> dict:
    """Executes search queries for one section and normalizes results."""
    section = state["section"]
    all_sources: List[SearchResult] = []

    for query_text in section.queries:
        try:
            raw_results = search_tool.invoke(query_text)
            normalized = [
                SearchResult(
                    url=r["url"],
                    title=r.get("title", ""),
                    snippet=r.get("content", "")[:500],  # truncate to budget
                    score=float(r.get("score", 0.5)),
                    source_type="web",
                )
                for r in raw_results
            ]
            all_sources.extend(normalized)
        except Exception as e:
            # Log and continue — one failed query shouldn't kill the section
            print(f"Search failed for query '{query_text}': {e}")

    # Deduplicate by URL, keep highest score
    seen_urls = {}
    for source in all_sources:
        if source.url not in seen_urls or source.score > seen_urls[source.url].score:
            seen_urls[source.url] = source

    section_with_sources = section.model_copy(
        update={"sources": list(seen_urls.values())}
    )

    # Return as a list — operator.add will append to ResearchState.sections
    return {"sections": [section_with_sources]}

# --- Node: Section Writer (runs after all search branches complete) ---

def section_writer_node(state: ResearchState) -> dict:
    """Synthesizes each section's content from retrieved sources."""
    from langchain_core.prompts import ChatPromptTemplate

    written_sections = []

    for section in state["sections"]:
        source_context = "\n\n".join(
            f"[{i+1}] {s.title}\n{s.url}\n{s.snippet}"
            for i, s in enumerate(section.sources[:8])  # cap context
        )

        prompt = ChatPromptTemplate.from_template(
            "Original research query: {query}\n\n"
            "Section: {title}\nSection plan: {plan}\n\n"
            "Retrieved sources:\n{sources}\n\n"
            "Write a detailed section for this topic. "
            "ONLY cite URLs that appear in the sources above. "
            "Format citations as [N] inline."
        )

        chain = prompt | SYNTHESIS_MODEL
        content = chain.invoke({
            "query": state["query"],
            "title": section.title,
            "plan": section.plan,
            "sources": source_context,
        }).content

        written_sections.append(section.model_copy(update={"content": content}))

    return {"sections": written_sections}

# --- Node: Final Synthesizer ---

def final_synthesizer_node(state: ResearchState) -> dict:
    """Assembles section drafts into a final structured report."""
    # Sort sections to match original plan order (fan-in order is non-deterministic)
    plan_order = {s.section_id: i for i, s in enumerate(state["plan"])}
    ordered_sections = sorted(
        state["sections"],
        key=lambda s: plan_order.get(s.section_id, 999)
    )

    report_parts = [f"# Research Report: {state['query']}\n"]
    for section in ordered_sections:
        report_parts.append(f"## {section.title}\n\n{section.content or '[No content generated]'}")

    # Deduplicated reference list
    all_sources = {
        s.url: s
        for section in ordered_sections
        for s in section.sources
    }
    refs = "\n".join(f"- [{s.title}]({url})" for url, s in all_sources.items())
    report_parts.append(f"\n## References\n\n{refs}")

    return {"final_report": "\n\n".join(report_parts)}

# --- Graph Assembly ---

def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("search_node", search_node)
    graph.add_node("section_writer", section_writer_node)
    graph.add_node("final_synthesizer", final_synthesizer_node)

    graph.add_edge(START, "planner")

    # Fan-out: planner → N parallel search_node invocations via Send
    graph.add_conditional_edges(
        "planner",
        route_to_search,
        ["search_node"],  # valid target node names
    )

    # Fan-in: all search_node branches → section_writer (waits for all)
    graph.add_edge("search_node", "section_writer")
    graph.add_edge("section_writer", "final_synthesizer")
    graph.add_edge("final_synthesizer", END)

    return graph.compile()

# --- Usage with Checkpointing ---

from langgraph.checkpoint.sqlite import SqliteSaver

def run_research(query: str, thread_id: str = "run-001") -> str:
    graph = build_research_graph()

    # Checkpointer enables resume on crash
    with SqliteSaver.from_conn_string("research_checkpoints.db") as checkpointer:
        checkpointed_graph = graph.with_checkpointer(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        result = checkpointed_graph.invoke(
            {"query": query, "plan": [], "sections": [], "final_report": None},
            config=config,
        )

    return result["final_report"]
```

### Key Architectural Notes on the Pattern Above

1. **`route_to_search` is where the fan-out happens.** It returns `List[Send]`, not a string node name — this is the dynamic branching primitive.

2. **`operator.add` on `sections`** is what makes fan-in work. Without this annotation, parallel branches would overwrite each other's results in state.

3. **Section writer runs after all parallel branches complete.** LangGraph guarantees that a node with multiple incoming edges (from parallel `Send` branches) waits for all branches before executing.

4. **Sorting by `plan_order` in the synthesizer** is non-optional in production. Network and LLM variance means section 5 may complete before section 1.

5. **Checkpointing with `SqliteSaver`** makes every node execution durable. Swap for `PostgresSaver` in production multi-tenant deployments.

---

*Study guide generated for Session 8: Open Deep Research. For production deployments, also review: LangGraph Studio for graph visualization, LangSmith for trace-level debugging of parallel branches, and the `interrupt_before`/`interrupt_after` APIs for human-in-the-loop review of planner output before expensive retrieval begins.*
