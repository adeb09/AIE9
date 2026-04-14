# Session 5: Multi-Agent Systems with LangGraph
### Interview-Ready Study Guide — Senior/Staff AI Engineer Level

---

## A. Core Concept Summary

Multi-agent systems decompose a complex AI task across a network of specialized LLM-powered nodes, each with a narrow scope of tools and context, coordinated either by a central orchestrator (supervisor) or via lateral control transfers (handoff). The core insight is borrowed directly from distributed systems: a monolithic agent suffers from context window pressure, tool namespace pollution, and single-point failure modes in the same way a monolithic service suffers under load. By routing work to specialized agents, you get scope isolation, independent failure boundaries, and parallelism — but you pay in coordination overhead, latency, and observability complexity.

In LangGraph, multi-agent graphs are first-class: each agent is a node (or subgraph), state flows along edges, and control transfer is explicit via `Command` objects or conditional edge routing. The practitioner's key mental model is **topology determines coupling**: supervisor patterns centralize routing logic and are easier to debug but create a bottleneck and hallucination surface; handoff patterns distribute routing logic and reduce latency but make system behavior harder to reason about holistically.

For production systems — especially in search, recommendation, and retrieval pipelines — multi-agent architectures matter because they let you swap individual agents (e.g., re-rank vs. retrieve) without rebuilding the entire pipeline, and they expose natural seams for A/B testing and observability hooks.

---

## B. Key Terms & Definitions

- **Supervisor Node**: A central LLM node whose sole responsibility is routing: it examines the current task state and emits a structured decision (`next: "agent_name"` or `FINISH`) that determines which sub-agent executes next. It holds no domain knowledge — it only orchestrates.

- **Handoff / Transfer Tool**: An explicit tool call (e.g., `transfer_to_research_agent()`) that an agent invokes to yield control to another agent. In LangGraph, this is implemented via `Command(goto="target_node")`, which modifies graph traversal at runtime rather than via static edges.

- **Command(goto=...)**: A LangGraph primitive returned by a node function that imperatively redirects execution to a named node, optionally carrying a state update. It decouples routing logic from graph topology definition, enabling dynamic multi-hop flows.

- **Shared State**: A `TypedDict` (or Pydantic model) accessible to all agents in the graph — analogous to a shared memory bus. All reads and writes are visible to every node unless scoped with private state patterns.

- **Private State / Subgraph State**: State that is local to a subgraph and not propagated to the parent graph. Achieved by defining a separate `StateGraph` with its own schema; the parent graph only sees what the subgraph explicitly returns via its output schema.

- **Tool Call Storm**: A failure mode where an agent enters a tight loop of tool invocations — typically because the tool's output is ambiguous and the LLM re-invokes it seeking confirmation. Especially dangerous in multi-agent systems where each hop can independently storm.

- **Run Tree (LangSmith)**: The hierarchical trace structure LangSmith builds for a multi-agent execution. The root run is the top-level graph invocation; each agent hop, LLM call, and tool call is a child run. Critical for debugging — a misrouted handoff appears as a child run on an unexpected branch.

- **Context Window Exhaustion**: As state is passed between agents, the accumulated message history grows. If agents blindly append rather than summarize or truncate, later agents receive a context that has ballooned beyond the model's effective attention range, degrading output quality silently.

- **Tavily Search Tool**: A web search API optimized for LLM retrieval — returns ranked, pre-cleaned text snippets with source URLs. Used to ground agent responses in real-time data without requiring a RAG pipeline. The `include_raw_content` flag returns full page text; the default returns summaries.

- **Structured Output Routing**: The supervisor pattern where the supervisor LLM is instructed (via system prompt + `with_structured_output(RouterSchema)`) to emit a typed JSON decision object rather than free text, reducing hallucinated route names.

---

## C. How It Works — Technical Mechanics

### Supervisor Pattern

The supervisor is a standard LangGraph node that calls an LLM. The LLM is bound with a routing schema — typically a `Literal` union of valid agent names plus `"FINISH"`. The supervisor reads the current state (task description + any prior results), emits a structured `next` field, and the graph uses a conditional edge to dispatch to the appropriate sub-agent.

```
Routing flow:
  - Supervisor calls LLM with structured output schema: { next: Literal["agent_a", "agent_b", "FINISH"] }
  - Conditional edge reads state["next"] and dispatches accordingly
  - Sub-agent executes, writes results back to shared state
  - Control returns to supervisor
  - Loop repeats until state["next"] == "FINISH"
```

**Failure surface**: The supervisor LLM can hallucinate a route name not in the schema. Mitigation: use `with_structured_output(..., strict=True)` with an enum-constrained schema so the model's output is validated before the conditional edge evaluates. Without this, a hallucinated `"agent_c"` causes a `KeyError` or silent no-op.

### Handoff Pattern

Each agent is given a set of `transfer_to_X()` tools. When an agent determines it has completed its portion of the task, it calls the appropriate transfer tool, which returns a `Command(goto="target_node", update={...})`. LangGraph intercepts this and re-routes execution without returning to a central coordinator.

```
Routing flow:
  - Agent A executes, determines it needs web data
  - Agent A calls transfer_to_search_agent() tool
  - Tool returns Command(goto="search_agent", update={"query": extracted_query})
  - search_agent executes with updated state
  - search_agent may transfer further or terminate
```

**Coupling difference**: Handoff is peer-to-peer; each agent must know the names and contracts of its downstream peers. This is tighter coupling but lower latency (no supervisor round-trip) and fewer LLM calls per hop.

---

### Topology Comparison

```
SUPERVISOR TOPOLOGY                     HANDOFF TOPOLOGY
─────────────────────────────────────   ─────────────────────────────────────
                                        
         ┌──────────────┐               ┌──────────┐    transfer   ┌──────────┐
         │  SUPERVISOR  │               │ Agent A  │─────────────▶│ Agent B  │
         │  (router LLM)│               └──────────┘               └──────────┘
         └──────┬───────┘                    ▲                           │
                │                            │         transfer          │
        ┌───────┼───────┐                    └───────────────────────────┘
        ▼       ▼       ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │Agent A │ │Agent B │ │Agent C │
   └────────┘ └────────┘ └────────┘
        │       │            │
        └───────┴────────────┘
                │ (return to supervisor)

Centralized routing, high observability      Decentralized, lower latency
Single routing LLM call per step             Agents know their peers
Easy to add new agents (update supervisor)   Adding agent = updating all peers
Supervisor is single point of failure        No central bottleneck
Hallucinated routes fail at supervisor       Hallucinated transfers fail at hop
```

### Shared vs. Private State

LangGraph uses `Annotated` reducers on `TypedDict` fields to control how state updates are merged. By default, sub-agents write into the same state dict visible to all nodes. Private state is achieved by composing subgraphs with distinct `StateGraph` schemas — the subgraph's internal working memory (e.g., intermediate retrieval steps) never surfaces to the parent graph unless explicitly included in the subgraph's output schema mapping.

This is architecturally equivalent to the distinction between a microservice's internal DB and its public API contract. A research sub-agent's raw Tavily results, intermediate re-rankings, and scratchpad should be private; only the distilled answer should propagate upward.

### Tavily Integration

Tavily's API returns a list of result objects with `title`, `url`, `content` (snippet), and optionally `raw_content`. The key operational concern: default `search_depth="basic"` returns ~5 snippets with truncated content; `search_depth="advanced"` does deeper crawling but costs more API credits and adds ~1–2s latency. For grounding in a multi-agent context, the search agent should summarize Tavily results before writing to shared state — raw Tavily output is verbose and will balloon context window usage for downstream agents.

### LangSmith Tracing in Multi-Agent Systems

When `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set, every LangGraph node invocation, LLM call, and tool call is logged as a nested run under the root graph execution. In multi-agent graphs:

- The root run corresponds to the `graph.invoke(...)` call
- Each agent node is a child run
- Each LLM call within an agent is a grandchild run
- Tool calls are further nested under their parent LLM run

To debug a routing failure: find the supervisor's run in the tree, inspect its LLM call's output (the structured route decision), and check whether it matches an existing node. A "missing child run" for an expected agent is the canonical symptom of a hallucinated route that was caught by schema validation.

---

## D. Common Interview Questions

---

**Q: How do you prevent a supervisor from entering an infinite routing loop?**

**A:** There are three layers of defense. First, structural: add a step counter to shared state and add a conditional edge that routes to an error/fallback node if `state["steps"] > MAX_STEPS`. Second, prompt-level: instruct the supervisor to emit `FINISH` if it has seen the same sub-agent output twice in a row — explicitly check for this in the supervisor's system prompt. Third, schema-level: if using `with_structured_output`, include a `confidence` or `loop_detected` field in the routing schema so the supervisor can self-report uncertainty. In production, I also instrument a LangSmith alert on traces where the same node appears more than N times in the run tree — that's a canary for loop behavior in newly deployed graphs.

> **Staff-Level Extension**: How do you distinguish a legitimate re-visit (e.g., supervisor legitimately calls search twice for different sub-queries) from a loop? Answer: the step counter alone is insufficient. You need semantic deduplication — hash the (agent, input_state_subset) tuple and track seen states. If the same (agent, input) pair appears twice, it's a loop regardless of step count.

---

**Q: When would you choose handoff over supervisor, and what does that decision cost you in observability?**

**A:** Handoff is preferable when latency is the primary constraint and the agent graph is a relatively linear pipeline (A → B → C with clear preconditions). Removing the supervisor round-trip saves one LLM call per hop — at GPT-4o pricing that's ~$0.005–0.02 per step, which compounds in high-throughput pipelines. The cost is observability: with a supervisor, every routing decision is centralized and auditable in one place in the LangSmith trace. With handoff, routing logic is distributed across all agents, so a misroute requires you to walk the entire run tree to find where the wrong `transfer_to_X` was called. In practice, I use handoff for well-understood, stable pipelines and supervisor for exploratory or dynamically composed workflows where route correctness is more important than raw latency.

> **Staff-Level Extension**: Can you get handoff's latency with supervisor's observability? Partially — you can wrap each `transfer_to_X` tool with a middleware layer that logs the (caller, callee, state_snapshot) tuple to LangSmith as a custom span. This gives you a synthetic "routing log" without the supervisor LLM call.

---

**Q: How do you handle context window exhaustion as state passes through multiple agents?**

**A:** This is one of the subtler production failure modes because it degrades silently — the model doesn't error, it just starts ignoring early context. The mitigation strategy has three levels. First, be aggressive about state schema design: don't accumulate raw messages in shared state; instead, have each agent write a structured summary field (e.g., `research_summary: str`) rather than appending to `messages`. Second, use a `HumanMessage`-only context window for each agent: strip `AIMessage` and `ToolMessage` history before passing context to sub-agents unless they explicitly need it. Third, for long-horizon workflows, introduce a "state compression" node that runs a cheap summarization pass (e.g., `gpt-4o-mini`) before each major agent hop. The measurable signal: monitor token counts per node in LangSmith and alert when any agent's input token count exceeds 60% of the model's context limit.

> **Staff-Level Extension**: What's the architectural analog in distributed systems? It's the "chatty protocol" anti-pattern — each microservice call appends headers and metadata, and by the time you're 10 hops deep, the payload is 80% overhead. The solution is the same: define a clean contract at each service boundary and strip internal working data before forwarding.

---

**Q: A supervisor LLM occasionally routes to agent names that don't exist. How do you make this robust?**

**A:** Three defenses. First, use `with_structured_output` with a `Literal` type that enumerates exactly the valid agent names — the parser will reject anything not in the enum before the conditional edge evaluates. Second, add a fallback in the conditional edge: if `state["next"]` is not in the known node map, route to a default error node rather than raising a `KeyError`. Third, inspect your supervisor prompt — hallucinated routes often happen because the system prompt lists agent names inconsistently or uses aliases. Normalize agent names to snake_case and ensure the prompt, schema, and graph node names are identical strings. In LangSmith, this failure shows up as a supervisor run that completes successfully but has no child runs — the route was emitted but never dispatched.

> **Staff-Level Extension**: If you're using tool-call-based routing (`transfer_to_X` functions), hallucination manifests differently — the model invents a function name that doesn't exist in the tool registry. The mitigation is the same at the schema level, but you also want to monitor tool call error rates in LangSmith's "Tool" run type filter.

---

**Q: How does LangSmith tracing help you debug a multi-agent system versus a single-agent system?**

**A:** In a single-agent system, the run tree is shallow — one LLM call with a few tool calls. In a multi-agent system, the run tree is a forest of nested runs, and the value of LangSmith shifts from "what did the model say?" to "which path did control take and why?" The key debugging workflow: start at the root run, check which child runs were created (which agents executed), then drill into the supervisor's LLM child run to see the raw routing output. If an agent ran unexpectedly, the routing decision is the smoking gun. If an expected agent never ran, either the supervisor hallucinated a different route or the conditional edge logic has a bug. I also use LangSmith's comparison view to diff two traces side-by-side — essential when a routing regression is introduced by a prompt change.

---

**Q: What are the signals that tell you a task should be multi-agent rather than a better-prompted single agent?**

**A:** Three concrete signals. First, tool namespace collision: if a single agent needs 15+ tools, the LLM's tool selection accuracy degrades measurably (empirically, most models start making systematic tool selection errors above ~10 tools). Splitting tools across specialized agents restores precision. Second, context contamination risk: if the task has phases where earlier context actively misleads later reasoning (e.g., a debate-style analysis where you want independent pro/con reasoning), agents with separate context windows are structurally superior to a single agent that has seen both sides. Third, parallelism opportunity: if two sub-tasks are genuinely independent (e.g., retrieve from two different knowledge bases), a `Send` API fan-out in LangGraph can execute them concurrently, which a sequential single agent cannot do. The coordination overhead — roughly one extra LLM call per routing decision — is only worth it if at least one of these three conditions is satisfied.

---

## E. Gotchas, Trade-offs & Best Practices

- **Shared state is a global variable — treat it like one.** The most common source of cross-agent contamination is agents writing to overlapping state keys without coordination. In production, enforce a strict ownership model: each agent has a designated write key (e.g., `research_agent` owns `state["research"]`, `draft_agent` owns `state["draft"]`). Read access is open; write access is scoped. Validate this in code review, not at runtime.

- **The supervisor prompt is your most failure-prone artifact.** Supervisor routing quality degrades when: (a) agent descriptions in the prompt are vague or overlapping, (b) agent names in the prompt don't exactly match graph node names, or (c) the prompt doesn't give the supervisor an explicit `FINISH` condition. Treat the supervisor prompt as a contract document — version it, test it with adversarial inputs, and monitor routing distribution in LangSmith to catch drift.

- **Tool call storms are amplified in multi-agent systems.** In a single-agent system, a storm from one tool call loop is bounded by the single agent's context window. In a multi-agent system, if agent A storms and passes a bloated state to agent B, B may also storm — a cascading failure. Mitigation: implement per-agent `max_iterations` limits at the `AgentExecutor` or node level, and add circuit breaker logic (if total token count across all agent hops exceeds threshold X, terminate with a graceful error).

- **Parallelism via `Send` is powerful but requires careful reducer design.** LangGraph's `Send` API lets you fan out to multiple agents concurrently, but all parallel branches write back to the same state. Without proper `Annotated` reducer definitions (e.g., `list` reducers that append rather than overwrite), parallel writes will race and you'll lose results. This is the same race condition you'd see in concurrent writes to a shared dict in any distributed system — the fix is the same: define merge semantics explicitly.

- **LangSmith in production requires sampling discipline.** Full tracing for every production request is expensive and creates GDPR/data-residency concerns if user data flows through agent state. In production, implement trace sampling: always trace error cases and a random 1–5% of success cases. Use `langsmith.trace(tags=["production", run_id])` to tag traces for filtering and set up alerts on error rate by agent node, not just overall graph error rate.

---

## F. Architecture Pattern — Supervisor Multi-Agent Graph

> The following is a structural / pseudocode-level pattern showing the key design decisions. Working out the exact LangGraph API details — `StateGraph`, `add_node`, `add_conditional_edges`, `Command` — is part of the learning exercise in your assignment notebook.

### State Schema Design

```python
# Shared state — each agent owns specific write keys
class AgentState(TypedDict):
    task: str                        # read-only input
    research_results: str            # owned by: research_agent
    draft_output: str                # owned by: writer_agent
    next: str                        # owned by: supervisor
    steps: int                       # incremented by supervisor
    error: Optional[str]             # written by any agent on failure
```

### Supervisor Routing Schema

```python
# Constrained to valid node names — rejects hallucinated routes at parse time
class RouteDecision(BaseModel):
    next: Literal["research_agent", "writer_agent", "FINISH"]
    reasoning: str  # forces chain-of-thought, improves routing accuracy
```

### Supervisor Node (structural sketch)

```
supervisor_node(state):
    - Guard: if state["steps"] > MAX_STEPS → route to error
    - Build routing prompt from: task, what's been done so far (research_results, draft_output)
    - Call LLM with .with_structured_output(RouteDecision, strict=True)
    - Return: { "next": decision.next, "steps": state["steps"] + 1 }
```

### Graph Topology (structural sketch)

```
StateGraph(AgentState)
  │
  ├── add_node("supervisor",       supervisor_node)
  ├── add_node("research_agent",   research_node)    # has Tavily tool
  ├── add_node("writer_agent",     writer_node)      # has formatting tools
  ├── add_node("error_handler",    error_node)
  │
  ├── set_entry_point("supervisor")
  │
  ├── add_conditional_edges(
  │     "supervisor",
  │     lambda state: state["next"],       # reads routing decision
  │     {
  │       "research_agent": "research_agent",
  │       "writer_agent":   "writer_agent",
  │       "FINISH":          END,
  │       "error":          "error_handler"
  │     }
  │   )
  │
  ├── add_edge("research_agent", "supervisor")   # always return to supervisor
  └── add_edge("writer_agent",   "supervisor")   # always return to supervisor
```

### LangSmith Tracing Configuration

```python
# Environment variables — set before graph compilation
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=<key>
# LANGCHAIN_PROJECT="session-5-multi-agent"

# Tag individual runs for filtering in LangSmith UI
config = {
    "configurable": {"thread_id": run_id},
    "tags": ["supervisor-pattern", "production"],
    "metadata": {"user_id": user_id, "session": "05"}
}
# graph.invoke(initial_state, config=config)
```

### What to look for in the LangSmith Run Tree

```
[ROOT] graph.invoke(...)
  ├── [supervisor] — LLM call → RouteDecision(next="research_agent")
  │     └── [LLM: gpt-4o] — raw structured output visible here
  ├── [research_agent] — Tavily tool call
  │     ├── [LLM: gpt-4o] — query formulation
  │     └── [Tool: tavily_search] — raw API response visible here
  ├── [supervisor] — LLM call → RouteDecision(next="writer_agent")
  ├── [writer_agent] — drafts from research_results
  └── [supervisor] — LLM call → RouteDecision(next="FINISH")

Debugging signal:
  - Missing [research_agent] child run → supervisor hallucinated route, caught by schema
  - [supervisor] appears 10+ times → loop; check MAX_STEPS guard
  - [Tool: tavily_search] token count >> expected → raw results not summarized before state write
```

---

*Study guide generated for Session 5: Multi-Agent Systems with LangGraph. Cross-reference with your `Multi_Agent_Applications_Assignment.ipynb` for hands-on implementation.*
