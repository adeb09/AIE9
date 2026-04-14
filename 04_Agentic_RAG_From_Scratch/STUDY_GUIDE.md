# Agentic RAG from Scratch: LangGraph + ReAct — Staff-Level Interview Study Guide

---

## A. Core Concept Summary

LangGraph is a stateful graph execution framework built on top of LangChain that lets you express agent control flow as a directed graph where nodes are Python callables and edges are transitions between them. Unlike chain-style orchestration, LangGraph models execution as a state machine with explicit cycles — meaning the graph can loop, which is the primitive needed for any reasoning-then-acting pattern. The ReAct loop (Reason → Act → Observe → repeat) maps directly onto this: a reasoning node inspects state and emits either a tool call or a final answer, a tool node executes and appends results, and a conditional edge decides whether to loop or terminate.

Why it matters: most production AI systems are not single-pass pipelines. Retrieval, validation, clarification, and multi-step reasoning all require cycles. LangGraph gives you explicit, inspectable, resumable control flow instead of hiding loops inside recursive LLM calls — which means you can interrupt, checkpoint, replay, and debug at every step. The mental model is: **state flows through the graph, nodes are pure-ish transformers of state, and edges encode the routing logic that determines what happens next.**

---

## B. Key Terms & Definitions

- **`StateGraph`** — LangGraph's core class for building stateful graphs. You register nodes, edges, and conditional edges against it, then compile it into a runnable. It manages the state schema, execution order, and checkpointing.

- **`TypedDict` as state schema** — The state shared across all nodes is typed using Python's `TypedDict`. Every node receives the full state dict and returns a partial dict of fields to update. The graph merges updates back using reducer functions.

- **Reducer / `Annotated` field** — An `Annotated[list, operator.add]` field tells LangGraph how to merge updates from multiple nodes. Without a reducer, a node's output overwrites the field. With `operator.add`, it appends — essential for message accumulation.

- **Conditional edge** — A function that inspects current state and returns a string key that maps to the next node. This is the branching primitive. Without it, all routing is static.

- **`END` node** — A sentinel string constant (`"__end__"`) that terminates graph execution. A conditional edge returning `END` stops the loop.

- **ReAct loop** — Reasoning + Acting: the agent reasons about what tool to call (or whether to stop), calls the tool, observes the result, and repeats. A LangGraph implementation makes each phase an explicit node.

- **`interrupt`** — A LangGraph primitive that pauses graph execution at a defined node and surfaces the current state to the caller. Execution resumes via `graph.invoke()` with updated state, enabling human-in-the-loop (HITL) gates.

- **Checkpointer** — LangGraph's persistence layer (e.g., `MemorySaver`, Postgres-backed). Checkpoints state after each superstep, enabling replay, fault tolerance, and cross-session memory.

- **Superstep** — One full execution cycle through the graph: one node fires, state is updated, the next node is determined. The unit of atomicity in LangGraph execution.

- **Tool node** — A node responsible solely for dispatching tool calls embedded in the last LLM message, collecting results, and appending `ToolMessage` objects back to state.

---

## C. How It Works — Technical Mechanics

### 1. State Schema and Reducers

State is a `TypedDict`. Every node receives the full current state and returns a `dict` of updates. LangGraph merges via reducers:

```
Field without reducer:  node returns {"foo": 2}  →  state["foo"] = 2  (overwrite)
Field with operator.add: node returns {"messages": [msg]}  →  state["messages"] += [msg]  (append)
```

Key insight: if multiple nodes could update the same field in a parallel fan-out, reducers determine how conflicts resolve. For a sequential ReAct loop this is straightforward, but in parallel subgraph topologies it becomes critical.

### 2. Graph Construction Sequence

```
1. Define StateGraph(MyState)
2. Add nodes (name → callable)
3. Set entry point (first node to fire)
4. Add deterministic edges (A → B: always go from A to B)
5. Add conditional edges (A → router_fn → {key: node_name})
6. Compile → runnable with .invoke() / .stream()
```

### 3. ReAct Graph Topology (Text Diagram)

```
         ┌──────────────────────────────────┐
         │                                  │
         ▼                                  │
[START] ──► [reason_node]                   │
                │                           │
                ▼                           │
         [route_after_reason]               │
          /           \                     │
    "tools"          "end"                  │
       │               │                    │
       ▼               ▼                    │
 [tool_node]         [END]                  │
       │                                    │
       └────────────────────────────────────┘
              (append ToolMessages, loop)
```

- `reason_node`: calls the LLM with the current message history; appends the LLM's response (which may contain `tool_calls`) to state.
- `route_after_reason`: inspects the last message — if it has `tool_calls`, route to `tool_node`; otherwise route to `END`.
- `tool_node`: iterates `tool_calls` in the last AI message, executes each tool, appends `ToolMessage` results to state, routes back to `reason_node`.

### 4. How Conditional Routing Works

The router is a plain Python function that receives state and returns a string key. LangGraph maps that key to a node name via a dict passed to `add_conditional_edges`. The function can be arbitrarily complex — inspect message content, count iterations, check a confidence score — but it must be a pure function of state (no side effects).

```
Pattern:
  last_message = state["messages"][-1]
  if hasattr(last_message, "tool_calls") and last_message.tool_calls:
      return "tools"
  return "end"
```

### 5. RAG as a LangGraph Tool vs. Standalone Node

| Approach | When to use | Trade-offs |
|---|---|---|
| **Retriever as a tool** | Agent decides when/whether to retrieve | Flexible; agent can skip retrieval if answer is in context. Risk: agent hallucinates without retrieving. |
| **Retriever as a node** | Retrieval is always required (e.g., grounded QA) | Deterministic; no agent discretion. Less flexible but more auditable. |
| **Hybrid: retrieval node + re-ranking tool** | Production systems with quality SLAs | Guarantees retrieval happens, but lets agent request more results or filter. |

For Agentic RAG specifically, RAG-as-tool is the right default: the agent can call it zero, one, or multiple times with different queries — enabling multi-hop retrieval and query decomposition.

### 6. `interrupt` and HITL Mechanics

`interrupt` is declared at compile time via `interrupt_before=["node_name"]` or `interrupt_after=["node_name"]`. When execution reaches that node:

1. Graph pauses — state is checkpointed.
2. The `.invoke()` call returns with a `GraphInterrupt` exception or the current state (depending on API version).
3. Human reviews/modifies state.
4. Caller resumes via `graph.invoke(Command(resume=...))` with the same `thread_id`.

Key: interrupts require a checkpointer. Without one, there's no way to resume because the state isn't persisted between the pause and resume calls.

### 7. State Machine Analogy — Where It Breaks Down

LangGraph *is* a state machine with these qualifications:

- **States are not enumerable** — the "state" is a rich dict, not a finite set of modes.
- **Transitions aren't purely deterministic** — conditional edges can call an LLM, which is non-deterministic.
- **Supersteps can fan out** — you can send to multiple nodes in parallel (unlike classical FSMs).
- **The graph can be dynamically modified** — subgraphs, map-reduce patterns, and dynamic tool registration break the pure FSM model.

Think of it as an **executable DAG that allows cycles**, not a classical FSM.

---

## D. Common Interview Questions (with Strong Answers)

---

**Q: Why build ReAct from scratch with LangGraph instead of using `create_react_agent`?**

**A:** `create_react_agent` is a convenience wrapper that handles the common case, but it hides the routing logic, tool dispatch, and state schema behind an opaque abstraction. In production, you almost always need to customize: maybe you want to inject a confidence gate before tool execution, add a retry node for failed tool calls, or interrupt for human approval before certain tool types. If you've built the loop manually, you know exactly where to insert those hooks. The abstraction also makes debugging harder — when something goes wrong at step 4 of a 10-step loop, you want to know *exactly* which node emitted what. Building from scratch also gives you control over the state schema, which matters the moment you need to carry non-message state (e.g., a retrieved context window, a loop counter for max-iterations guard, a citation list).

> **Staff-Level Extension:** How would you enforce a maximum iteration budget without a loop counter in state? You'd have to inspect message history length, which is fragile. What happens if you're using `operator.add` with no max-iter guard and the LLM gets stuck in a tool-calling loop? You've built an unbounded compute hole. Production systems need an explicit `iteration_count: int` field with a hard cap in the routing function.

---

**Q: How does state management in LangGraph differ from passing context through a LangChain chain, and why does it matter at scale?**

**A:** LangChain chains pass context via explicit return values threaded through a pipeline — it's functional composition, and state lives in the call stack. LangGraph externalizes state as a first-class object with a defined schema and persistence layer. This matters at scale for several reasons: first, you can checkpoint and replay from any superstep without re-running the entire graph — critical for long-running agents that might fail midway. Second, multiple concurrent executions don't share mutable state — each `thread_id` gets its own state slot in the checkpointer. Third, the schema forces you to be explicit about what data flows through the system, which is essential for auditability in regulated environments. The trade-off is that TypedDict schemas are rigid — adding a field mid-production requires careful migration logic, similar to a database schema change.

> **Staff-Level Extension:** What happens if two nodes in a parallel fan-out both update the same state field with `operator.add`? LangGraph executes them in a single superstep and merges all updates after both complete. This means the order of merge is non-deterministic unless you control it through the reducer. For message lists this is usually fine, but for numeric accumulators or deduplication-sensitive data it's a subtle correctness bug.

---

**Q: You're building a RAG agent that needs to retrieve from multiple sources (vector DB, SQL, web). How do you architect the tool layer?**

**A:** I'd expose each retriever as a separate tool with a clear semantic contract — `retrieve_from_vectordb(query: str)`, `query_sql(sql: str)`, `web_search(query: str)` — and let the reasoning node's LLM decide which to call and in what order. The key architectural decisions: first, tool schemas must be precise enough that the LLM can select the right one without trial and error (poor schemas are the #1 source of tool-calling failures I've seen). Second, I'd add a result-scoring/filtering node after tool dispatch that re-ranks retrieved chunks before appending to state — the agent shouldn't make decisions based on raw unfiltered retrieval. Third, I'd instrument each tool call with latency and result-quality metrics, because in production you'll quickly discover that web search is the bottleneck and SQL is often mis-queried. The graph topology would be: reason → tool dispatch → [parallel: vectordb, sql, web] → merge/rerank → reason (loop).

> **Staff-Level Extension:** How do you handle tool call failures in the graph? A tool that throws an exception will crash the superstep. You need a try/except wrapper in the tool node that converts exceptions into `ToolMessage` objects with an error content field. The LLM then sees the error and can retry with a modified query — but you need to be careful that the retry budget is finite.

---

**Q: Explain `interrupt` in LangGraph. When would you use it in a production agentic RAG system and when is it the wrong tool?**

**A:** `interrupt` pauses execution at a declared node and surfaces current state to the caller. It's the right tool when you need human judgment before irreversible actions — for example, before a write-to-database tool executes, or before sending a drafted email. In a RAG context, a useful HITL gate is after the reasoning node proposes a retrieval strategy but before execution — a human can validate that the query decomposition makes sense before burning retrieval budget. The wrong use cases are high-throughput, low-stakes retrieval (interrupting every query is a latency and UX disaster) and any flow where the human-in-the-loop adds latency that violates an SLA. The deeper issue is operational: interrupts require your system to maintain session state between the pause and resume, which means your serving infrastructure needs a stateful execution model. If you're running on stateless Lambda functions, you need the Postgres checkpointer, not `MemorySaver`.

> **Staff-Level Extension:** How do you handle the case where a graph is interrupted and the human never resumes? You need TTL-based cleanup on stale checkpoints, and you need to surface interrupted graphs in an operator dashboard. This is infrastructure work that `interrupt` alone doesn't solve.

---

**Q: When would you choose Ollama + a local model over an API model like GPT-4o for your LangGraph agent?**

**A:** The decision is multi-dimensional. **Data privacy** is the strongest forcing function — if the documents being retrieved contain PII, PHI, or trade secrets, you can't send them to an external API without legal exposure. **Cost at scale** is the second driver — at 10M tokens/day, the per-token cost of API models dominates infra cost; a quantized 13B model on a single A10 is cheaper at that volume. **Latency** is more nuanced — local models avoid network RTT but are slower per-token on consumer hardware; on a well-provisioned GPU server, local inference can match or beat API latency for short sequences. **Quality** is where local models lose for complex multi-step reasoning — a 7B or 13B model will fail on tool use and multi-hop reasoning that GPT-4o handles reliably. My pragmatic heuristic: use local models for the retrieval-heavy, tool-light steps (e.g., chunk scoring, query rewriting) and API models for the reasoning steps that determine control flow.

> **Staff-Level Extension:** Ollama exposes an OpenAI-compatible `/v1/chat/completions` endpoint. This means you can point LangChain's `ChatOpenAI` at `http://localhost:11434/v1` with no code changes. But the compatibility is surface-level — structured output (JSON mode, tool calling) has quality gaps in smaller models. If your ReAct loop depends on reliable tool_call JSON, you need to benchmark the specific model, not assume compatibility means functional equivalence.

---

**Q: How do you debug a LangGraph graph that's behaving unexpectedly — e.g., the agent loops infinitely or terminates prematurely?**

**A:** The debugging stack has three layers. First, `graph.get_graph().draw_mermaid()` gives you the topology — premature termination is often a misconfigured conditional edge that always returns `END` (I've seen this from a typo in the router's return string). Second, `.stream()` instead of `.invoke()` lets you inspect state after every superstep — you can print the state dict at each step and see exactly what the LLM returned and what the router decided. Third, for production, I'd attach LangSmith (or equivalent) tracing, which captures every node's input/output with timing. Infinite loops are almost always a routing logic bug — either the conditional edge never returns `END` because `tool_calls` is always populated, or the LLM is stuck in a pattern where it keeps generating the same tool call because the tool result doesn't change its reasoning. The fix for the latter is a max-iteration guard in the router: `if state["iteration_count"] >= MAX_ITERS: return "end"`.

---

## E. Gotchas, Trade-offs & Best Practices

- **Unbounded loops are a production outage waiting to happen.** The ReAct loop has no inherent termination guarantee. Always add an explicit `iteration_count` field to state and a hard cap in the routing function. A stuck LLM + no iteration limit = runaway token spend. This is the #1 operational failure mode in production agentic systems.

- **Reducer semantics are invisible until they bite you.** The difference between overwrite and `operator.add` on a field is a one-line annotation, but getting it wrong produces subtle bugs: messages disappearing (wrong reducer), duplicate messages accumulating (reducer applied when it shouldn't be), or state corruption in parallel subgraphs. Document your state schema with explicit comments on every field's reducer intent.

- **Tool schema quality dominates tool-calling reliability.** LLMs pick tools based on the name, description, and parameter schema. Vague descriptions like `"search for information"` produce inconsistent tool selection. Precise descriptions like `"retrieve semantically similar document chunks from the product catalog vector store, given a natural language query"` dramatically improve reliability. This is worth more than model size in my experience.

- **Checkpointer choice is an infrastructure decision, not a library choice.** `MemorySaver` is in-process and lost on restart — fine for development, a footgun in production. `SqliteSaver` is single-node. Postgres-backed checkpointing is the production default but requires you to manage schema migrations when your state TypedDict changes. Version your state schema or you'll corrupt existing checkpoints on deployment.

- **Local models via Ollama require explicit tool-calling benchmarking.** The OpenAI API compatibility layer doesn't mean the model reliably generates valid tool call JSON. Smaller quantized models often hallucinate parameter names or omit required fields. Always validate tool calls in the tool node and return a structured error `ToolMessage` on schema validation failure — never let an invalid tool call crash the superstep.

---

## F. Code or Architecture Pattern

Rather than a complete implementation (which you should build from the mechanics above), here's the **architectural skeleton with the three critical decision points** annotated.

### State schema — the two fields every ReAct graph needs

```python
# The reducer on messages is the core primitive.
# Without operator.add, each node would overwrite the list.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    iteration_count: int  # no reducer → overwrite; node increments explicitly
```

### The routing function — this is where correctness lives

```python
# This function is the entire control flow of the agent.
# It must handle: tool call present, no tool call, and max-iter exceeded.
def route_after_reason(state: AgentState) -> str:
    last = state["messages"][-1]
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "end"               # hard stop
    if getattr(last, "tool_calls", None):
        return "tools"             # continue loop
    return "end"                   # natural termination
```

### Graph wiring — the minimal topology

```python
graph = StateGraph(AgentState)
graph.add_node("reason", reason_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("reason")
graph.add_conditional_edges("reason", route_after_reason, {"tools": "tools", "end": END})
graph.add_edge("tools", "reason")   # the loop-back edge
runnable = graph.compile(checkpointer=MemorySaver())
```

### RAG tool wiring — the key pattern

```python
# Retriever becomes a tool via a thin wrapper.
# The agent calls it like any other tool; the graph doesn't need a dedicated retrieval node.
@tool
def retrieve(query: str) -> str:
    """Retrieve relevant document chunks from the knowledge base."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

# Bind to the LLM in the reason_node, not at graph construction time.
# llm_with_tools = llm.bind_tools([retrieve, ...])
```

### HITL interrupt — one-line addition to compile

```python
# interrupt_before fires before the named node executes.
# State is checkpointed; caller receives control.
runnable = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]  # human approves every tool call
)
```

---

> Work through building each of these pieces yourself — specifically the `reason_node` (where you construct the prompt, call the LLM, and increment `iteration_count`) and the `tool_node` (where you iterate `tool_calls`, dispatch, and construct `ToolMessage` objects). Those two functions, plus the routing logic above, are what an interviewer will probe on. Understanding *why* each line is there is more valuable than memorizing the complete implementation.
