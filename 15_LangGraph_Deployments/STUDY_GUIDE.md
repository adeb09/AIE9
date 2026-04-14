# Session 15: LangGraph Deployments (Serving Agents)

> **Audience:** Senior / Staff AI Engineer interview prep  
> **Prerequisite assumed:** LangGraph graph construction, LangChain tool patterns, REST API design, Docker, distributed state

---

## A. Core Concept Summary

LangGraph Platform bridges the gap between a working graph prototype and a production-grade, horizontally-scalable agent service. It wraps any compiled `StateGraph` in an HTTP server exposing standardized REST endpoints for runs, threads, and assistants — enabling polyglot clients, replay, resumability, and operational observability without requiring teams to write that infrastructure themselves.

The central mental model: **a LangGraph deployment is a typed, versioned API over a stateful computation graph**. The graph defines *what* can happen; the checkpointer defines *whether* state survives between requests; the assistant layer defines *how* a graph is configured for a specific use case. These three axes are orthogonal and compose independently.

Understanding where state lives (in-memory MemorySaver → Postgres/Redis) is the single most consequential architectural decision in LangGraph deployments. It determines cost, scaling topology, failure recovery, and whether you can run more than one server replica at all.

`langgraph dev` collapses the full deployment lifecycle into a single CLI command suitable for inner-loop development, exposing the same REST contract that LangGraph Cloud and self-hosted Docker images expose — so test clients written against localhost work identically against production without modification.

---

## B. Key Terms & Definitions

- **`langgraph dev`** — CLI command that launches a local LangGraph API server (default port **2024**) from a `langgraph.json` manifest, with hot-reload on file changes. It runs an in-process uvicorn server and registers all graphs declared in the manifest as assistants.

- **`langgraph.json`** — The deployment manifest. Declares which Python import paths map to compiled graph objects (`graphs`), which env file to load (`env`), which packages to install (`dependencies`), and optionally pre-configured assistants (`assistants`). It is the single source of truth the CLI and Docker builder both read.

- **Assistant** — A named, versioned, configurable binding between a graph and a set of runtime configuration defaults. Multiple assistants can reference the same underlying graph with different `config` payloads, allowing behavioral parameterization without code forks (e.g., a `gpt-4o` assistant and a `gpt-4.1-mini` assistant over the same graph).

- **Thread** — A persistent session object, identified by a UUID, that ties together a sequence of runs against a shared checkpointed state. Stateful agents require a thread; stateless (threadless) runs pass `None` as the thread ID.

- **Run** — A single invocation of an assistant against an optional thread. The `/runs` endpoint accepts input, config, and stream mode. Runs can be synchronous (blocking), streaming (SSE), or background (fire-and-forget).

- **Checkpointer** — The persistence adapter attached to a compiled graph. Determines whether state is saved between runs. `MemorySaver` is process-local (dev only). `AsyncPostgresSaver` and `AsyncRedisSaver` are network-accessible and required for multi-replica deployments.

- **`RemoteGraph`** — A LangGraph SDK client object that proxies calls to a deployed LangGraph server as if the remote graph were a local `CompiledGraph`. Supports `invoke`, `stream`, and `astream` with the same interface.

- **`stream_mode`** — Controls the granularity of SSE chunks during streaming. `"updates"` emits each node's output delta; `"values"` emits the full state after each step; `"messages"` emits token-level LLM output for chat UX.

- **FastMCP / MCP Tools** — The Model Context Protocol allows external processes to serve tools over a transport (stdio, SSE). A LangGraph graph can load MCP tools at startup via `MultiServerMCPClient`, making external tool surfaces composable with the agent's native tool belt without embedding the tool logic in the graph package.

- **ToolNode** — LangGraph's prebuilt node that executes all `tool_calls` present on the last `AIMessage` in state, dispatches to the matching tool, and appends `ToolMessage` results. It handles parallel tool calls and error marshaling automatically.

---

## C. How It Works — Technical Mechanics

### The Full Deployment Lifecycle

```
langgraph.json
    │
    ▼
langgraph dev / langgraph build
    │  Reads manifest, installs deps (via uv), loads graphs, starts uvicorn
    ▼
LangGraph API Server  :2024
    │
    ├── POST /assistants          → create / upsert assistant with config
    ├── GET  /assistants/{id}     → fetch assistant metadata
    ├── POST /threads             → create a thread (session)
    ├── POST /threads/{id}/runs   → create a run on a thread
    ├── POST /runs                → stateless (threadless) run
    ├── GET  /runs/{id}/stream    → SSE stream of run events
    └── GET  /threads/{id}/state  → inspect checkpointed state
    │
    ▼
Client (langgraph_sdk / RemoteGraph / raw HTTP)
```

### `langgraph.json` Field Semantics

```json
{
  "version": 1,
  "dependencies": ["."],          // pip/uv install targets; "." = local package
  "env": ".env",                   // dotenv file loaded into server process env
  "python_version": "3.13",        // Python interpreter constraint for Docker build
  "graphs": {
    "simple_agent": "app.graphs.simple_agent:graph"
    // key = graph_id referenced by assistants & SDK calls
    // value = "module.path:compiled_graph_object"
  },
  "assistants": {
    "agent": {
      "graph_id": "simple_agent",  // must match a key in `graphs`
      "name": "Simple Agent",
      "description": "..."
    }
  }
}
```

The `graphs` key is authoritative: the `graph_id` referenced in an assistant must exactly match a key here. The value is a Python import path — the module is imported at server startup, and the named attribute must be a `CompiledGraph`. The `dependencies` field accepts anything pip/uv accepts: `"."` installs the local package from `pyproject.toml`, but you can mix in `".[extras]"`, path installs, or pinned packages.

### Hot Reload Behavior

`langgraph dev` uses watchfiles to monitor the working directory. On any `.py` change it restarts the uvicorn worker without restarting the manifest-parsing or env-loading layers. This means code changes take effect in ~1 s but changes to `langgraph.json` itself (adding a new graph, changing env file path) require a full CLI restart. This distinction bites teams who add a new graph to the manifest and wonder why the endpoint 404s.

### Assistants vs. Graphs

A `graph_id` is static (it's a code artifact). An `assistant_id` is a runtime entity. The same `graph_id` can be instantiated as many assistants as needed, each carrying different `config` payloads:

```python
# Two assistants over the same graph, different model configs
await client.assistants.create(
    graph_id="simple_agent",
    config={"configurable": {"model": "gpt-4o"}},
    name="premium-agent",
)
await client.assistants.create(
    graph_id="simple_agent",
    config={"configurable": {"model": "gpt-4.1-mini"}},
    name="economy-agent",
)
```

At invocation time the `config` is merged (run-level config overrides assistant-level config), then injected into the graph's `RunnableConfig`. The graph reads it via `config["configurable"]`.

### Run Lifecycle (Stateless vs. Stateful)

**Stateless (threadless):** Pass `thread_id=None`. The server allocates an ephemeral in-memory state for the duration of the run, discards it on completion. No checkpointer is involved even if one is compiled into the graph. Useful for single-turn agents or when the caller manages conversation history externally.

**Stateful (threaded):** Pass a thread UUID. The checkpointer is called at every `SuperStep` boundary: state is serialized and persisted. On the next run against the same thread, state is hydrated from the store. Enables human-in-the-loop interrupts (`interrupt_before`, `interrupt_after`), time-travel replay, and multi-turn memory.

### Streaming Mechanics

The SDK's `client.runs.stream(...)` returns a generator of `StreamPart` objects. Each part has an `.event` type (e.g., `"updates"`, `"metadata"`, `"error"`) and `.data` containing the node output or state delta. Under the hood this is an SSE (Server-Sent Events) connection — the server holds the HTTP response open and flushes newline-delimited JSON chunks as each graph node completes.

```python
for chunk in client.runs.stream(thread_id, assistant_id, input=..., stream_mode="updates"):
    if chunk.event == "updates":
        # chunk.data is {"node_name": {state_delta}}
        process_delta(chunk.data)
    elif chunk.event == "error":
        raise RuntimeError(chunk.data)
```

### FastMCP Tool Integration

MCP tools are loaded at graph startup (module import time) using `MultiServerMCPClient`. The client connects to one or more MCP servers, fetches their tool manifests, and returns LangChain-compatible `BaseTool` objects that can be bound to a model or dropped into a `ToolNode`. The critical decision is **startup vs. request-time loading**: loading at startup (module level) means tool manifests are cached for the lifetime of the server process; loading per-request enables dynamic tool sets but adds latency to every invocation.

```python
from mcp import MultiServerMCPClient

async def load_mcp_tools():
    client = MultiServerMCPClient({
        "filesystem": {"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]},
    })
    return await client.get_tools()

# Called once at module import; result cached in module scope
import asyncio
MCP_TOOLS = asyncio.get_event_loop().run_until_complete(load_mcp_tools())
```

---

## D. Common Interview Questions (with Strong Answers)

---

**Q: Walk me through what happens from `langgraph dev` to a client receiving the first streaming token.**

**A:** `langgraph dev` reads `langgraph.json`, resolves `dependencies` via `uv pip install`, imports each module path in `graphs` to retrieve compiled graph objects, then starts a uvicorn server on port 2024. When a client POSTs to `/runs/stream` with an assistant ID and input, the server looks up the assistant's `graph_id` and merges the run-level `config` on top of the assistant's stored config. It then calls `graph.astream(input, config=merged_config, stream_mode=...)`, which executes the graph node-by-node. After each node completes, the `SuperStep` output is serialized to JSON and flushed as an SSE chunk. If a checkpointer is attached, state is persisted before the flush. The client's SSE connection receives each chunk as a `data:` event, which the SDK's `stream()` generator deserializes into `StreamPart` objects. The first token therefore appears as soon as the first node that emits an LLM response finishes — typically the `agent` node after one LLM call.

> **Staff-Level Extension:** What determines the P99 latency of that first token? It's the sum of: (1) server-side routing overhead (negligible, ~1ms), (2) checkpointer read latency if stateful (Postgres round-trip: 5–20ms), (3) TTFT from the upstream LLM provider (dominant term, 200ms–2s depending on provider and model). If you're building a chat product, the checkpointer read is on the critical path — which argues for Redis (sub-ms reads) over Postgres for high-frequency conversational threads.

---

**Q: When would you use `MemorySaver` vs. `AsyncPostgresSaver` vs. `AsyncRedisSaver`?**

**A:** `MemorySaver` is strictly a development tool. It stores state in a Python dict inside the server process — any restart loses all threads, and it can't be shared across replicas. The moment you deploy more than one instance (even two containers behind a load balancer), you need a network-accessible checkpointer. `AsyncPostgresSaver` is the right choice when thread state must be durable, queryable, and consistent: it survives server restarts, supports point-in-time replay, and can serve as an audit log. The cost is ~10–30ms per checkpoint write. `AsyncRedisSaver` trades durability for latency — checkpoint reads/writes are sub-millisecond, making it ideal for high-throughput, short-lived sessions where losing state on a Redis eviction is acceptable (or where you configure `appendonly yes` for durability). My rule of thumb: use Postgres for compliance-sensitive or long-running threads (days/weeks), Redis for sub-second response chat UX with session TTLs.

> **Staff-Level Extension:** There's a third consideration: checkpoint serialization size. LangGraph checkpoints serialize the entire graph state, including the full message history, as JSON blobs. For long conversations this can reach megabytes per checkpoint. In Postgres this inflates row sizes and query times; in Redis it can push keys above the `maxmemory` eviction threshold. Both cases argue for a message compaction strategy — periodically summarizing old messages and truncating the raw history — which should be a first-class concern in the graph design, not retrofitted after production incidents.

---

**Q: How do you safely parameterize agent behavior (e.g., model selection, system prompt) across environments without code changes?**

**A:** LangGraph's `config["configurable"]` dict is the canonical mechanism. The graph reads `config["configurable"].get("model", DEFAULT_MODEL)` at node execution time; the assistant definition in `langgraph.json` provides the default config; and the run-level `config` passed by the client can override any key at invocation time. This three-layer merge (graph default → assistant config → run config) means you can have a single graph binary that serves dev (cheap model, verbose logging system prompt), staging (production model, standard prompt), and production (production model, concise prompt) purely through configuration, with no code branches or separate deployments. Environment variables handle secrets (API keys) and should never flow through `configurable` — they belong in `.env` loaded by the `env` field in `langgraph.json`.

> **Staff-Level Extension:** The failure mode here is config schema drift: a graph is updated to expect a new `configurable` key, but assistants in production still carry the old schema, and the key silently defaults to `None`. You can guard against this by defining a Pydantic model for `configurable`, validating it at node entry, and failing fast with a descriptive error rather than silently using a default. LangGraph doesn't enforce this out of the box — it's an architectural discipline you have to impose.

---

**Q: Your stateful LangGraph agent is deployed behind a load balancer with three replicas and `AsyncPostgresSaver`. Under what conditions can you get a state corruption or stale-read bug?**

**A:** The canonical failure mode is a concurrent write race: two requests for the same `thread_id` arrive within the same checkpoint interval, both read the same state snapshot from Postgres, both execute, and both try to write back. The second write overwrites the first's state delta. LangGraph's `AsyncPostgresSaver` uses optimistic concurrency control — each checkpoint has a `version` integer; a write fails with a conflict error if the version in the DB doesn't match what was read. This means the second run gets an exception, not silent corruption. You need to handle this at the API layer: either serialize requests per thread (a per-thread queue or Redis lock), or surface the conflict to the client and let them retry. The more subtle bug is stale reads during a long-running stream: if a run takes 30 seconds and a second run starts against the same thread mid-stream, the second run reads a state snapshot that's 30 seconds stale. Whether this is acceptable depends entirely on the application semantic.

> **Staff-Level Extension:** The principled solution is per-thread request serialization — route all requests for a given `thread_id` to the same server instance (consistent hashing at the load balancer) and process them through a per-thread asyncio lock. This eliminates the Postgres conflict race and the stale-read race. The cost is that it constrains horizontal scaling: you can't add replicas to reduce per-thread latency, only to increase total thread throughput. For most agent workloads (one active turn per thread at a time) this is fine. For branching / parallel subgraphs within a single thread it requires more careful design.

---

**Q: How would you structure a LangGraph deployment that needs to consume tools from an external MCP server that you don't control?**

**A:** The key decision is whether the MCP server's tool manifest is stable. If the tool list is static (you control the MCP server or it versions its manifest), load tools at module import time, cache them in a module-level variable, and bind them to the model during `build_graph()`. This gives you zero per-request overhead and a predictable tool surface. If the manifest is dynamic (tools change at runtime, e.g., a user-specific tool server), you must load tools inside the node function on each invocation, which adds MCP round-trip latency (~10–50ms for a local stdio server) to every agent step. In both cases, MCP tool errors need explicit handling — MCP tools can return error results that don't raise Python exceptions, so your `ToolNode` error handling needs to inspect `ToolMessage.status == "error"` and route accordingly rather than catching Python exceptions alone.

> **Staff-Level Extension:** A third architecture is a tool-catalog sidecar: a separate service that polls MCP servers, caches their manifests, and exposes a REST endpoint your graph calls at startup. This decouples tool manifest freshness from graph deployment cycles and lets you invalidate the cache without redeploying the graph. It also lets you apply ACL logic (which threads can access which tools) at the catalog layer rather than inside each graph.

---

**Q: What's the operational cost of stateful checkpointing at scale, and how do you decide whether a given agent workflow needs it?**

**A:** Every checkpoint write is a serialization + network round-trip. At 20ms per checkpoint and 10 nodes per run, you're adding 200ms of pure I/O overhead per invocation — before any LLM latency. For an agent that costs 2 seconds in LLM time, this is 10% overhead; for a sub-200ms retrieval agent, it can double latency. Beyond latency, persistent state has storage costs: a 10-turn conversation with tool results can produce a 50KB–500KB checkpoint blob. At 1M threads/day that's 50–500 GB/day of checkpoint writes. My framework: you need stateful checkpointing if (a) the user or system must resume a partially-complete multi-turn workflow, (b) you need human-in-the-loop approval steps with arbitrary delay between turns, or (c) compliance requires an immutable audit trail of agent decisions. For single-turn question-answering agents, stateless runs are strictly cheaper and simpler.

> **Staff-Level Extension:** There's a hybrid strategy worth knowing: run the graph stateless, but persist only the final output and the initial input to a separate application DB. This gives you an audit trail and basic replay capability at a fraction of the storage cost of full checkpoint history. You lose mid-graph resumability, but for workflows that don't need it this is the right trade. LangGraph's checkpointer abstraction makes it straightforward to swap in this pattern by implementing a custom `BaseCheckpointSaver` that writes only terminal states.

---

## E. Gotchas, Trade-offs & Best Practices

- **`langgraph.json` graph keys are runtime IDs, not just labels.** The string you put in `graphs` is the `graph_id` that clients reference in API calls and that assistants bind to. Renaming a key is a breaking change to all clients and all persisted assistant records. Treat graph IDs like API route paths: version them (`simple_agent_v2`) rather than mutate them.

- **`MemorySaver` silently passes in multi-replica deployments until it fails catastrophically.** If your staging environment runs a single replica and your production environment runs three, thread state will appear to work in staging and mysteriously vanish (or interleave) in production. The only safe policy is to use `AsyncPostgresSaver` or `AsyncRedisSaver` everywhere except local dev, and to explicitly document which checkpointer is expected in each environment via an environment variable (`CHECKPOINTER_BACKEND=postgres|redis|memory`).

- **Streaming `stream_mode="updates"` vs `"values"` has meaningfully different downstream semantics.** `"updates"` gives you the minimal delta — great for chat UX where you're appending to a message thread. `"values"` gives you the full state after each step — easier to debug but 5–10x more bytes over the wire for large message histories. Using `"values"` in production for high-throughput applications can saturate network bandwidth between service and client before LLM latency becomes the bottleneck.

- **Tool loading side effects at module import time are invisible errors.** If `get_tool_belt()` makes an HTTP call (to an MCP server, a tool registry, etc.) and that call fails, the LangGraph server silently fails to import the module and the graph never registers. The `/assistants` endpoint returns 404 for that graph ID with no useful error message in the default log level. Always add explicit startup health checks that verify every tool dependency is reachable, and log failures with `logging.error` at import time so they surface in container stdout.

- **`uv` dependency resolution in `langgraph.json` uses the project's `uv.lock` if present; bypassing it risks environment drift.** The `dependencies: ["."]` pattern installs the local package and respects `pyproject.toml`. But if you add a package to `dependencies` in `langgraph.json` that isn't in `pyproject.toml`, it gets installed outside the lock file, creating a reproducibility hole. Keep `langgraph.json` dependencies minimal (ideally just `"."`) and manage all packages through `pyproject.toml` + `uv.lock`.

---

## F. Code & Architecture Patterns

### 1. `langgraph.json` — Production-Ready Config

```json
{
  "version": 1,
  "dependencies": ["."],
  "env": ".env",
  "python_version": "3.13",
  "graphs": {
    "simple_agent":          "app.graphs.simple_agent:graph",
    "agent_with_helpfulness": "app.graphs.agent_with_helpfulness:graph",
    "simple_summary_agent":  "app.graphs.simple_summary_agent:graph"
  },
  "assistants": {
    "agent": {
      "graph_id": "simple_agent",
      "name": "Simple Agent",
      "description": "Tool-calling agent with conditional routing."
    },
    "agent_helpful": {
      "graph_id": "agent_with_helpfulness",
      "name": "Agent with Helpfulness Check",
      "description": "Agent with post-response helpfulness evaluation loop."
    },
    "agent_summary": {
      "graph_id": "simple_summary_agent",
      "name": "Agent with Summary",
      "description": "Agent that appends a tool-call summary to its final response."
    }
  }
}
```

### 2. Stateful Agent Graph with Postgres Checkpointer

```python
"""Production-ready stateful agent with AsyncPostgresSaver checkpointer."""

from __future__ import annotations
import os
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.models import get_chat_model
from app.state import MessagesState
from app.tools import get_tool_belt


def call_model(state: MessagesState) -> dict:
    model = get_chat_model().bind_tools(get_tool_belt())
    return {"messages": [model.invoke(state["messages"])]}


def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("action", ToolNode(get_tool_belt()))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", tools_condition, {"tools": "action", END: END}
    )
    graph.add_edge("action", "agent")
    return graph


async def create_compiled_graph():
    """Called once at server startup; checkpointer is shared across all requests."""
    checkpointer_backend = os.environ.get("CHECKPOINTER_BACKEND", "memory")

    if checkpointer_backend == "postgres":
        db_url = os.environ["POSTGRES_CONNECTION_STRING"]
        checkpointer = AsyncPostgresSaver.from_conn_string(db_url)
        await checkpointer.setup()  # Creates tables if not present
    else:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    return build_graph().compile(checkpointer=checkpointer)


# LangGraph Platform calls this to get the compiled graph object.
# For async initialization, expose via an async factory if supported,
# otherwise initialize synchronously via asyncio.run() at module level.
import asyncio
graph = asyncio.get_event_loop().run_until_complete(create_compiled_graph())
```

### 3. Test Client — `RemoteGraph` Pattern (Recommended)

```python
"""RemoteGraph client: treats the deployed server as a local graph object."""

from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import HumanMessage

# RemoteGraph proxies invoke/stream/astream to the remote server.
# Same interface as a local CompiledGraph — no HTTP boilerplate.
remote_graph = RemoteGraph(
    "simple_agent",                   # graph_id (must match langgraph.json `graphs` key)
    url="http://localhost:2024",
)

# Stateless (threadless) invocation — no persistence, simplest path
result = remote_graph.invoke({
    "messages": [HumanMessage(content="How often should I deworm my cat?")]
})
print(result["messages"][-1].content)


# Stateful streaming invocation with an explicit thread ID
import uuid
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

for chunk in remote_graph.stream(
    {"messages": [HumanMessage(content="What vaccines does my cat need?")]},
    config=config,
    stream_mode="updates",
):
    for node_name, node_output in chunk.items():
        if "messages" in node_output:
            for msg in node_output["messages"]:
                print(f"[{node_name}] {getattr(msg, 'content', msg)}")

# Continue the same thread (state hydrated from checkpointer)
for chunk in remote_graph.stream(
    {"messages": [HumanMessage(content="And for a dog?")]},
    config=config,                    # same thread_id → state resumes
    stream_mode="updates",
):
    for node_name, node_output in chunk.items():
        if "messages" in node_output:
            for msg in node_output["messages"]:
                print(f"[{node_name}] {getattr(msg, 'content', msg)}")
```

### 4. Raw SDK Streaming Client (Low-Level, from Session)

```python
"""Direct SDK client — full control over streaming events."""

from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:2024")

for chunk in client.runs.stream(
    None,             # thread_id=None → stateless (threadless) run
    "simple_agent",   # assistant_id from langgraph.json `assistants` key
    input={
        "messages": [{"role": "human", "content": "How often should I deworm my cat?"}]
    },
    stream_mode="updates",
):
    print(f"[{chunk.event}]", chunk.data)
```

### 5. FastMCP Tool Integration at Startup

```python
"""Load MCP tools at module import and expose them to the agent."""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

_MCP_CONFIG = {
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"],
    },
    "brave_search": {
        "transport": "sse",
        "url": "http://localhost:8080/sse",  # external MCP SSE server
    },
}


async def _load_mcp_tools():
    async with MultiServerMCPClient(_MCP_CONFIG) as mcp_client:
        return mcp_client.get_tools()


# Module-level cache; shared across all requests in this server process.
# If MCP servers are unavailable at startup, this raises — fail fast, not silently.
MCP_TOOLS = asyncio.get_event_loop().run_until_complete(_load_mcp_tools())


def get_full_tool_belt():
    """Return native tools + MCP tools for model binding."""
    from app.tools import get_tool_belt
    return get_tool_belt() + MCP_TOOLS
```

---

## Quick Reference: Decision Matrix

| Dimension | Choice A | Choice B | Deciding Factor |
|---|---|---|---|
| **Checkpointer** | `MemorySaver` | `AsyncPostgresSaver` / `AsyncRedisSaver` | More than 1 replica, or need durability |
| **Thread mode** | Stateless (`thread_id=None`) | Stateful (UUID thread) | Multi-turn memory or HITL interrupts needed |
| **Stream mode** | `"updates"` | `"values"` | Delta for chat UX; full state for debug/audit |
| **MCP loading** | Module-level (startup) | Per-request | Static manifest = startup; dynamic = per-request |
| **Redis vs Postgres** | Redis | Postgres | Sub-ms latency vs. durability + queryability |
| **Client pattern** | `RemoteGraph` | `get_sync_client` | `RemoteGraph` for code reuse; SDK for event-level control |
