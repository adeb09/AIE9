# Interview Study Guide: The Agent Loop (LangChain / LCEL)
> Senior / Staff AI Engineering — Deep-Dive Reference

---

## Table of Contents

1. [LangChain Expression Language (LCEL)](#1-lcel)
2. [create_react_agent() — Abstractions & Trade-offs](#2-create_react_agent)
3. [The ReAct Loop in Detail](#3-the-react-loop)
4. [Tools: Decorators, Schemas, and bind_tools()](#4-tools)
5. [Qdrant as a Production Vector Store](#5-qdrant)
6. [Middleware Patterns in the Agent Loop](#6-middleware)
7. [Human-in-the-Loop (HITL)](#7-human-in-the-loop)
8. [Error Handling in Agent Loops](#8-error-handling)
9. [Rapid-Fire Interview Q&A](#9-rapid-fire-qa)

---

## 1. LCEL

### The `Runnable` Interface

Every component in LangChain (prompts, models, retrievers, parsers, custom functions) implements a single interface:

```python
class Runnable(ABC):
    def invoke(self, input, config=None)               # single call
    def batch(self, inputs, config=None)               # parallel calls
    def stream(self, input, config=None)               # token-by-token
    async def ainvoke(self, input, config=None)        # async single
    async def astream(self, input, config=None)        # async stream
```

The `|` operator calls `__or__`, which returns a `RunnableSequence`. This is the entire design of LCEL: **any two Runnables can be composed** because they share the same interface contract.

```python
chain = prompt | model | output_parser
# Equivalent to:
chain = RunnableSequence(first=prompt, middle=[model], last=output_parser)
```

### Key Runnable Primitives

#### `RunnablePassthrough`

Passes input through unchanged. Its primary use is **routing values around a chain** so they remain available downstream.

```python
from langchain_core.runnables import RunnablePassthrough

# Classic RAG pattern — preserve original question alongside retrieved docs
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

**Interview insight:** The dict literal `{"context": retriever, "question": RunnablePassthrough()}` is sugar for `RunnableParallel`. The question passes through so the prompt template has both `{context}` and `{question}` in scope.

#### `RunnableLambda`

Wraps any Python callable as a Runnable. Used for lightweight transformations inline.

```python
from langchain_core.runnables import RunnableLambda

format_docs = RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)
```

Use `RunnableLambda` when:
- You need a quick transformation between two components
- You want to keep the chain composable (e.g., `.batch()` still works)
- You want introspection/tracing in LangSmith to show the step

Avoid it when: the logic is complex enough to warrant a proper class with error handling.

#### `RunnableParallel`

Runs multiple Runnables on the **same input** concurrently and returns a dict.

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summarize_chain,
    sentiment=sentiment_chain,
    keywords=keyword_chain,
)
result = parallel.invoke("Some long document text...")
# result = {"summary": ..., "sentiment": ..., "keywords": ...}
```

Under the hood, LangChain uses `ThreadPoolExecutor` for sync batch execution. For async, it uses `asyncio.gather`.

### Design Philosophy: When LCEL Adds Value vs. Friction

| Adds value | Adds friction |
|---|---|
| Simple linear pipelines | Complex conditional branching |
| Uniform streaming / async support for free | Multi-step state management |
| Built-in LangSmith tracing per step | Debugging operator-chained lambdas |
| Easy `.batch()` parallelism | When you need explicit error recovery per step |
| Declarative: you can `.get_graph().print_ascii()` | Teams unfamiliar with functional composition |

**Senior-level take:** LCEL is excellent for stateless, linear transformations. The moment your chain needs to **loop**, **branch on state**, or **accumulate messages**, you should reach for LangGraph instead. LangGraph is built on the same Runnable abstraction but adds explicit state, edges, and conditional routing that LCEL cannot express cleanly.

---

## 2. `create_react_agent()`

### What it Abstracts

`create_react_agent()` (from `langchain.agents`) is a convenience factory that wires together:

1. A `ChatPromptTemplate` with the ReAct system prompt
2. Tool binding via `model.bind_tools(tools)`
3. A LangGraph `StateGraph` with `agent` and `tools` nodes
4. Conditional edges: if the last message has `tool_calls`, route to the tools node; otherwise, end

The returned object is a `CompiledStateGraph` — itself a `Runnable`.

```python
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[search, calculator],
    state_modifier="You are a helpful assistant."
)
# Returns: CompiledStateGraph (a Runnable)
result = agent.invoke({"messages": [HumanMessage(content="...")]})
```

### The Underlying Prompt Template

Without the abstraction, the system prompt that drives ReAct reasoning looks like:

```
You are designed to help with a variety of tasks, from answering questions \
to providing summaries to other types of analyses.

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.

{tools}

## Output Format
To answer the question, please use the following format:

Thought: I need to use [tool] because ...
Action: tool_name
Action Input: {"param": "value"}
Observation: <tool output appended here>
... (this Thought/Action/Observation loop repeats N times)
Thought: I now have enough information to answer.
Final Answer: [answer here]
```

**Critical detail:** Modern function-calling models (GPT-4o, Claude 3.5) do **not** literally produce `Thought:` / `Action:` text. The "thinking" is implicit in the model's reasoning, and tool invocations are structured JSON in the `tool_calls` field of the `AIMessage`. The ReAct *pattern* (reason → act → observe → repeat) still applies, but the wire format is the OpenAI tool-call API, not free-text parsing.

### What You Lose with the Abstraction vs. Building from Scratch

| `create_react_agent()` | Raw LangGraph |
|---|---|
| Opaque state schema — hard to extend | Full control over `TypedDict` state |
| Fixed routing logic | Custom conditional edges |
| Hard to add pre/post-processing per node | Explicit node functions with full Python |
| Limited streaming granularity | Stream individual node updates |
| No built-in persistence/checkpointing | Plug in any `Checkpointer` (SQLite, Redis, Postgres) |

**When to build from scratch:** Any production agent that needs conversation history across sessions, complex branching (different tools for different user roles), or tight latency budgets where you need to short-circuit the loop early.

---

## 3. The ReAct Loop in Detail

### The Full Cycle

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   AGENT LOOP                        │
│                                                     │
│  ┌─────────────┐   tool_calls?   ┌───────────────┐ │
│  │  LLM Call   │ ──── yes ──────▶│  Tool Executor│ │
│  │ (+ history) │                 │  (parallel or │ │
│  └─────────────┘                 │   sequential) │ │
│        ▲                         └───────┬───────┘ │
│        │                                 │         │
│        └──────── ToolMessages ◀──────────┘         │
│                  appended to                       │
│                  message history                   │
│                                                     │
│  ┌─────────────┐                                    │
│  │  LLM Call   │ ──── no tool_calls ──▶  END        │
│  │  (+ history)│    (Final Answer)                  │
│  └─────────────┘                                    │
└─────────────────────────────────────────────────────┘
```

### How Tool Results Are Injected Back

After every tool execution, the result is appended to the message list as a `ToolMessage`:

```python
from langchain_core.messages import ToolMessage

ToolMessage(
    content="The current time is 14:32:07",
    tool_call_id="call_abc123",   # must match the id from AIMessage.tool_calls
    name="get_current_time"
)
```

The next LLM call receives the entire growing message list:
```
HumanMessage("What time is it and divide 100 by the hour?")
AIMessage(tool_calls=[{id: "c1", name: "get_current_time", args: {}}])
ToolMessage(content="14:32:07", tool_call_id="c1")
AIMessage(tool_calls=[{id: "c2", name: "calculate", args: {expression: "100/14"}}])
ToolMessage(content="7.142...", tool_call_id="c2")
AIMessage(content="The answer is 7.14")    ← Final Answer
```

**Key insight for interviews:** The model never "sees" its own tool calls as final facts — it sees the `ToolMessage` responses. The growing context window is the agent's working memory. This is why token usage grows linearly with loop iterations, and why context window size is a hard ceiling on loop depth.

### Multi-Step Reasoning Across Steps

The model doesn't have persistent memory between steps — it only has the **message list**. The apparent "reasoning across steps" is just the model attending to prior `ToolMessage` outputs when generating the next decision. This makes LLM-based agents fundamentally different from symbolic AI planners: all state is in the prompt context.

---

## 4. Tools

### `@tool` Decorator

The simplest way to create a tool. LangChain parses the **docstring** and **type annotations** to generate the JSON Schema automatically.

```python
from langchain_core.tools import tool

@tool
def search_web(query: str, max_results: int = 5) -> list[str]:
    """Search the web for recent information.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default 5)
    
    Returns:
        List of result snippets
    """
    return web_search_api(query, max_results)
```

The generated schema:
```json
{
  "name": "search_web",
  "description": "Search the web for recent information.\n\nArgs:\n    query: ...",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "max_results": {"type": "integer", "default": 5}
    },
    "required": ["query"]
  }
}
```

**Interview trap:** The docstring is not optional flavor text — it IS the tool's interface to the model. A poor docstring (vague, no parameter descriptions) directly degrades tool selection accuracy. This is equivalent to writing a bad API spec.

### `StructuredTool`

More explicit than `@tool`. Use when you need Pydantic validation, custom error messages, or to separate the schema from the implementation.

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Result count (1-20)")

def _search(query: str, max_results: int = 5) -> list[str]:
    return web_search_api(query, max_results)

search_tool = StructuredTool.from_function(
    func=_search,
    name="search_web",
    description="Search the web for recent information.",
    args_schema=SearchInput,
    return_direct=False   # if True, tool output bypasses further LLM processing
)
```

`return_direct=True` is a footgun: the tool's raw output is returned as the final answer, bypassing the model's ability to synthesize or reason about the result.

### Tool Schemas: JSON Schema vs. Pydantic

Under the hood, LangChain converts Pydantic models to JSON Schema for the OpenAI API:

```python
# Pydantic v2 model → JSON Schema
SearchInput.model_json_schema()
# {
#   "properties": {
#     "query": {"description": "...", "title": "Query", "type": "string"},
#     "max_results": {"default": 5, "ge": 1, "le": 20, ...}
#   },
#   "required": ["query"],
#   "title": "SearchInput",
#   "type": "object"
# }
```

Pydantic gives you free validation, coercion, and descriptive error messages when the model generates malformed tool arguments. Always use Pydantic for complex tool schemas.

### `bind_tools()`

This is how tools are attached to a chat model so it knows they exist at inference time:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
model_with_tools = model.bind_tools([search_tool, calculator_tool])

# Now the model's API call includes the tools in the request payload
response = model_with_tools.invoke([HumanMessage("search for the latest AI news")])
# response.tool_calls → [{"name": "search_web", "args": {"query": "latest AI news"}, "id": "..."}]
```

`bind_tools()` returns a new Runnable (does not mutate the original model). Internally it calls `model.bind(tools=[...])` which sets up the `tools` parameter in every API request.

**Parallel tool calls:** OpenAI and Anthropic models can emit multiple `tool_calls` in a single `AIMessage`. The LangGraph tools node executes them in parallel by default and returns multiple `ToolMessage` objects, one per call.

---

## 5. Qdrant as a Production Vector Store

### Collection Management

A Qdrant **collection** is analogous to a table in a relational database — it holds vectors and their associated payloads.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,              # Must match your embedding model's output dimension
        distance=Distance.COSINE
    )
)
```

**Distance metrics — when to use each:**

| Metric | Formula | Use when |
|---|---|---|
| `COSINE` | 1 - cosine similarity | Text embeddings (direction matters, magnitude doesn't) |
| `DOT` | dot product | When vectors are already normalized; faster than COSINE |
| `EUCLID` | L2 distance | Image/audio embeddings where magnitude carries meaning |

For OpenAI `text-embedding-3-*` models, **COSINE** is the default and correct choice. Normalized embeddings make DOT and COSINE equivalent, but COSINE is safer when normalization isn't guaranteed.

### Payload Filtering

Qdrant's key production advantage: **pre-filtering** before ANN search, which is far more efficient than post-filtering at the application layer.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Sparse filtering: exact match on metadata
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="source", match=MatchValue(value="arxiv")),
            FieldCondition(key="year", range=Range(gte=2023))
        ]
    ),
    limit=10
)
```

**Sparse vs. Dense filtering:**

- **Dense filtering** = many documents match the filter (low selectivity). Qdrant uses its HNSW graph but applies the filter during traversal.
- **Sparse filtering** = very few documents match (high selectivity). Qdrant can switch to a brute-force scan of the matching subset, which is faster than graph traversal when the filtered set is tiny.

Qdrant handles this automatically via **payload indexes**. Always create an index on fields you filter frequently:

```python
client.create_payload_index(
    collection_name="documents",
    field_name="source",
    field_schema="keyword"   # "keyword", "integer", "float", "geo", "text"
)
```

### LangChain Integration

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="documents",
    url="http://localhost:6333",
)

retriever = vectorstore.as_retriever(
    search_type="mmr",        # "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

**MMR (Maximal Marginal Relevance):** Balances relevance and diversity. It fetches `fetch_k` candidates by similarity, then greedily selects `k` documents that maximize relevance while minimizing redundancy with already-selected docs. Use MMR when you have highly repetitive corpora.

---

## 6. Middleware Patterns in the Agent Loop

### The Core Pattern

Middleware wraps a Runnable to add cross-cutting behavior without modifying the core logic. LangChain's `RunnableLambda` and the `@chain` decorator are the primary mechanisms.

```
Input → [Middleware 1] → [Middleware 2] → Core Runnable → [Middleware 2] → [Middleware 1] → Output
                    (like Python's @decorator stack)
```

### Logging Middleware

```python
from langchain_core.runnables import RunnableLambda
import logging

logger = logging.getLogger(__name__)

def with_logging(runnable):
    def _invoke(input, config=None):
        logger.info("Input: %s", input)
        output = runnable.invoke(input, config)
        logger.info("Output: %s", output)
        return output
    return RunnableLambda(_invoke)

logged_agent = with_logging(agent)
```

For production, prefer **LangSmith** callbacks, which capture the full trace tree including nested tool calls, token counts, and latency per step — all without instrumenting your own code.

### Rate Limiting Middleware

```python
import time
from threading import Semaphore

def with_rate_limit(runnable, calls_per_second: float):
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]   # mutable closure

    def _invoke(input, config=None):
        elapsed = time.monotonic() - last_called[0]
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        last_called[0] = time.monotonic()
        return runnable.invoke(input, config)

    return RunnableLambda(_invoke)
```

For async/high-concurrency scenarios, use `asyncio.Semaphore` or a token-bucket algorithm backed by Redis.

### Retry Middleware

```python
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.runnables import RunnableLambda

def with_retry(runnable, max_attempts=3, base_wait=1):
    @retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_wait, min=1, max=30)
    )
    def _invoke(input, config=None):
        return runnable.invoke(input, config)
    return RunnableLambda(_invoke)
```

Note: LangChain's `.with_retry()` method on any Runnable provides this natively:

```python
chain_with_retry = chain.with_retry(
    retry_if_exception_type=(RateLimitError, APIConnectionError),
    stop_after_attempt=3,
    wait_exponential_jitter=True
)
```

### Input/Output Transformation

The most common use case: adapting the interface between components that weren't designed together.

```python
# Normalize input keys before hitting the LLM
normalize = RunnableLambda(lambda x: {
    "question": x.get("query") or x.get("q") or x["question"],
    "history": x.get("chat_history", [])
})

# Sanitize output for downstream consumers
sanitize = RunnableLambda(lambda x: {
    "answer": x.content.strip(),
    "sources": extract_citations(x.content)
})

pipeline = normalize | model | sanitize
```

---

## 7. Human-in-the-Loop (HITL)

### Why HITL Matters

Agents with tool-calling authority (write access to databases, file systems, external APIs) must have a mechanism for human approval before executing irreversible actions. HITL is not a UX feature — it's a **safety architecture concern**.

### Interrupt Strategies

#### Strategy 1: Pre-tool interrupt (approve before execution)

The agent proposes a tool call; execution is paused until a human approves or rejects.

```python
# LangGraph approach with breakpoints
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

# Interrupt BEFORE the tools node executes
compiled = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]
)
```

#### Strategy 2: Post-tool interrupt (review after execution)

Tools run, but the agent is paused before presenting results to the user — useful for content moderation.

```python
compiled = graph.compile(
    checkpointer=checkpointer,
    interrupt_after=["tools"]
)
```

### `input()` Patterns in Notebooks vs. Production

**Notebook pattern (synchronous, acceptable for prototyping):**

```python
for chunk in agent.stream({"messages": [HumanMessage(query)]}, config):
    if "__interrupt__" in chunk:
        pending = chunk["__interrupt__"][0]
        print(f"Agent wants to call: {pending.value}")
        approval = input("Approve? (y/n): ")
        if approval.lower() == "y":
            result = agent.invoke(Command(resume=True), config)
        else:
            result = agent.invoke(Command(resume="User rejected this action."), config)
```

**Production HITL via async queues:**

```python
import asyncio

approval_queues: dict[str, asyncio.Queue] = {}

async def run_agent_with_hitl(session_id: str, query: str):
    config = {"configurable": {"thread_id": session_id}}
    approval_queues[session_id] = asyncio.Queue()

    async for chunk in agent.astream({"messages": [HumanMessage(query)]}, config):
        if "__interrupt__" in chunk:
            pending = chunk["__interrupt__"][0]
            # Send pending action to frontend via WebSocket / SSE
            await notify_frontend(session_id, pending.value)
            # Block here until the human responds
            decision = await approval_queues[session_id].get()
            result = await agent.ainvoke(Command(resume=decision), config)
            return result

# Separate HTTP endpoint to receive human decision
async def approve_action(session_id: str, approved: bool):
    decision = True if approved else "User rejected."
    await approval_queues[session_id].put(decision)
```

**Key production considerations:**
- Use a **persistent checkpointer** (Postgres, Redis) so the paused agent state survives process restarts
- Set a **timeout** on the approval queue to prevent dangling sessions
- Store the pending action in the queue/DB for audit trails

---

## 8. Error Handling in Agent Loops

### Max Iterations

Without a bound, a buggy agent can loop forever, burning tokens. `create_react_agent` and LangGraph both support recursion limits:

```python
# LangGraph: set globally
compiled = graph.compile(checkpointer=checkpointer)
result = compiled.invoke(
    {"messages": [HumanMessage(query)]},
    config={"recursion_limit": 10}  # Default is 25
)
# If limit exceeded: raises GraphRecursionError
```

Always catch `GraphRecursionError` at the application layer and return a graceful degradation response.

### Tool Execution Errors

The recommended pattern is **catch-and-return**: have the tool return an error string rather than raising, so the agent can reason about the failure and try an alternative approach.

```python
@tool
def fetch_stock_price(ticker: str) -> str:
    """Fetch the current stock price for a given ticker symbol."""
    try:
        price = stock_api.get_price(ticker)
        return f"{ticker}: ${price:.2f}"
    except TickerNotFoundError:
        return f"Error: Ticker '{ticker}' not found. Please verify the symbol."
    except APITimeoutError:
        return f"Error: Stock API timed out. Try again or use a different source."
    except Exception as e:
        return f"Unexpected error fetching {ticker}: {str(e)}"
```

**Why this is better than raising:** If the tool raises, the agent loop crashes unless you have a global try/catch. If the tool returns an error string as `ToolMessage.content`, the model sees it and can choose to retry, ask for clarification, or escalate — which is exactly what a good agent should do.

### Malformed Tool Calls

Even strong models occasionally generate invalid JSON arguments. Handle at the tool execution layer:

```python
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
import json

def safe_tool_invoke(tool: BaseTool, tool_call: dict) -> ToolMessage:
    try:
        result = tool.invoke(tool_call["args"])
        return ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
    except ValidationError as e:
        return ToolMessage(
            content=f"Invalid arguments: {e.errors()}. Please retry with correct parameters.",
            tool_call_id=tool_call["id"],
            status="error"
        )
    except Exception as e:
        return ToolMessage(
            content=f"Tool execution failed: {str(e)}",
            tool_call_id=tool_call["id"],
            status="error"
        )
```

### Graceful Degradation Strategies

| Failure Mode | Strategy |
|---|---|
| Max iterations reached | Return partial result + "could not fully answer" message |
| Critical tool unavailable | Fall back to LLM knowledge, clearly state uncertainty |
| All retries exhausted | Return cached/stale result with staleness warning |
| Context window exceeded | Summarize intermediate tool results, compress history |
| Model refuses (content policy) | Route to human support or conservative fallback prompt |

**Context window overflow** is a particularly insidious failure: the agent silently degrades as old tool results are truncated from context. Mitigate with:

```python
# Summarize tool history when message count exceeds threshold
from langchain_core.messages import SystemMessage

def trim_message_history(messages: list, max_tokens: int = 3000) -> list:
    from langchain_core.messages import trim_messages
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=ChatOpenAI(model="gpt-4o"),
        include_system=True,
        allow_partial=False,
        start_on="human"
    )
```

---

## 9. Rapid-Fire Interview Q&A

**Q: What is the difference between `RunnablePassthrough` and `RunnableLambda(lambda x: x)`?**

They're functionally equivalent for a single synchronous call. `RunnablePassthrough` is preferred because it's semantically explicit, is recognized by LangChain's graph visualization tools, and has a proper `.assign()` method for adding fields without replacing the whole dict.

---

**Q: Why does `create_react_agent` return a `CompiledStateGraph` instead of a plain Runnable chain?**

Because agent loops are not DAGs — they're cycles. LCEL's `|` pipe operator builds directed acyclic graphs (sequences). A loop requires a graph with a back-edge from the tools node back to the model node, which only LangGraph can express. `create_react_agent` is a thin wrapper over LangGraph's `StateGraph`.

---

**Q: What happens if two tools are called in parallel and one fails?**

By default, LangGraph's tools node collects all results (or errors) before passing them back to the model. A failing tool returns a `ToolMessage` with `status="error"`. The model sees both results (one success, one error) and can reason about the partial failure. No exception is propagated unless you explicitly re-raise in the tool or the tool node's error handler.

---

**Q: How would you add per-user rate limiting to an agent in production?**

Maintain a token bucket per user ID in Redis. In a custom tools node or pre-model middleware, check and decrement the bucket before invoking the model. Use `asyncio.Semaphore` for per-process limits, Redis + Lua scripts for distributed limits. Pass the user ID via `config["configurable"]["user_id"]` using LangGraph's config pattern.

---

**Q: What is the difference between `bind_tools()` and `with_structured_output()`?**

- `bind_tools()` adds tools to the model's request payload. The model can call 0 or more tools, and may also return plain text. Used for agents where tool use is optional.
- `with_structured_output()` forces the model to always return a specific schema via function-calling or JSON mode. Used for extraction / classification where you need guaranteed structured output, not agent behavior.

---

**Q: When would you use Qdrant's `DOT` distance instead of `COSINE`?**

When your embedding vectors are L2-normalized (unit vectors), DOT and COSINE produce identical rankings, but DOT is slightly faster (no normalization step). OpenAI's embeddings are normalized, so DOT is safe. Use COSINE as the default safe choice unless you've benchmarked and verified normalization.

---

**Q: What is the risk of `return_direct=True` on a tool?**

The tool's raw output bypasses the LLM's final synthesis step and is presented directly to the user as the agent's answer. This is dangerous for tools that return machine-readable formats (JSON, code), error messages, or unformatted data. It also prevents the model from checking whether the tool actually answered the user's original question.

---

**Q: How do you prevent an agent from entering an infinite retry loop on a failing tool?**

Three layers of defense:
1. **Tool level:** Return an error string after N internal retries; never let the tool block indefinitely.
2. **Agent level:** Set `recursion_limit` on the LangGraph compilation.
3. **Application level:** Wrap `compiled.invoke()` in a timeout (`asyncio.wait_for`) and catch `GraphRecursionError`.

---

**Q: In a production HITL system, how do you handle an agent that was paused waiting for approval but the user never responds?**

Store the checkpoint in a persistent store (Postgres). Run a background job that scans for checkpoints older than the timeout threshold and either:
- Sends a reminder notification to the user
- Auto-rejects the action and resumes the agent with a timeout message
- Marks the session as abandoned and notifies the requesting system

This requires that the `Checkpointer` is persistent (not `MemorySaver`) and that your queue timeout propagates a cancellation back into the agent's resume path.

---

**Q: How does LangChain's `with_retry()` differ from wrapping a Runnable in a `tenacity` retry decorator?**

`with_retry()` is Runnable-aware: it retries the entire Runnable's `invoke/batch/stream` interface, preserves streaming behavior, and integrates with LangSmith's trace tree (each retry appears as a child span). A raw `tenacity` decorator wraps the Python function, which breaks streaming (you can't retry a generator mid-stream) and creates opaque traces.

---

*Generated for AIE9 Bootcamp — Session 03: The Agent Loop*
