# Session 17: MCP Servers & Agent-to-Agent (A2A) Communication
### Interview-Ready Study Guide — Senior / Staff AI Engineer Level

---

## A. Core Concept Summary

The AI tool ecosystem is converging on two complementary open protocols that together define how agents interact with the world: **MCP (Model Context Protocol)** handles synchronous tool/resource access — think of it as a structured RPC layer between an LLM and external capabilities — while **A2A (Agent-to-Agent)** handles asynchronous delegation of work from one autonomous agent to another. MCP is Anthropic's open standard for exposing tools and resources to language models in a transport-agnostic way; A2A is Google's open standard for inter-agent communication, covering capability advertisement, task lifecycle, and streaming result delivery. The key mental model: **MCP extends what a single agent can do; A2A extends who can do it.** In production architectures, an orchestrator agent uses MCP to call tools (databases, APIs, code runners) and A2A to delegate entire sub-tasks to specialist agents, forming a composable, heterogeneous agent network. Mastering both protocols — their semantics, failure modes, and security surfaces — is the defining systems-design competency for staff-level AI Engineering roles in 2025–2026.

---

## B. Key Terms & Definitions

- **MCP (Model Context Protocol):** Anthropic's open protocol for exposing tools, resources, and prompts to LLM clients via a typed, transport-agnostic RPC interface. The server exposes a capability catalog; the client (LLM host) invokes individual operations synchronously.

- **`@mcp.tool()` decorator:** Registers a Python function as a callable MCP tool. The decorator introspects the function's signature and docstring to auto-generate the JSON Schema that is advertised to the LLM client.

- **`@mcp.resource()` decorator:** Exposes a URI-addressable read-only resource (file, database row, API response). Semantically distinct from tools — resources are *read* by the model for context, whereas tools are *called* for side effects or computation.

- **stdio transport:** MCP transport mode where the client spawns the server as a subprocess and communicates over stdin/stdout. Zero network surface area; ideal for local CLI tools and local agent frameworks like Claude Desktop.

- **SSE transport (Server-Sent Events):** MCP transport mode where the server runs as an HTTP server and streams responses via SSE. Required for remote/cloud-hosted MCP servers, multi-client scenarios, and integration with hosted LLM services.

- **OAuth 2.0 PKCE:** Proof Key for Code Exchange — the OAuth flow used when a public client (CLI tool, local MCP server) cannot safely hold a client secret. Replaces the secret with a cryptographically random verifier/challenge pair to prevent authorization code interception.

- **ngrok tunnel:** A secure reverse proxy that creates a stable public HTTPS URL pointing to a local port. In MCP development, it lets a remote LLM host (e.g., Claude.ai) discover and call a locally running SSE-mode MCP server.

- **AgentCard:** The A2A capability advertisement document (JSON, served at `/.well-known/agent.json`). Describes the agent's name, description, supported task types, input/output schemas, authentication requirements, and endpoint URL. It is the A2A equivalent of an OpenAPI spec.

- **A2A Task lifecycle:** The state machine governing a delegated task: `submitted → working → (input-required →) completed | failed | canceled`. A task ID is assigned on submission; the client can poll or stream SSE events for state transitions and partial results.

- **AgentExecutor:** The server-side A2A interface that receives an incoming `Task`, runs the agent's logic, and yields `TaskStatusUpdateEvent` or `TaskArtifactUpdateEvent` objects back to the caller. The SDK handles SSE serialization and lifecycle bookkeeping.

---

## C. How It Works — Technical Mechanics

### MCP Server Internals

An MCP server is a process that maintains a **capability registry** — a list of tools and resources — and responds to JSON-RPC 2.0 messages from a client. When the server starts (`mcp.run()`), it performs a capability handshake: the client calls `initialize`, receives the server's tool/resource catalog, and then issues `tools/call` or `resources/read` requests.

**stdio flow:**
```
LLM Host spawns process → writes JSON-RPC to stdin → reads JSON-RPC from stdout
```
No network socket; the process lifetime is tied to the host. Latency is sub-millisecond for IPC.

**SSE flow:**
```
Client HTTP POST /messages → Server streams SSE events
Server also exposes GET /sse for the push channel
```
The HTTP+SSE transport splits the channel: the client POSTs requests, and the server pushes responses and notifications back over a persistent SSE connection. This decouples transport lifecycle from request lifecycle and enables true server-push (e.g., progress events for long-running tools).

### OAuth PKCE in MCP

An MCP server often needs to act on behalf of a *human user* to call third-party APIs (Google Drive, GitHub, Slack). The server itself is a public client — it runs locally or in an untrusted environment — so it cannot hold a static client secret. PKCE solves this:

1. Server generates `code_verifier` (random 43–128 char string) and `code_challenge = BASE64URL(SHA256(code_verifier))`.
2. Server redirects user to the authorization endpoint with `code_challenge` and `code_challenge_method=S256`.
3. Authorization server stores the challenge; issues an authorization code.
4. Server exchanges the code + `code_verifier` for tokens. The authorization server verifies the verifier matches the original challenge.
5. Tokens are stored (encrypted at rest) and refreshed via the `refresh_token` grant before expiry.

The refresh loop must be handled proactively: a token expiring mid-conversation breaks tool calls silently unless the MCP server checks expiry before every `tools/call` and refreshes eagerly.

### ngrok Tunnel Mechanics

```
Remote LLM Host ──HTTPS──► ngrok edge ──HTTP──► localhost:PORT ──► MCP SSE server
```

ngrok assigns a stable URL (or a custom subdomain on paid plans). The local MCP server registers its SSE endpoint with that URL in whatever MCP client config is used (e.g., `claude_desktop_config.json`). Security considerations for dev tunnels:
- **Auth tokens are not enough** — anyone who discovers the ngrok URL can call your tools. Add a shared-secret header check in the MCP server.
- **Replay attacks:** SSE over plain ngrok has no built-in request signing. Use short-lived HMAC tokens per session.
- **Tunnel enumeration:** Free ngrok URLs are guessable. Use `--auth` or IP allowlisting in ngrok's dashboard for anything touching real credentials.

### A2A Protocol Deep Dive

A2A uses HTTP+JSON as the wire format with SSE for streaming. The protocol has four primary operations:

| Operation | Endpoint | Description |
|---|---|---|
| `tasks/send` | POST | Submit a new task (non-streaming) |
| `tasks/sendSubscribe` | POST | Submit and subscribe to SSE stream |
| `tasks/get` | GET | Poll task status by ID |
| `tasks/cancel` | POST | Cancel an in-flight task |

The **AgentCard** is served at `GET /.well-known/agent.json` and is the discovery contract. An orchestrator fetches this before delegating, validates that the target agent supports the required task type, and uses the card's auth spec to acquire credentials.

**Task state machine:**

```
submitted
    │
    ▼
 working ──────────────────► input-required
    │                              │
    │◄─────────────────────────────┘
    ▼
completed | failed | canceled
```

`input-required` is the critical state for multi-turn agent interactions: the delegating agent must respond with a follow-up `tasks/send` carrying the additional input, using the same `task_id`.

### MCP vs. A2A: Side-by-Side

| Dimension | MCP | A2A |
|---|---|---|
| **Semantic model** | Synchronous RPC — call a function, get a result | Async task delegation — submit work, get lifecycle events |
| **Who calls whom** | LLM client → tool server | Orchestrator agent → specialist agent |
| **Transport** | stdio (subprocess) or HTTP+SSE | HTTP+SSE (always networked) |
| **Response shape** | Single typed return value | Stream of `TaskStatusUpdateEvent` / `TaskArtifactUpdateEvent` |
| **State management** | Stateless per call | Stateful task with persistent ID |
| **Discovery** | Client config / manifest URL | `/.well-known/agent.json` AgentCard |
| **Python SDK** | `mcp[cli]` (`FastMCP`) | `a2a-sdk` (`AgentExecutor`, `A2AClient`) |
| **Typical topology** | 1 LLM host ↔ N tool servers | 1 orchestrator ↔ N specialist agents |
| **Auth model** | OAuth per-user token in server; or API key | Agent-to-agent JWT / API key; human auth delegated upstream |
| **Use when** | Need a tool, data fetch, or resource read | Need to delegate a multi-step sub-task with uncertain duration |

### Combined Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Orchestrator Agent                  │
│  ┌─────────────┐    ┌───────────────────────────┐    │
│  │  LLM Core   │◄──►│  MCP Client               │    │
│  │  (Claude /  │    │  - code_runner tool        │    │
│  │   GPT-4o)   │    │  - web_search tool         │    │
│  │             │    │  - database resource        │    │
│  │             │◄──►│  A2A Client               │    │
│  │             │    │  - delegate to ResearchAgent│    │
│  │             │    │  - delegate to WriterAgent  │    │
│  └─────────────┘    └───────────────────────────┘    │
└──────────────────────────────────────────────────────┘
          │ MCP stdio/SSE              │ A2A HTTP+SSE
          ▼                            ▼
   ┌─────────────┐             ┌──────────────────┐
   │  Tool       │             │  ResearchAgent   │
   │  Servers    │             │  (A2A server)    │
   │  (MCP)      │             ├──────────────────┤
   └─────────────┘             │  WriterAgent     │
                               │  (A2A server)    │
                               └──────────────────┘
```

---

## D. Common Interview Questions (with Strong Answers)

---

**Q: When would you use A2A instead of just giving your orchestrator more MCP tools? What's the architectural inflection point?**

A: The inflection point is when the sub-task has **uncertain duration, requires its own internal tool usage, or benefits from isolation of failure**. An MCP tool call is synchronous from the orchestrator's perspective — if it takes 90 seconds, the orchestrator's context window is blocked and the LLM is idle. A2A's async task model lets the orchestrator fire-and-forget, continue reasoning about other things, and reconnect to the result stream when needed. More importantly, if the sub-task itself is an agent that needs to call tools, manage memory, and make multi-step decisions, you'd be embedding an entire agent loop inside a single MCP tool call — that's an abstraction violation. A2A enforces that the specialist agent is a first-class citizen with its own identity, auth surface, and lifecycle. The practical rule: if the "tool" has its own system prompt and agent loop, make it an A2A agent.

> **Staff-Level Extension:** A principal interviewer will push on *failure isolation*. In MCP, a hanging tool call can stall the entire orchestrator conversation. In A2A, you can set a task timeout, cancel via `tasks/cancel`, and handle `failed` state gracefully — without losing the orchestrator's context. Design your retry and fallback logic at the A2A boundary, not inside the tool.

---

**Q: Walk me through the security threat model for a multi-agent A2A system. What can go wrong?**

A: Three primary threat surfaces: **agent impersonation, capability escalation, and prompt injection via task artifacts**. Agent impersonation occurs when a malicious actor stands up a server that returns a fake AgentCard claiming to be your trusted ResearchAgent — if orchestrators don't pin expected agent identities (public key or verified URL), they'll delegate sensitive tasks to a rogue agent. Capability escalation happens when an agent advertises capabilities in its AgentCard that exceed what it's authorized to do — the orchestrator must cross-reference the card against a capability policy registry, not blindly trust self-reported capabilities. Prompt injection via artifacts is the sneakiest: a malicious document processed by a sub-agent could embed instructions that the sub-agent faithfully returns as a task artifact, and the orchestrator's LLM might execute those instructions when rendering the result. Mitigations: mutual TLS between agents, signed AgentCards (verify with the issuing authority), output sandboxing (treat all task artifacts as untrusted data until parsed and sanitized), and principle of least privilege in capability scoping.

> **Staff-Level Extension:** Push further on **confused deputy attacks** — an orchestrator agent that has elevated privileges can be tricked by a low-privilege sub-agent into performing privileged operations on its behalf. Solution: propagate a capability token alongside each A2A task that represents the *intersection* of orchestrator and sub-agent permissions, not a union. The A2A spec leaves this to implementers; you need to design it explicitly.

---

**Q: Your MCP server uses OAuth to access Google Drive on behalf of users. How do you handle token refresh without breaking in-flight tool calls?**

A: The naive approach — refresh on 401 — has a race condition: concurrent tool calls can all get a 401 simultaneously, all attempt refresh, and only one succeeds while the rest fail with "invalid_grant" because the refresh token was already rotated. The correct pattern is **optimistic pre-refresh with a distributed lock**: before every `tools/call` execution, check if the access token expires within a buffer window (e.g., 5 minutes). If so, acquire a lock keyed on the user's token identity, re-check after acquiring (double-checked locking), and only then call the token endpoint. For the MCP server specifically, since stdio transport is single-process and single-client, a simple asyncio lock suffices. For SSE transport with multiple concurrent clients, you need Redis-backed locking or a token refresh microservice. Also critical: store the refresh token encrypted at rest (AES-256-GCM with a KMS-managed key), and implement refresh token rotation — if the token service issues a new refresh token on each use, you must atomically update storage or you'll permanently lose access.

> **Staff-Level Extension:** What happens when a user revokes consent mid-session? The MCP server must handle the `invalid_grant` error from the token endpoint as a hard stop, clear all stored tokens for that user, and surface an actionable error (not a generic 500) that tells the LLM client to prompt re-authorization. Design this as a typed error response in the MCP tool's return schema.

---

**Q: Explain the trade-offs between stdio and SSE transport for an MCP server in production. When is each the right choice?**

A: stdio is the right default for **developer tooling, local agents, and single-user deployments** — it has zero network attack surface, the process lifecycle is managed by the parent (no orphan servers), and latency is minimal. The failure mode is tight coupling: if the host crashes, the MCP server dies with it, and vice versa (a blocking tool call in the server blocks the entire host process if not handled async). SSE is required when the MCP server needs to serve **multiple concurrent clients** (e.g., a team-shared tool server), run as a long-lived service (Kubernetes pod), or be reached by a remote LLM host (Claude.ai, OpenAI Assistants). The operational cost is real: you now have a network service to secure (TLS, auth), monitor (health checks, connection pools), and scale (SSE connections are long-lived and stateful, which fights against horizontal scaling — you need sticky sessions or a shared pub/sub layer like Redis Streams to fan out events). For internal platform teams building shared MCP infrastructure, SSE with an mTLS sidecar and connection pooling is the production pattern.

---

**Q: How does the A2A AgentCard's capability advertisement interact with orchestrator routing decisions? What are the failure modes of a naive implementation?**

A: A naive orchestrator fetches the AgentCard once at startup, caches it indefinitely, and routes tasks based on the static capability list. This breaks in three ways: (1) **capability drift** — the specialist agent deploys a new version with different input schemas, and the orchestrator sends tasks with the old schema, causing silent failures or malformed requests; (2) **stale availability** — the card says the agent is available but the underlying service is down; (3) **version skew** — during a rolling deploy, some instances serve old cards and some serve new, causing non-deterministic routing. A robust implementation treats AgentCards as **short-TTL cached documents** (5–15 minutes), validates the card schema version on every task submission, and treats a 404 or schema mismatch on the card endpoint as a circuit-breaker signal. More importantly, the orchestrator should not hard-code routing logic against specific agents — instead, maintain a capability registry where cards are indexed by capability type, enabling dynamic agent selection and fallback to alternate agents when a primary is unhealthy.

---

**Q: How would you design the observability layer for a production orchestrator that uses both MCP tools and A2A sub-agents?**

A: The core challenge is **distributed trace correlation across protocol boundaries**. When an orchestrator calls an MCP tool, you need a trace context (W3C TraceContext headers for SSE, or a trace ID embedded in the JSON-RPC request metadata for stdio) that flows into the tool server so all spans share the same trace. For A2A, the task ID is a natural correlation key, but it's not a trace ID — you need to propagate an OpenTelemetry trace context in the A2A task's metadata field. Instrumenting this properly means: (1) the orchestrator creates a root span per user request; (2) each MCP `tools/call` creates a child span; (3) each A2A `tasks/send` creates a child span with the task ID as a span attribute, and the receiving AgentExecutor creates its own root span linked to the orchestrator's span via `LINK` relationship (not parent-child, since A2A is async). Token consumption, latency, and error rates per tool and per sub-agent should be emitted as OTEL metrics. Alert on: p99 A2A task completion time, MCP tool error rate, and A2A `input-required` rate (a high rate signals the orchestrator is not providing sufficient context at submission time).

---

## E. Gotchas, Trade-offs & Best Practices

- **MCP tool docstrings are your API contract, not just documentation.** The `@mcp.tool()` decorator uses the function docstring and type annotations to generate the JSON Schema that the LLM sees. A vague docstring ("processes data") leads to wrong tool selection; an overly broad type annotation (`str` instead of `Literal["asc", "desc"]`) removes constraints the LLM needs to produce valid calls. Treat tool signatures as a typed API — use `Annotated` types with `Field(description=...)` from Pydantic for fine-grained schema control.

- **A2A's `input-required` state is frequently unimplemented and frequently needed.** Most tutorial implementations only handle the happy path (`submitted → working → completed`). In production, ambiguous delegated tasks commonly require clarification — the ResearchAgent doesn't know which of three interpretations to pursue. Without `input-required`, sub-agents either guess (hallucinate) or fail. Implement the full state machine; design the orchestrator to handle the clarification loop explicitly rather than treating it as an error.

- **ngrok tunnels in development create OAuth redirect URI drift.** OAuth providers require pre-registered redirect URIs. Every time ngrok assigns a new URL (on free plans), your OAuth redirect URI changes and authorization fails. Fix: use ngrok's reserved domains (paid), or use a fixed local redirect URI (`http://localhost:PORT/callback`) and handle the OAuth redirect locally — only use ngrok for the MCP SSE endpoint itself, not for the OAuth callback.

- **stdio MCP servers block on synchronous tool calls — always use `async def`.** If a tool makes a network call (database, external API) using synchronous `requests` instead of `aiohttp`/`httpx`, it blocks the entire event loop and the MCP server becomes unresponsive to the host during that call. Every tool handler should be `async def`, and all I/O inside must use async libraries. For CPU-bound tools, use `asyncio.run_in_executor` to offload to a thread pool.

- **Capability scoping in A2A is opt-in and operator responsibility.** The A2A spec defines the AgentCard's `capabilities` field but does not enforce what a requesting agent can actually invoke. Without a policy enforcement layer, any agent that can reach the A2A endpoint can submit any task type. In production: implement task-type-level authorization (JWT claims or API key scopes map to allowed task types), log every `tasks/send` with the caller identity, and treat unexpected task types as a security signal, not just a validation error.

---

## F. Code & Architecture Patterns

### (1) MCP Server with `@mcp.tool()` and `@mcp.resource()`

```python
# server.py
from mcp.server.fastmcp import FastMCP
from mcp import Resource
from pydantic import BaseModel, Field
from typing import Annotated
import httpx

mcp = FastMCP(
    name="product-catalog",
    version="1.0.0",
)


class SearchParams(BaseModel):
    query: str = Field(description="Full-text search query for product catalog")
    limit: Annotated[int, Field(ge=1, le=50, description="Max results to return")] = 10
    category: Annotated[
        str | None,
        Field(description="Optional category filter: 'electronics' | 'apparel' | 'home'"),
    ] = None


@mcp.tool()
async def search_products(params: SearchParams) -> list[dict]:
    """
    Search the product catalog and return matching items with price and inventory.
    Returns an empty list if no matches are found. Raises ToolError on API failure.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            "https://api.internal/products/search",
            params=params.model_dump(exclude_none=True),
            headers={"Authorization": f"Bearer {_get_service_token()}"},
        )
        response.raise_for_status()
        return response.json()["results"]


@mcp.resource("catalog://categories")
async def list_categories() -> str:
    """Returns the current product category taxonomy as a JSON string."""
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.internal/products/categories")
        r.raise_for_status()
        return r.text  # JSON string; MCP resources return str or bytes


def _get_service_token() -> str:
    # In production: load from encrypted token store, refresh if near expiry
    import os
    return os.environ["PRODUCT_API_TOKEN"]


if __name__ == "__main__":
    import sys
    transport = "sse" if "--sse" in sys.argv else "stdio"
    mcp.run(transport=transport)
    # stdio: python server.py
    # SSE:   python server.py --sse  (listens on http://localhost:8000)
```

---

### (2) A2A `AgentExecutor` — Streaming Task Skeleton

```python
# a2a_agent.py
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    Artifact,
    TextPart,
    Message,
    Role,
)
from a2a.server.apps import A2AStarlette
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
import uvicorn


class ResearchAgentExecutor(AgentExecutor):
    """
    Specialist agent that performs multi-step research for a given query.
    Streams intermediate status updates and yields a final artifact.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task: Task = context.current_task
        user_message = self._extract_text(task)

        # Signal that work has begun
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task.id,
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
        )

        try:
            # Step 1: gather sources (would use MCP tools internally)
            sources = await self._gather_sources(user_message)

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task.id,
                    status=TaskStatus(
                        state=TaskState.working,
                        message=Message(
                            role=Role.agent,
                            parts=[TextPart(text=f"Gathered {len(sources)} sources, synthesizing...")],
                        ),
                    ),
                    final=False,
                )
            )

            # Step 2: synthesize result
            synthesis = await self._synthesize(user_message, sources)

            # Emit artifact (the actual deliverable)
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=task.id,
                    artifact=Artifact(
                        name="research_report",
                        parts=[TextPart(text=synthesis)],
                        index=0,
                        last_chunk=True,
                    ),
                )
            )

            # Terminal state
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task.id,
                    status=TaskStatus(state=TaskState.completed),
                    final=True,
                )
            )

        except Exception as exc:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task.id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=Message(
                            role=Role.agent,
                            parts=[TextPart(text=f"Research failed: {exc}")],
                        ),
                    ),
                    final=True,
                )
            )
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.current_task.id,
                status=TaskStatus(state=TaskState.canceled),
                final=True,
            )
        )

    def _extract_text(self, task: Task) -> str:
        for part in task.history[0].parts:
            if isinstance(part, TextPart):
                return part.text
        raise ValueError("No text part found in task message")

    async def _gather_sources(self, query: str) -> list[str]:
        # In production: call MCP web_search tool here
        return [f"Source about: {query}"]

    async def _synthesize(self, query: str, sources: list[str]) -> str:
        # In production: call LLM with gathered sources
        return f"Research synthesis for '{query}': {sources}"


# A2A client usage (orchestrator side)
async def delegate_research(orchestrator_mcp_server_url: str, query: str) -> str:
    from a2a.client import A2AClient
    from a2a.types import SendTaskParams, Message, TextPart, Role
    import httpx

    async with httpx.AsyncClient() as http_client:
        client = await A2AClient.get_client_from_agent_card_url(
            http_client,
            "http://localhost:9000",  # ResearchAgent A2A server base URL
        )

        task_response = await client.send_task(
            SendTaskParams(
                message=Message(
                    role=Role.user,
                    parts=[TextPart(text=query)],
                )
            )
        )

        task_id = task_response.result.id

        # Stream results until terminal state
        async with client.send_task_streaming(
            SendTaskParams(
                id=task_id,
                message=Message(role=Role.user, parts=[TextPart(text=query)]),
            )
        ) as event_stream:
            async for event in event_stream:
                if isinstance(event.result, TaskArtifactUpdateEvent):
                    for part in event.result.artifact.parts:
                        if isinstance(part, TextPart):
                            return part.text

        return ""


if __name__ == "__main__":
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    agent_card = AgentCard(
        name="ResearchAgent",
        description="Multi-step research agent with web access",
        url="http://localhost:9000/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="deep-research",
                name="Deep Research",
                description="Researches a topic across multiple sources and synthesizes a report",
                inputModes=["text"],
                outputModes=["text"],
            )
        ],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=ResearchAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarlette(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(app, host="0.0.0.0", port=9000)
```

---

*Study guide generated for AIE9 Session 17 — MCP Servers & Agent-to-Agent (A2A) Communication.*
*Audience: Senior / Staff AI Engineer interview preparation.*
