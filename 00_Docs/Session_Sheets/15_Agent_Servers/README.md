# Session 15: 🚢 Agent Servers

🎯 Learn to deploy complex agent applications to production endpoints that you can use elsewhere.

📚 **Learning Outcomes**

- Understand the shift from prototyping (notebooks) to production (structured Python packages)
- Learn to package, build, and deploy agents with tool and data access directly to a production API endpoint
- Use LangGraph Studio to visualize, debug, and test deployed agents
- Understand how the deployed API fits into full-stack end-to-end applications

🧰 **New Tools**

Deployment: [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments)

## 📛 Required Tooling & Account Setup

In addition to the tools we've already learned, in this session you'll need:

1. A [LangSmith](https://smith.langchain.com/) account (you should already have one from Session 3)
2. **Docker** installed locally (needed for `langgraph up` production deploys)
3. A **GitHub** repository with your agent code pushed to it (needed for LangSmith cloud deploys)
4. *(Optional)* **LangSmith Plus** subscription (~$40/month) if you want one-click cloud deploys via `langgraph deploy`

## 📜 Recommended Reading

- [You don't know what your agent will do until it's in production](https://blog.langchain.com/you-dont-know-what-your-agent-will-do-until-its-in-production/)
- [LangSmith Deployment components](https://docs.langchain.com/langsmith/components)
- [Agent Server](https://docs.langchain.com/langsmith/agent-server), by LangSmith

# 🗺️ Overview

In Sessions 3 and 4, we learned how to build agents with `create_agent` and LangGraph, customize them with middleware, and observe their behavior through LangSmith tracing. In this session, we take those agents to production.

The challenge, as the LangChain team puts it: agents are easy to prototype and hard to ship to production. Any input or change to an agent can create a host of unknown outcomes. You don't know what your agent will do until it's in the hands of real users — until you're having real conversations in production, you can't fully predict the conversation contexts that will emerge. Building reliable agents requires what they call **agent engineering** — the iterative process of refining non-deterministic LLM systems into reliable experiences that combines product engineering and data science.

This session is about the "ship" in build, ship, and share. We'll take an agent from a local `langgraph dev` environment all the way to a deployed API endpoint that a frontend application can call.

## 🔧 Production Readiness: What We Already Have

Before diving into deployment, it's worth recognizing how much production-grade functionality we get out of the box from the tools we've been using throughout the course:

- **Async requests and parallelization** from LangChain Expression Language
- **Session management** built into LangGraph
- **Automatic retries** from the LangChain/LangGraph stack
- **Scalable vector storage** from Qdrant
- **Optimized parallel execution** with no code changes from prototype to production

That said, production readiness isn't just about tooling — security for LLM applications looks a lot like security for traditional applications. Containerization, authentication, authorization, DevOps best practices — these are mature, classic disciplines that remain essential. The LLM-specific layer (prompt injection defense, content safety, guardrails) sits on top of all the same infrastructure security you'd do for any application.

## 🔄 The Ops Cycle: LLM Ops, Agent Ops, AI Ops

Once agents are in production, you enter an iterative ops cycle: traces and runs feed into evaluations, human or AI annotators label what's important, those datasets inform the next iteration, and the loop continues. This is the data science of AI engineering — continuously evaluating and improving your agentic systems in production.

Key production concerns include low latency, accuracy, cost optimization, scalability, privacy, and data security. For enterprises, keeping customer data safe and secure is paramount — open-source models with private endpoints are one path to achieving this (covered in Session 16).

# 🏷️ What Changed: LangChain's "Final Form"

As of the LangChain v1.0 release (and their rebrand just this week!), **all deployment functionality lives under LangSmith**. The Deployments tab in LangSmith is the single place to deploy, manage, and monitor your agent APIs in production.

| Old Name | New Name |
|---|---|
| LangGraph Platform | LangSmith (Deployments tab) |
| LangServe | Deprecated; replaced by LangSmith deploys |

LangChain and LangGraph are for **building**, LangSmith is for **monitoring and deploying** (the platform), and LangGraph Studio is for **visualization and debugging**.

# 📁 From Notebooks to Production: Project Structure

A key insight from this session: **the move from notebooks to a structured Python package is the move from prototyping to production.**

Up until now, we've been building with notebooks — great for learning, experimenting, and iterating fast. But notebooks don't cut it for production:

- You can't easily serve a notebook — nobody is calling a Jupyter kernel over HTTP to expose an agent as an API
- Notebooks hide execution order — cells run out of order, state leaks between runs, and "restart and run all" is your best option for reproducibility
- No separation of concerns — the Single Responsibility Principle matters in production code
- Version control is harder — notebook JSON files are difficult to diff and manage

Everything we've learned in notebooks still applies — same LangGraph, same LangChain, same concepts — but now organized into proper Python modules.

### Example Project Structure

```
session_15/
├── langgraph.json          # Manifest file — how LangGraph discovers your graphs
├── app/
│   ├── state.py            # Shared state schema (MessagesState)
│   ├── models.py           # Model factory function (defaults to GPT-4.1 Nano)
│   ├── tools.py            # Tool belt (Tavily, Arxiv, local PDF RAG)
│   ├── rag.py              # RAG pipeline as a mini LangGraph (graph-inside-a-tool)
│   └── graphs/
│       ├── simple_agent.py         # Basic agent with tool calling loop
│       └── helpfulness_agent.py    # Agent with helpfulness evaluation node
├── data/
│   └── cat_health_guide.pdf
└── test_served_graph.py    # Test script for the deployed endpoint
```

This separation means you can swap models, add tools, or create new graphs without touching other modules. Every graph imports from the same shared building blocks.

### Key Files Walkthrough

**`state.py`** — Uses LangGraph's pre-built `MessagesState` schema with a messages list and built-in `add_messages` reducer for safe message accumulation across graph steps.

**`models.py`** — A factory function for creating models. Defaults to GPT-4.1 Nano with temperature 0, configurable via environment variables.

**`tools.py`** — Defines the tool belt: Tavily search, Arxiv, and a local PDF retrieval tool. The `get_tool_belt()` function returns all available tools.

**`rag.py`** — The RAG pipeline is itself a mini two-node LangGraph (retrieve → generate). This graph lives inside a tool, inside a bigger agent graph — demonstrating how composable LangGraph is. Uses tiktoken for token-based splitting (GPT-4.0/4.1 share the same encoder family), Qdrant in-memory for vector storage, and `text-embedding-3-small` for embeddings.

**`simple_agent.py`** — A straightforward agent using pre-built LangGraph utilities: `ToolNode` automatically executes tools, and `tools_condition` creates the conditional edge that makes it an agentic loop (route to tools if there's a tool call, otherwise end). The compiled graph is exported for LangGraph to discover.

**`helpfulness_agent.py`** — Extends the simple agent with an evaluate-and-retry pattern. After the agent responds, a **helpfulness node** uses **structured output** to check if the answer is helpful. If not, it loops back to the agent. Key patterns:
- **Structured output** — Forces the LLM to return a specific JSON format (helpful yes/no)
- **Model-as-judge** — GPT-4.1 Mini evaluates the smaller GPT-4.1 Nano's output (more capable model as judge is a common cost-effective pattern)
- **Loop guard** — Checks state length to prevent runaway loops that burn through budget

### The `langgraph.json` Manifest

This is the manifest file — how the LangGraph platform discovers your graphs. Think of **graphs as classes and assistants as instances**.

```json
{
  "graphs": {
    "simple_agent": "./app/graphs/simple_agent.py:graph",
    "agent_with_helpfulness": "./app/graphs/helpfulness_agent.py:graph"
  },
  "assistants": [
    {
      "assistant_id": "simple_agent",
      "name": "Simple Agent",
      "graph_id": "simple_agent",
      "description": "A basic agent with tool access"
    },
    {
      "assistant_id": "agent_with_helpfulness",
      "name": "Agent with Helpfulness",
      "graph_id": "agent_with_helpfulness",
      "description": "Agent with helpfulness evaluation"
    }
  ]
}
```

One graph can have multiple assistants with different configurations.

# 🧪 Local Development → Studio → Deploy

The deployment workflow follows a natural progression:

## Step 1: Develop and Deploy Locally

Run your agent server locally with:

```bash
langgraph dev
```

or, if using `uv`:

```bash
uv run langgraph dev
```

This launches **LangGraph Studio** automatically in your browser (Chromium-based browsers work best). Your agent is now hosted locally at `localhost:2024` and ready to receive requests.

## Step 2: Test with the SDK

Use the LangGraph SDK to test your deployed agent programmatically:

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2024")

# Threadless run with streaming
response = client.runs.create(
    assistant_id="simple_agent",
    input={"messages": [{"role": "human", "content": "How often should I deworm my cat?"}]},
    stream_mode="updates"  # Node-by-node updates, not token-by-token
)
```

Studio and the SDK stream the same events — Studio is for debugging and exploration, the SDK is for production integration. Same data, different interfaces.

## Step 3: Visualize and Debug in LangGraph Studio

LangGraph Studio (accessible at `smith.langchain.com` or via the link in your terminal) provides:

- **Graph topology visualization** — Nodes, edges, and conditional branches rendered exactly as defined in code
- **Step-by-step execution inspection** — Click on any node to see its input/output, including tool calls and results
- **Conversation forking** — Branch conversations to test different paths (similar to ChatGPT's edit feature)
- **Multiple view modes** — Graph view for debugging, chat view for a more traditional interface
- **Multi-assistant switching** — Select between your different agents defined in `langgraph.json`

If you can't visualize it, you can't optimize it. Studio is also a great tool for demoing to stakeholders — the graph visualization maps directly to whiteboard diagrams you might draw with product teams.

## Step 4: Deploy to Production

You have several options for production deployment:

### Self-Hosted (Docker)
```bash
uv run langgraph up
```
This creates a Docker container ready for production. Deploy it to any VPS (virtual private server) and you're live. You'll want to add authentication and authorization on top.

### LangSmith Cloud Deploy
```bash
uv run langgraph deploy
```
This deploys directly to LangSmith's hosted infrastructure. Requires a LangSmith Plus subscription (~$40/month). It handles hosting for you if you don't want to manage Docker and VPS setup yourself.

# ⚠️ Critical: API Backend Only

This is the most important thing to understand about LangSmith deployments:

> **LangSmith deploys your agent as an API backend only.** It does not serve a frontend.

This means LangSmith is **not** a replacement for something like Vercel. What you get is a hosted API endpoint that your agent runs behind. You still need a separate frontend deployment that calls into this API.

A typical production architecture looks like:

- **Frontend** (e.g., deployed on Vercel, Netlify, etc.) → calls into →
- **Agent API** (deployed on LangSmith or self-hosted) → traced and monitored by →
- **LangSmith Observability** (tracing, evals, metrics)

# 🔄 CI/CD with Auto-Deploy

When you enable **auto-update on push** in LangSmith cloud deploys, you get a lightweight CI/CD pipeline out of the box. Every commit to your configured branch triggers a rebuild and redeploy. This means your workflow becomes:

1. Develop and test locally with `langgraph dev` + Studio
2. Push to `main`
3. LangSmith automatically rebuilds and deploys

No manual intervention needed after the initial setup. This is especially powerful when combined with LangSmith's tracing — you can push a change, watch the deployment rebuild, and immediately observe production traces to see if the new version behaves as expected.

# 🧭 Choosing Your Deployment Path

LangSmith one-click deploy is not the only option. Here's how to think about it:

| Approach | Best For | Trade-off |
|---|---|---|
| **LangSmith Cloud Deploy** | Teams already in the LangX ecosystem who want tracing + deploy in one place | Requires LangSmith Plus (~$40/month); API backend only |
| **Self-hosted** (`langgraph up` with Docker) | Teams with existing infrastructure who want full control | You manage scaling, uptime, auth, and monitoring yourself |
| **Custom API** (FastAPI, Flask, etc.) | Teams that need a non-LangGraph deployment or have specific infrastructure requirements | No built-in tracing integration; more setup work |

You can deploy LangGraph locally for free — anytime you use cloud compute, you have to pay somebody. It's up to you to decide whether to pay LangSmith directly or manage your own infrastructure.

# 👀 Looking Ahead

This session kicks off the production phase of the course. Here's what's coming:

| Session | Topic |
|---|---|
| **16** | **LLM Servers** — Deploying open-source models (local + Fireworks AI). Set up a Fireworks AI account before next class! |
| **17** | **MCP Servers** — Building tools that agents can use, agent-to-agent protocol, agent meshes |
| **18** | **Production Optimization** — Cost optimization, caching, guardrails, and other layers that matter once you're in users' hands |

---

Do you have any questions about how to best prepare for Session 15 after reading? Please don't hesitate to provide direct feedback to `greg@aimakerspace.io` or `Dr Greg` on Discord!
