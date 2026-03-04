# Session 15: 🚢 Agent Servers

🎯 Learn to deploy complex agent applications to production endpoints that you can use elsewhere.

📚 **Learning Outcomes**

- Learn to package, build, and deploy agents with tool and data access directly to a production API endpoint
- Understand how to use the API in full-stack end-to-end applications

🧰 **New Tools**

Deployment: [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments)

## 📛 Required Tooling & Account Setup

In addition to the tools we've already learned, in this session you'll need:

1. A [LangSmith](https://smith.langchain.com/) account (you should already have one from Session 3)
2. **LangSmith Plus** subscription (~$40/month) for one-click deploys
3. A **GitHub** repository with your agent code pushed to it

## 📜 Recommended Reading

- [You don't know what your agent will do until it's in production](https://blog.langchain.com/you-dont-know-what-your-agent-will-do-until-its-in-production/)
- [LangSmith Deployment components](https://docs.langchain.com/langsmith/components)
- [Agent Server](https://docs.langchain.com/langsmith/agent-server), by LangSmith

# 🗺️ Overview

In Sessions 3 and 4, we learned how to build agents with `create_agent` and LangGraph, customize them with middleware, and observe their behavior through LangSmith tracing. In this session, we take those agents to production.

The challenge, as the LangChain team puts it: agents are easy to prototype and hard to ship to production. Any input or change to an agent can create a host of unknown outcomes. Building reliable agents requires what they call **agent engineering** — the iterative process of refining non-deterministic LLM systems into reliable experiences that combines product engineering and data science.

This session is about the "ship" in build, ship, and share. We'll take an agent from a local `langgraph dev` environment all the way to a deployed API endpoint that a frontend application can call.

# 🏷️ What Changed: LangGraph Platform → LangSmith Deployments

If you've heard us reference "LangGraph Platform" or even "LangServe" in earlier sessions — those names are gone. As of LangChain v1.0, **all deployment functionality lives under LangSmith**. The Deployments tab in LangSmith is the single place to deploy, manage, and monitor your agent APIs in production.

| Old Name | New Name |
|---|---|
| LangGraph Platform | LangSmith (Deployments tab) |
| LangServe | Deprecated; replaced by LangSmith deploys |

Everything else in LangSmith — tracing, evaluation, prompt testing — remains where you left it. The deployment piece is simply a new tab sitting alongside those familiar tools.

# 📁 Project Structure for Deployment

Before you can deploy, your project needs a specific structure. The key requirement is a **`langgraph.json`** config file at the root of your repository. This is the same config format used when running locally with `langgraph dev`, so if you've been developing locally with LangGraph Studio, you're already set.

Your `langgraph.json` points to the agent definitions in your source code. For example, a project with multiple agents (a simple agent, a human-in-the-loop agent, a RAG agent, and a multi-agent supervisor) would reference each from a common `src/` directory.

The entire repo must be **pushed to GitHub** — LangSmith pulls directly from your repository to build the deployment.

# 🧪 Local Development → Studio → Deploy

The deployment workflow follows a natural progression:

## Step 1: Develop Locally

Run your agent server locally with:

```bash
langgraph dev
```

or, if using `uv`:

```bash
uv run langgraph dev
```

This launches **LangGraph Studio**, where you can interact with every agent defined in your `langgraph.json`. Studio lets you send messages, observe human-in-the-loop interrupts, and validate behavior before deploying anything.

## Step 2: Validate in Studio

LangGraph Studio gives you a visual interface to test all your agents. You can select between your different agent configurations (e.g., simple agent, RAG agent, human-in-the-loop agent, multi-agent), send messages, and watch the execution unfold — including interrupt/resume flows for human-in-the-loop patterns.

Once you're confident the agent behaves correctly, you're ready to deploy.

## Step 3: Deploy via LangSmith

Navigate to the **Deployments** tab in LangSmith and click **New Deployment**. The configuration requires:

- **Repository**: Select the GitHub repo containing your agent code
- **Name**: A label for this deployment (e.g., `my-rag-agent-v1`)
- **Branch**: Which branch to build from (e.g., `main`)
- **Config file**: Path to your `langgraph.json` (defaults to root)
- **Auto-update**: Optionally rebuild automatically on every push to the selected branch — whenever you commit to `main`, the deployment rebuilds
- **Environment variables**: Add API keys (e.g., `OPENAI_API_KEY`) and **mark them as secrets**
- **Tracing project**: Associate a LangSmith tracing project so all production runs are automatically observed

Hit **Submit** and LangSmith queues the job. You'll see the build progress through stages: queued → building → agent server launching. Build logs stream in real time so you can watch for errors.

# ⚠️ Critical: API Backend Only

This is the most important thing to understand about LangSmith deployments:

> **LangSmith deploys your agent as an API backend only.** It does not serve a frontend.

This means LangSmith is **not** a replacement for something like Vercel. What you get is a hosted API endpoint that your agent runs behind. You still need a separate frontend deployment that calls into this API.

A typical production architecture looks like:

- **Frontend** (e.g., deployed on Vercel, Netlify, etc.) → calls into →
- **Agent API** (deployed on LangSmith) → traced and monitored by →
- **LangSmith Observability** (tracing, evals, metrics)

# 🔄 CI/CD with Auto-Deploy

When you enable **auto-update on push**, you get a lightweight CI/CD pipeline out of the box. Every commit to your configured branch triggers a rebuild and redeploy. This means your workflow becomes:

1. Develop and test locally with `langgraph dev` + Studio
2. Push to `main`
3. LangSmith automatically rebuilds and deploys

No manual intervention needed after the initial setup. This is especially powerful when combined with LangSmith's tracing — you can push a change, watch the deployment rebuild, and immediately observe production traces to see if the new version behaves as expected.

# 🧭 Choosing Your Deployment Path

LangSmith one-click deploy is not the only option. Here's how to think about it:

| Approach | Best For | Trade-off |
|---|---|---|
| **LangSmith Deploy** | Teams already in the LangX ecosystem who want tracing + deploy in one place | Requires LangSmith Plus (~$40/month); API backend only |
| **Self-hosted** (`langgraph dev` in production) | Teams with existing infrastructure who want full control | You manage scaling, uptime, and monitoring yourself |
| **Custom API** (FastAPI, Flask, etc.) | Teams that need a non-LangGraph deployment or have specific infrastructure requirements | No built-in tracing integration; more setup work |

For most teams building with LangChain and LangGraph, LangSmith Deploy is the path of least resistance to getting a production API endpoint stood up quickly.

---

Do you have any questions about how to best prepare for Session 15 after reading? Please don't hesitate to provide direct feedback to `greg@aimakerspace.io` or `Dr Greg` on Discord!
