# Session 8: 🕵️ Deep Research

🎯 Understand how deep research systems work under the hood and how to build them.

📚 **Learning Outcomes**
- Understand the Bitter Lesson and why general methods that leverage computation win in the long run
- Learn the lessons the LangGraph team learned building Open Deep Research — including when to add and remove structure
- Understand the three-step process for conducting research: scope, research, write
- Recognize deep research as a composition of patterns you already know: planning, reflection, tool use, and multi-agent orchestration

🧰 **New Tools**
- [Open Deep Research](https://github.com/langchain-ai/open_deep_research)
- [Deep Research from Scratch](https://github.com/langchain-ai/deep_research_from_scratch)

## 📛 Required Tooling & Account Setup
No additional tools or accounts required.

## 📜 Recommended Reading
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) by Rich Sutton (March 2019) — the foundational essay on why general methods leveraging computation always win
- [Learning the Bitter Lesson](https://rlancemartin.github.io/2025/07/30/bitter_lesson/) by Lance Martin (July 2025) — how the Bitter Lesson applies directly to building deep research systems
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) (June 2025) — Anthropic's lessons on system architecture, tool design, and prompt engineering for research agents
- [Deep Research Bench](https://deepresearch-bench.github.io/) — 100 PhD-level research tasks for benchmarking deep research systems

# 🗺️ Overview

The killer app of 2025 — the multi-agent system that helps us search and research — has seen broad adoption throughout the industry and serves as our primary cohort use case, which we build from scratch using OSS tools.

Doing **Deep Research** — autonomously exploring, gathering, and synthesizing information from various sources (e.g., search tools, reference documents, or code execution) — is the kind of capability that, if you can build intelligently for your domain and organization, will make you indispensable to leadership undergoing an AI transformation.

## The Bitter Lesson

Before diving into deep research, we need to understand the philosophical foundation behind how these systems evolve.

**Rich Sutton** — pioneering AI researcher, 2024 Turing Award winner, and the father of reinforcement learning — penned an incredibly influential essay in March 2019 called [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).

The core argument: **general methods that leverage computation are ultimately the most effective, and by a large margin.** The reason is Moore's Law — the continued, exponentially falling cost per unit of computation. Over a slightly longer time than a typical research project, massively more computation inevitably becomes available.

The Bitter Lesson is based on these historical observations:

1. AI researchers have often tried to build knowledge into their agents
2. This always helps in the short term, and is personally satisfying to the researcher
3. In the long run, it plateaus and even inhibits further progress
4. Breakthrough progress eventually arrives by an opposing approach based on **scaling computation through search and learning**

The two methods that seem to scale arbitrarily are **search** and **learning**. As Lance Martin summarized: the structure we impose on models often limits their ability to leverage growing computation. The practical takeaway for builders: **add structures for the given level of compute and data available, and plan on removing them later** — because they'll become bottlenecks.

We've seen this pattern play out across AI history: the GPT-3 in-context learning paper (May 2020), the Chinchilla scaling paper (March 2022), and the scaling of test-time compute with reasoning models (Google, August 2024).

## The Deep Research Landscape

The TLDR on deep research: **these tools generate reports.** They take a prompt, do deep research, and produce a report. Every major AI lab has shipped their version:

### The Timeline

| Date | Release | Key Detail |
|------|---------|------------|
| **Dec 11, 2024** | [Google Deep Research](https://blog.google/products/gemini/google-gemini-deep-research/) | First mover — explore complex topics on your behalf |
| **Feb 2, 2025** | [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/) | Find, analyze, and synthesize hundreds of online sources |
| **Feb 4, 2025** | [Hugging Face Open Deep Research](https://huggingface.co/blog/open-deep-research) | 24-hour mission to reproduce and open-source |
| **Feb 14, 2025** | [Perplexity Deep Research](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research) | Research with reasoning — iterative search, read, reason |
| **Feb 19, 2025** | [Grok DeepSearch](https://x.ai/news/grok-3) | Lightning-fast agent to seek truth across human knowledge |
| **Feb 20, 2025** | [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research) | Lance Martin and team — the system we build today |
| **Apr 15, 2025** | [Claude Research](https://www.anthropic.com/news/research) | Agentic search with building investigations and citations |
| **Jun 2025** | [Microsoft Researcher](https://www.microsoft.com/en-us/research/) | Researcher + Analyst; later added computer use |
| **Jun 2025** | Chinese labs (Baidu Keen Fan, Dubao, Kimi) | Competing on Deep Research Bench |
| **Jul 2025** | [Mistral Deep Research](https://mistral.ai/news/le-chat-dives-deep) | Tool-augmented deep research agent |
| **Aug 2025** | NVIDIA | Deep research entry |
| **Oct 2025** | Salesforce | Enterprise deep research |
| **Nov 2025** | Google adds Deep Research to Notebook LM | Deep research integrated into the viral podcast tool |
| **Dec 11, 2025** | Gemini Deep Research | One year anniversary update |
| **Jan 2026** | [Tavily Deep Research](https://tavily.com/) | From GPT Researcher (July 2023) → $25M raise → research API |

> 🧪 **Aside:** Using AI to accelerate research is not a new idea. See [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) and its [math breakthrough](https://www.youtube.com/watch?si=Xp8pHr-RjWb6YOUY&v=vC9nAosXrJw&feature=youtu.be) — beating the 1969 Strassen algorithm record for 4×4 matrix multiplication (from 49 down to 48 scalar multiplications). More computation, more search, more learning.

## Deep Research Bench

The [Deep Research Bench](https://deepresearch-bench.github.io/) provides 100 PhD-level research tasks meticulously crafted by humans across domains like automation controls engineering, finance, and more. It uses evaluation frameworks like:

- **RACE** — Reference-based, Adaptive, Criteria-driven Evaluation
- **FACT** — Framework for Factual Abundance and Citation Trustworthiness

Notably, LangChain's Open Deep Research (the system we build today) is competitive on this benchmark alongside commercial offerings.

## Lessons from Anthropic's Research System

Anthropic published what they learned building their multi-agent research system — lessons on system architecture, tool design, and prompt engineering. Key patterns:

- A **planning agent** that scopes the research and generates a task list
- **Parallelized sub-agents** that go off and execute research tasks using tools
- **Compression** — sub-agents distill insights from large volumes of data, which is almost the essence of search itself

## Learning the Bitter Lesson in Practice

Lance Martin (creator of Open Deep Research, now at Anthropic) documented how the Bitter Lesson played out in his own work building deep research systems:

**The original architecture (early 2025):** Scope → parallel research + writing per section → combine. This imposed structure: forced parallel writing produced disjoint reports, and no tool calling meant missing out on the growing MCP ecosystem.

**What went wrong:** As tool calling improved and MCP gained momentum through winter 2025, the imposed structure became a bottleneck. The reports were sometimes disjoint because workers were forced to write sections in parallel.

**The evolved architecture:** Scope → parallel research agents (with tools) → gather all research → **one-shot final report writing.** This removes structure: writing moves to a single final step where all research context is available, producing more coherent reports. Research agents get more agency with tools rather than rigid workflows.

The key design principles:
- **Understand your structure** — know what constraints you've imposed and why
- **Re-evaluate structure as models improve** — what was necessary with weaker models may now be a bottleneck
- **Make it easy to remove structure** — design for flexibility so you can drop in better models and capabilities

This connects directly to context engineering: *can you get every single thing you need in context for this step, where you are in the process?*

## Building Deep Research: Key Patterns

Deep research is a composition of patterns you already know:

### Planning
A planning agent generates a task list for the research. This is the critical first step — just like project management, you review the plan before executing. Use a **reasoning model** for planning to leverage test-time compute.

### Reflection
An LLM assessing its own thinking — "are you sure, bro?" Based on the self-refinement paper (2023) and the LLM-as-a-judge pattern. Reflection can be:
- **Ungrounded** — the model critiques its own output (what's good, what's bad, regenerate)
- **Grounded** — reflection connected to external tools and clear success criteria (reflexion pattern)

### Configuration & Depth Control
The "deep" in deep research comes from configurable depth: max researcher iterations, max ReAct tool calls, token budgets. This lets you trade off cost/latency vs. thoroughness:
- Low settings for quick, cheap runs
- High settings when latency doesn't matter and the task requires deep investigation

### When to Use Deep Research
Deep research shines for **PhD-level hard questions** — tasks that require connecting insights across documents and web pages ("long-distance connections"). You don't send a PhD down a technician task route — you're wasting tokens. The sweet spot: tasks where humans currently spend hours pulling newsletters, stocks, news, and other sources to synthesize insights.

As models get faster and cheaper, this pattern becomes even more attractive. If deep research takes 15 seconds instead of 15 minutes, why wouldn't you always use it?

## Should You Build Your Own or Use an API?

Both OpenAI and Tavily now offer deep research APIs. You *could* just hit those. But:

- **Everybody can use the API** — if you're building a product, there has to be something you're doing beyond forwarding queries
- **Building your own gives you control** — tailor the implementation to your domain, your users, your data sources
- **Deep research for your org** — the real opportunity is building a research agent specific to your function, department, or organization, connected to your internal databases and tools

## Trust and Evaluation

How do we know deep research gives us good answers? 

- **Day one: you can't trust it.** Get a human to check everything, especially if it could lose money or upset customers.
- **Over time: trust builds.** If it doesn't mess up 100 times in a row, you start feeling risk-tolerant. This is how all technology adoption works.
- **Trust but verify.** Deep research systems are fundamentally citation aggregators. Use the citations to verify both credibility and accuracy.
- **Benchmarks are emerging** (Deep Research Bench) but still new and of debatable quality. This is a great area to contribute if you have ideas.

---

We look forward to building our own Deep Research application!
