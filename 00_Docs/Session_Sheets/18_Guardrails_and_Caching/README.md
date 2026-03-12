# Session 18: 🛤️ Guardrails & Caching 

🎯 **Goal**: Learn a few upgrades: for performance and security/trustworthiness.

📚 **Learning Outcomes**
- Understand guardrails, including the key categories of guardrails
- Understand the importance of semantic caching
- How to use Prompt and Embedding caches
- Learn how to use a skills.md file

🧰 **New Tools**
- [Guardrails](https://guardrailsai.com/)
- [CacheBackedEmbeddings](https://docs.langchain.com/oss/python/integrations/text_embedding#caching) 
- [Prompt Caching](https://docs.langchain.com/oss/python/langchain/models#prompt-caching)
## 📛 Required Tooling & Account Setup
In addition to the tools we've already learned, in this session you'll need:
    
1. 🔑 Get your Guardrails AI API key from [here](https://hub.guardrailsai.com/keys).
   
## 📜 Recommended Reading

- [The AI Guardrails Index](https://www.guardrailsai.com/blog/introducing-the-ai-guardrails-index) (Feb 2025)
- [Caching](https://planetscale.com/blog/caching) (July 2025)
- [Semantic Caching for Low-Cost LLM Serving](https://arxiv.org/html/2508.07675v1) (Aug 2025)
- [Guardrails](https://docs.langchain.com/oss/python/langchain/guardrails), by LangChain

# 🗺️ Overview

In Session 18, we’ll cover a few nice-to-haves for our production LLM applications that will become more and more important as we scale:  Skills.md, Caching, and Guardrails

As they say, cache is king. If we can save some cash with 🏦 caching, we should. We can leverage both prompt and embedding caches in general, depending on our use cases.

🛤️  Guardrails are the runtime checks that keep your AI application’s inputs and outputs on track—enforcing safety, policy, brand, correctness, and structure. This one‑pager clarifies definitions, shows where guards live in your stack, summarizes a minimally sufficient guard set for 2025, and gives a tiny flowchart to choose the right guard quickly.

# *🏦* Caching

If you’re familiar with building digital applications in a non-LLM world already, caching will not be new to you. Caching is one of the easiest things that we can do to save cash. 

The TL;DR on caching is that `if we've seen it before, then we can just remember it`. We don’t need a fresh generation of new tokens, we can use ones that we’ve already saved from previous runs.

This works equally well for embeddings (e.g., during RAG retrieval processes), and for remembering responses to entire prompts, as long as similar ones have been used in the past.

## **Cache-Backed Embeddings**

The process of embedding data is time-consuming. For every vector in our vector database, as well as for every single query, do a few different things:

1. Send the text to an API endpoint (self-hosted, OpenAI, etc)
2. Wait for processing
3. Receive response

This process costs time and money. **The more data we have, the more time and money it costs!**

Instead, what if we could set up a cache to hold all of the data chunks that we’ve already converted to embeddings, and check it every time we need to use them?

**That’s exactly what we do**, and it works great.

That is, If we cache [query, context], *we can avoid using the retrieval parts of our RAG applications.* 

The process is as follows:

```python
1. Set up cache
2. Send text to embedding endpoint
3. Check cache
    - If Y, return cached
    - If N, convert text
        - Store new embeddings
4. Return vector embeddings
```

There are very few reasons *not to use caching* for production LLM applications.

## Prompt Caching

If we've seen this prompt before, we just use the stored response. That is, if we cache [query, response], we can *avoid using our entire RAG chain*.

In this way, we can speed up repeated large language model (LLM) requests by storing the results of certain parts of a prompt so they don’t have to be recomputed. Even OpenAI offers this as a service to help you get “automatic discounts on inputs that the model has recently seen” [[Ref](https://openai.com/index/api-prompt-caching)].

Just like embedding caching, we can decrease latency AND increase throughput with this approach, a truly magical combination 🪄.

# 🛤️ Guardrails

The definition of guardrails is slippery. We had a great discussion with the CEO of Guardrails AI, the tool that we’ll look at, about this. You can watch it [here](https://www.youtube.com/watch?v=gwfzhSu1F38&t=660s). Here is what we came up with in terms of a definition of guardrails:

> Guardrails are the policies and runtime controls that keep an AI system’s inputs and outputs on track. 

Practically, they’re input/output modules that monitor and constrain behavior in real time—detecting issues (e.g., safety violations, hallucinations), enforcing structure/access rules, and triggering fallbacks. 

In this framing, alignment is the goal state, evaluation is offline pre-deployment testing, and validation is the online verification guardrails often perform.
> 

We can break this down into a few components:

- **Guardrails (runtime):** Live validators around your LLM workflow that **check, block, fix, or route** content at **input** and **output**.
- **Alignment (goal):** System behaves per org values and product intent.
- **Evaluation (offline):** Pre‑deploy testing on datasets to pick guards, thresholds, and fallbacks.
- **Validation (online):** Inference‑time checks—what most guard frameworks do.
- **What a “guard” is:** Usually a **small, tuned classifier** (fast), sometimes **rules/heuristics**, with optional **LLM‑as‑judge fallback** for edge cases. Guards can also **enforce schemas** (valid JSON/XML/YAML) for tool/API calls.

## Practical Implementation

The first practical thing that we need to understand about guardrails is that they are typically *other LLMs*. We can put them on the input or the output side of our applications as pictured below:

![Event_Guardrails.jpg](attachment:33f0089d-55df-450c-8142-4751e61f0d03:Event_Guardrails.jpg)

We might use input or output guards for different things, for instance:

- **Input guards:** on‑topic, jailbreak/prompt‑injection, PII detection/redaction.
- **Output guards:** content moderation/profanity, restricted topics/brand policy, hallucination/faithfulness, competitor mentions, **schema validation/repair**.

This is visualized well in Guardrails AI documention [[Ref](https://github.com/guardrails-ai/guardrails?tab=readme-ov-file)]. We can see that we can about different things on either side!

![image.png](attachment:ca113505-42bf-480e-8c4d-1071031dbe3e:image.png)

## **Design patterns**

While models are getting ever-better at *aligning* to end users and use cases, and guardrails should be considered in the context of the LLM’s off-the-shelf ability to perform the required tasks, there are some best practices and design patterns to keep in mind when it comes to putting guardrails into your production LLM applications.

- **Rules vs. Models (e.g., Workflows vs. Agency)**: It is preferred to use simple rules/heuristics of very small models when possible for performance. Alternatively, we can add more robust larger LLM guardrails in a fallback capacity if we are not getting high-confidence outputs to send directly to users.
- **Beyond Escalation**: On‑fail strategies can go beyond escalating to a larger fallback model. An alternate, manual, labor-intensive approach would be to just put a human in the loop. The simplest strategy would simply be to log undesirable information and block the response with a default error message.
- **Internal vs. External Applications**: We want heavier guards on apps exposed directly to external users. In this case, it’s probably quite important to emphasize **output** policy/brand & **schema** correctness.
- **Guards as a Service**: We want to treat guards as any other LLM-powered service with monitoring of both relevant LLM-specific and application-specific metric monitoring, including but not limited to latency budgets, threshold tuning, A/B, drift monitoring. Iterate.

## 🦺 AI Guardrails Index

The AI Guardrails Index, released by Guardrails AI [[Ref](https://index.guardrailsai.com/)] covers key areas important to enterprise today, including:

1. **Jailbreak Prevention** — Detects prompt‑injection & role‑bypass patterns.
2. **PII Detection** — Finds/redacts sensitive entities (credit cards, SSNs, emails, phones, etc.).
3. **Content Moderation** — Toxicity/harassment/sexual/violent content filters.
4. **Hallucination / Faithfulness** — Checks answer grounding vs. provided sources (RAG‑aware) beyond simple relevance.
5. **Competitor Presence** — Blocks or flags mentions of disallowed brands/products.
6. **Restricted Topics** — Policy taxonomy for business‑specific “do/don’t discuss”.

We found it quite interesting to check out their model leaderboard, which plots accuracy/performance (via F1 Score) vs. cost (latency) for leading guardrails options. We talked abou this with the Guardrails cofounder [here](https://www.youtube.com/live/gwfzhSu1F38?si=Zb7ZtvtOseqr3lEW).

## Choose The Right Guard for The Job

While there are no universal rules for selecting guardrails, it is important to think about your application, what guards you need today, and what guards you might need in the future. We might consider a few axes of decision-making in general, and start tailoring to our specific use case.

```
  ├─> Is your app public-facing? ── Yes ──> Prioritize INPUT guards
  │                                   │
  │                                   └─ No ──> Prioritize OUTPUT + SCHEMA guards
  │
  ├─> Primary risk = Safety/Compliance? ── Yes
  │                                         │
  │                                         └─ No (Reliability/Brand) ──> Schema 
  │
  ├─> Tight latency budget? ── Yes ──> Small classifier / rules first
  │                              │
  │                              └─ No ──> Allow LLM‑judge fallback on low‑confidence
  │
  └─> Pick on‑fail: rais**e | fix | fallback | log+escalate**

```

**Quick Map (Risk → Guard → IO → Default On‑Fail)**

| Risk | Guard Type | I/O | Default On‑Fail |
| --- | --- | --- | --- |
| Policy/safety | Restricted topics | Output | **raise** |
| Privacy | PII detection/redaction | Input & Output | **fix** (redact) |
| Toxic/brand | Moderation/profanity | Output | **raise** (or fix if light) |
| Reliability | Schema validation/repair | Output | **fix** (repair JSON) |
| Factuality | Hallucination/faithfulness | Output | **fallback** (regenerate/safer route) |
| Abuse | Jailbreak/prompt‑injection | Input | **raise** |
| Brand | Competitor presence | Output | **raise** (or route) |
| Misuse | On‑topic classifier | Input | **raise** |

**Quick Start Checklist**

- Define **policies & risks** (scope per surface: chat, tools, actions).
- Pick guards per surface and set **latency budget**.
- Choose **thresholds** and **on‑fail behaviors**.
- **Evaluate offline** on real samples; collect FP/FN and adjust.
- **Validate online** with metrics (block rate, redaction rate, fallback rate, latency). Add alerts.

## Conclusions

- Guardrails ≠ only “safety”; they’re a **broad runtime control layer** for **safety + policy + reliability + structure**.
- Start small: **on‑topic + PII + moderation**, then add **schema** and **hallucination** checks for RAG/tooling.
- Favor **fast classifiers**; use **LLM fallback** sparingly where accuracy matters most.
- Close the loop: **evaluate offline, validate online, monitor & iterate**.

