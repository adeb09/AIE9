# Session 16: LLM Servers & Dedicated Endpoints (Fireworks AI)
### Interview-Ready Study Guide — Senior/Staff AI Engineer

---

## A. Core Concept Summary

LLM serving is a systems problem masquerading as a model problem. The critical insight is that **the same model can have 10× different production characteristics depending purely on how it's served** — serverless shared infrastructure versus dedicated GPU endpoints are two fundamentally different cost/latency/reliability contracts, and choosing wrong can tank a product's economics or SLA compliance.

Fireworks AI sits in a distinct tier of the LLM-serving landscape: it surfaces open-source frontier models (Llama, Mistral, Mixtral, Qwen, etc.) through a fully OpenAI-compatible API, meaning zero application-layer refactoring when switching providers. This interoperability collapses the switching cost that previously locked teams into OpenAI or Anthropic.

The mental model a practitioner must internalize is the **serving triangle**: latency, throughput, and cost — you can optimize for two. Dedicated endpoints let you control all three to a tighter band but require capacity planning. Serverless endpoints abstract that planning away at the cost of variance and cold-start risk. Every architectural decision in LLM serving is a navigation of this triangle under real traffic distributions.

For staff-level thinking, the key leverage point is matching the **traffic shape** of your workload to the pricing model of the endpoint. Bursty, low-volume workloads bleed money on dedicated capacity; high-volume, latency-sensitive workloads hit P99 ceilings on serverless. Getting this wrong by 2× on volume projections translates directly to engineering credibility loss and budget overruns.

---

## B. Key Terms & Definitions

- **TTFT (Time to First Token):** The latency from request submission to the arrival of the first generated token. Drives perceived responsiveness in streaming UIs; dominated by prefill compute and queue depth at the server.

- **TPOT (Time Per Output Token):** The average time between consecutive generated tokens after the first. Determines "streaming feel" and is gated by the autoregressive decode step — proportional to model size and inversely proportional to batch efficiency.

- **Throughput (tokens/sec):** The sustained rate at which a server produces output tokens across all concurrent requests. The primary metric for cost efficiency; higher throughput means lower cost-per-token at fixed GPU cost.

- **P50 / P99 Latency:** Median and 99th-percentile response latencies across a distribution of requests. P50 is your "typical" user experience; P99 is your SLA ceiling. In LLM serving, P99 can be 5–10× P50 under load due to queue contention and variable output lengths.

- **Dedicated Endpoint:** A provisioned serving instance with reserved GPU capacity exclusively allocated to your workload. Eliminates cold starts, provides predictable latency, but charges for idle capacity.

- **Serverless Inference:** Pay-per-token serving on shared, multi-tenant GPU infrastructure. Elastic by design, but exposes workloads to cold start penalties (seconds to tens of seconds on first request) and latency variance under shared load.

- **Cold Start:** The initialization delay when a serverless function or container spins up from zero. For LLMs, this includes model loading into GPU VRAM — can be 10–60 seconds for large models and is a P99 killer.

- **Knee of the Curve:** The inflection point on a throughput vs. concurrency graph where adding more concurrent requests stops increasing throughput and starts degrading per-request latency. Identifying this point is the goal of load testing.

- **KV Cache:** The key-value attention cache maintained during autoregressive decoding. Its memory footprint grows linearly with sequence length and is the primary driver of GPU memory constraints in dedicated serving, directly limiting max concurrency per GPU.

- **OpenAI-Compatible API Surface:** A REST API that mirrors OpenAI's `/v1/chat/completions` and `/v1/embeddings` schemas exactly, enabling drop-in substitution of the endpoint URL and API key without code changes.

---

## C. How It Works — Technical Mechanics

### The Serving Stack

A request to a Fireworks endpoint travels through: client → Fireworks load balancer → routing layer → inference worker (vLLM/TGI or proprietary runtime) → GPU → token stream back to client via SSE.

For **serverless**, the routing layer picks from a pool of shared workers, potentially cold. For **dedicated**, your traffic routes exclusively to your reserved worker(s). The inference runtime (often vLLM under the hood) handles continuous batching — packing multiple concurrent requests into a single forward pass to maximize GPU utilization.

### Continuous Batching vs. Static Batching

Static batching waits to fill a fixed batch before running a forward pass — simple but wasteful when requests complete at different times. **Continuous batching** (pioneered by vLLM) inserts new requests into in-progress batches as slots free up. This is why TPOT can spike under load even on dedicated endpoints — the effective batch size grows, slowing per-token decode for individual requests.

### Prefill vs. Decode Phase

LLM inference has two distinct compute phases:
1. **Prefill**: Processes the entire input prompt in parallel (high GPU utilization, fast). Drives TTFT.
2. **Decode**: Generates tokens one at a time autoregressively (memory-bandwidth bound, slower). Drives TPOT.

Long prompts (RAG context, many few-shot examples) significantly increase prefill time and therefore TTFT. This is why speculative decoding and prefix caching are high-leverage optimizations in production.

### Serverless vs. Dedicated Endpoint — Decision Matrix

| Dimension | Serverless | Dedicated Endpoint |
|---|---|---|
| **Latency (P50)** | Good (50–300ms TTFT at low load) | Excellent (20–100ms TTFT, stable) |
| **Latency (P99)** | Poor to variable (cold start: 10–60s) | Predictable (2–4× P50 under load) |
| **Cost Model** | Pay-per-token; zero idle cost | Hourly/monthly reserved capacity; cost at zero traffic |
| **Throughput ceiling** | Elastic — scales with demand (with latency variance) | Fixed by provisioned GPU count; requires explicit scaling |
| **Cold Start** | Yes — first request after idle period | None — always warm |
| **Data Privacy** | Multi-tenant; data co-located with other users | Single-tenant; data isolation guarantees |
| **Break-even Point** | Below ~40–60% GPU utilization equivalent | Above ~40–60% GPU utilization equivalent |
| **Best For** | Dev/test, bursty workloads, cost-sensitive low-volume | Latency-sensitive prod, high-volume, compliance-regulated |

### Fireworks AI Platform Mechanics

Fireworks exposes its model catalog via a single API base URL (`https://api.fireworks.ai/inference/v1`). Model IDs follow the pattern `accounts/fireworks/models/{model-slug}`. Key catalog entries: `llama-v3p1-70b-instruct`, `mixtral-8x7b-instruct`, `qwen2p5-72b-instruct`, `nomic-embed-text-v1` (embeddings).

Dedicated endpoint provisioning on Fireworks is a control-plane operation: you select a model, GPU type (A100 80GB, H100, etc.), replica count, and auto-scaling policy. The endpoint gets a stable URL. Pricing is per-GPU-hour; the cost math is straightforward — compare against (tokens/month × serverless-price-per-token) to find the break-even.

### Embedding Endpoint Mechanics

Fireworks embedding endpoints (e.g., `nomic-embed-text-v1`) expose the same `/v1/embeddings` schema as OpenAI. The critical compatibility check before switching: **embedding dimensions must match your vector store index**. `text-embedding-ada-002` outputs 1536 dims; `nomic-embed-text-v1` outputs 768 dims. You cannot mix models in a single Qdrant collection without re-indexing.

---

## D. Common Interview Questions (with Strong Answers)

---

**Q1: Walk me through how you would decide between serverless and dedicated endpoints for a new production RAG system expected to handle 500 concurrent users at peak.**

**A:** I'd start by characterizing the traffic shape before touching infrastructure. 500 concurrent users doesn't tell me the request rate or average latency target — I need tokens-per-second throughput and the acceptable P99 TTFT. If this is a customer-facing product with a 2-second TTFT SLA, serverless P99 variance is immediately disqualifying because a single cold start blows the SLA. I'd model the cost cross-over: if 500 concurrent users at, say, 500 tokens/request average response generates X million tokens/day, I'd price that on serverless (per-token) and on dedicated (hourly GPU cost × replicas needed to sustain that throughput). For most production systems above ~50K tokens/day per model, dedicated breaks even. I'd also factor in whether this is regulated data — HIPAA or SOC2 contexts push toward dedicated for tenant isolation guarantees regardless of cost.

*Staff-Level Extension: A principal interviewer will push on: "What if the traffic is highly diurnal — peaks at 10AM, near-zero at 3AM?" The right answer is auto-scaling dedicated endpoints on a schedule (or triggered by queue depth metrics), combined with a serverless fallback for burst-above-provisioned-capacity. This is a hybrid topology and requires the routing layer to be traffic-aware.*

---

**Q2: You're running a load test on a Fireworks dedicated endpoint and you observe that TTFT stays flat up to 20 concurrent requests then starts climbing sharply. What's happening and what do you do?**

**A:** That inflection is the knee of the curve — at 20 concurrent requests you've likely saturated the continuous batching capacity of the single GPU. What's happening mechanically: the prefill queue is building up, so new requests wait behind in-flight decodes before their prefill even begins, blowing TTFT. I'd first confirm this by plotting TTFT vs. queue depth — if they're correlated, it's queue contention, not model throughput that's the bottleneck. The fix depends on whether the constraint is GPU memory (KV cache exhausted — add a replica) or compute (add a larger GPU tier or enable quantization to reduce decode memory pressure and free up space for more concurrent slots). I'd also check whether speculative decoding is available on the runtime — it can improve throughput without adding hardware in decode-bound scenarios.

*Staff-Level Extension: "How do you set the knee programmatically as an autoscaling trigger?" — The answer is using a custom metric (queue_depth or p95_ttft from the serving runtime's metrics endpoint) as the autoscaling signal, not CPU/GPU utilization, which lags behind user-visible degradation.*

---

**Q3: How do you benchmark the quality difference between `nomic-embed-text-v1` on Fireworks vs. `text-embedding-ada-002` for your specific RAG use case, and what would make you choose one over the other?**

**A:** Generic MTEB scores are a starting point but not a decision. I'd build an eval set from actual production queries (or near-production synthetic queries generated from the document corpus) and run both models through a retrieval evaluation — precision@k and NDCG@k against labeled relevant documents. The dimension mismatch (768 vs. 1536) needs to be handled at the vector store level — separate collections for the eval. In practice, Nomic's models are competitive on semantic similarity tasks but can underperform on highly domain-specific or technical retrieval. The cost calculus usually makes Fireworks embeddings 5–10× cheaper than OpenAI at scale, so even a modest quality trade-off is often acceptable if retrieval metrics stay within a few percentage points. The decisive factor for me is usually the reindexing cost: if switching means re-embedding 100M documents, you need very high confidence in the quality delta before committing.

*Staff-Level Extension: "How do you handle the case where you're already in production with ada-002 embeddings and want to migrate?" — Zero-downtime migration requires running dual embedding models in parallel, building the new index shadow-mode, validating retrieval quality on live traffic (A/B shadow scoring), then flipping the routing. Never do a hard cutover on embeddings in production.*

---

**Q4: Describe a scenario where you would NOT use Fireworks or any open-source endpoint, and insist on staying with a frontier API like GPT-4o or Claude 3.5 Sonnet.**

**A:** Three clear scenarios: (1) **Hard capability gaps** — complex multi-step reasoning, code generation with long context, or tasks that require frontier-model calibration and instruction-following. I've seen Llama 70B fail on structured output adherence and tool-calling reliability in ways that GPT-4o handles cleanly, and the engineering workaround cost (retry logic, output validation, fallback paths) can exceed the model cost savings. (2) **Time-to-market pressure** — if I need to ship in 2 weeks and the quality bar requires frontier performance, I'll take the OpenAI cost hit and optimize later. (3) **Vendor-specific modalities** — if the use case requires vision + audio + function calling in a single pass, frontier models are often ahead of open-source equivalents by 6–12 months. The staff-level nuance is that "staying with frontier" doesn't mean permanent — I'd set up the abstraction layer so the LLM provider is swappable when capability parity catches up, which it usually does within a year.

---

**Q5: How would you design the cost monitoring and automatic shutdown strategy for a dedicated Fireworks endpoint serving an internal tool used only during business hours?**

**A:** Dedicated endpoints are pure burn when idle — 8 hours of business use out of 24 means you're paying for 16 hours of nothing. The architecture I'd use: a scheduled job (cron or Airflow) that calls the Fireworks control-plane API to start the endpoint 5 minutes before business hours start and stop it at EOD. For safety, I'd also attach an inactivity monitor: if no requests in 30 minutes during business hours (unexpected), trigger a Slack alert before stopping early to avoid a legitimate user getting a cold endpoint. Cost monitoring goes into a dedicated cost center tag, with a budget alert at 80% of monthly allocation. The risk to model here is the startup time for the dedicated endpoint — if it takes 3–5 minutes to provision, you need that buffer in the pre-start schedule. I'd measure this in staging and add margin. For a team tool, I'd also expose a `/warm-endpoint` slash command so any team member can pre-warm before a demo or heavy session.

---

**Q6: What are the data privacy and compliance implications of using Fireworks AI compared to self-hosted inference for a healthcare company's internal LLM?**

**A:** Fireworks offers SOC 2 Type II and can sign BAAs, which puts it in the same compliance tier as Azure OpenAI for most HIPAA use cases — it's not automatically disqualifying. The relevant questions are: (1) Is PHI in the prompt? If yes, BAA is required, and you need to verify whether Fireworks's data retention policy (are prompts logged? for how long?) meets your DPA requirements. (2) Is single-tenant isolation sufficient, or do you need on-premises data residency? Dedicated endpoints give you logical isolation but the infrastructure is still in Fireworks's cloud — if your compliance team requires data never to leave your VPC, self-hosted is the only answer. (3) What's the incident response SLA? For regulated data, you need a data breach notification commitment from the provider. The TCO comparison for self-hosted vs. managed is often the deciding factor — self-hosting an 8× H100 cluster for Llama 70B with redundancy, monitoring, and ops overhead can easily cost $500K+/year before you consider the engineering headcount. For most healthcare companies, managed + BAA + dedicated endpoints is the pragmatic path unless they're at hyperscaler scale.

---

## E. Gotchas, Trade-offs & Best Practices

- **Cold start is a silent SLA killer.** Teams building on serverless endpoints often miss cold starts in load tests because load tests keep the endpoint warm. The failure mode surfaces in production at 2AM or after a weekend. Always test with cold endpoints, add a health-check endpoint that the monitoring system pings every few minutes to keep serverless warm, or move to dedicated for latency-sensitive paths. The most common production incident pattern: a low-traffic feature works fine in load test, gets deployed, and the first real user after a 15-minute quiet period gets a 45-second timeout.

- **Embedding dimension lock-in compounds over time.** Every day you run production traffic with ada-002, your Qdrant index grows. Switching embedding models later means re-indexing all of it — not just new documents. The architectural best practice is to treat the embedding model as a versioned dependency from day one: store which model version was used per document, maintain a migration path in your indexing pipeline, and design the vector store collections to be model-version-scoped. Never co-mingle embeddings from different models in the same collection.

- **Throughput ceiling is not linear with GPU count.** Adding a second replica doubles raw compute but not necessarily observed throughput if the bottleneck is in the network or routing layer, or if your request pattern has highly variable output lengths (long-tail outputs serialize the batch). Always load test the actual multi-replica configuration, not just single-replica × N math. The 80/20 rule in LLM serving: 20% of requests with long outputs consume 80% of inference time and blow your throughput estimates.

- **The OpenAI compatibility shim hides breaking differences.** Fireworks's API is compatible at the HTTP/JSON schema level, but model behavior is not identical. Tool/function calling schemas, system prompt sensitivity, stop token handling, and structured output fidelity all differ by model and runtime. Teams that swap the base URL and API key and call it "done" inevitably get subtle regression bugs — wrong JSON format, truncated outputs, different tokenization counts affecting context window management. The right practice is a model behavior integration test suite that runs on every provider change.

- **Reserved capacity pricing is a negotiation, not a catalogue item.** At meaningful scale (multi-GPU, multi-month commitments), Fireworks and similar providers offer negotiated pricing that can be 30–50% below catalogue. Teams that treat cloud AI inference like a commodity SaaS subscription and never talk to sales are leaving significant margin on the table. For any deployment expected to exceed $20K/month in inference costs, a commercial negotiation is worth the effort.

---

## F. Code & Architecture Pattern

### Fireworks LLM + Embeddings via LangChain, RAG Chain, Throughput Test

```python
import os
import time
import asyncio
from typing import List

from langchain_fireworks import ChatFireworks, FireworksEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ── 1. Initialize Fireworks LLM and Embeddings ──────────────────────────────

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]

llm = ChatFireworks(
    model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    fireworks_api_key=FIREWORKS_API_KEY,
    temperature=0.0,
    max_tokens=1024,
    streaming=True,   # enables TTFT measurement via first chunk timing
)

embeddings = FireworksEmbeddings(
    model="nomic-ai/nomic-embed-text-v1",  # 768-dim; verify matches your Qdrant collection
    fireworks_api_key=FIREWORKS_API_KEY,
)

# ── 2. Build Qdrant Vector Store ─────────────────────────────────────────────

EMBEDDING_DIM = 768   # nomic-embed-text-v1 — do NOT mix with ada-002 (1536) collections

client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

COLLECTION_NAME = "fireworks_rag_demo"

# Idempotent collection creation
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# Seed with sample documents (in production: replace with your corpus loader)
sample_docs = [
    "Fireworks AI provides OpenAI-compatible inference for open-source LLMs.",
    "Dedicated endpoints reserve GPU capacity for predictable latency.",
    "Nomic embed text produces 768-dimensional embeddings suitable for semantic search.",
]
vector_store.add_texts(sample_docs)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ── 3. RAG Chain ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise technical assistant. Answer only from the provided context.
If the context does not contain the answer, say "I don't have that information."

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Single invocation with TTFT measurement
def invoke_with_ttft(question: str) -> dict:
    ttft = None
    full_response = []
    t_start = time.perf_counter()

    for chunk in rag_chain.stream(question):
        if ttft is None:
            ttft = time.perf_counter() - t_start
        full_response.append(chunk)

    total_time = time.perf_counter() - t_start
    response_text = "".join(full_response)
    token_estimate = len(response_text.split())  # rough; use tiktoken for precision

    return {
        "ttft_ms": round(ttft * 1000, 1),
        "total_ms": round(total_time * 1000, 1),
        "tpot_ms": round(((total_time - ttft) / max(token_estimate - 1, 1)) * 1000, 2),
        "tokens_estimated": token_estimate,
        "response": response_text,
    }

# ── 4. Throughput / "Endpoint Slammer" Test ──────────────────────────────────

async def single_request_async(session_id: int, question: str) -> dict:
    """Fires a single async request and returns latency metrics."""
    loop = asyncio.get_event_loop()
    t_start = time.perf_counter()
    # Run sync chain in thread pool to avoid blocking the event loop
    result = await loop.run_in_executor(None, lambda: invoke_with_ttft(question))
    result["session_id"] = session_id
    return result

async def slam_endpoint(
    question: str,
    concurrency_levels: List[int] = [1, 5, 10, 20, 40],
    requests_per_level: int = 10,
) -> dict:
    """
    Sweeps concurrency levels to find throughput ceiling.
    Returns per-level p50/p99 TTFT and aggregate tokens/sec.
    """
    results = {}

    for concurrency in concurrency_levels:
        print(f"\n→ Concurrency: {concurrency}")
        tasks = [
            single_request_async(i, question)
            for i in range(requests_per_level * concurrency)
        ]

        batch_start = time.perf_counter()
        # Run in batches of `concurrency` to simulate sustained load
        batch_results = []
        for i in range(0, len(tasks), concurrency):
            batch = tasks[i : i + concurrency]
            batch_results.extend(await asyncio.gather(*batch))

        wall_time = time.perf_counter() - batch_start
        ttft_values = sorted(r["ttft_ms"] for r in batch_results)
        total_tokens = sum(r["tokens_estimated"] for r in batch_results)

        n = len(ttft_values)
        results[concurrency] = {
            "p50_ttft_ms": ttft_values[int(n * 0.50)],
            "p99_ttft_ms": ttft_values[int(n * 0.99)],
            "throughput_tokens_per_sec": round(total_tokens / wall_time, 1),
            "total_requests": n,
        }

        p50 = results[concurrency]["p50_ttft_ms"]
        p99 = results[concurrency]["p99_ttft_ms"]
        tps = results[concurrency]["throughput_tokens_per_sec"]
        print(f"  P50 TTFT: {p50}ms | P99 TTFT: {p99}ms | Throughput: {tps} tok/s")

    return results

# ── 5. Main Entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Single invocation demo
    print("=== Single RAG Invocation ===")
    metrics = invoke_with_ttft("What embedding dimensions does Nomic produce?")
    print(f"TTFT: {metrics['ttft_ms']}ms | Total: {metrics['total_ms']}ms | ~{metrics['tokens_estimated']} tokens")
    print(f"Response: {metrics['response']}\n")

    # Throughput test
    print("=== Throughput Sweep (find the knee of the curve) ===")
    throughput_results = asyncio.run(
        slam_endpoint(
            question="What are the advantages of dedicated endpoints?",
            concurrency_levels=[1, 5, 10, 20],
            requests_per_level=5,
        )
    )

    # The "knee" is the first concurrency level where P99 TTFT jumps >2x P50
    for concurrency, stats in throughput_results.items():
        ratio = stats["p99_ttft_ms"] / max(stats["p50_ttft_ms"], 1)
        flag = " ← KNEE" if ratio > 2.5 else ""
        print(f"Concurrency {concurrency:>3}: P99/P50 ratio = {ratio:.2f}{flag}")
```

### Architecture Diagram (LangGraph RAG Agent on Open-Source Stack)

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Agent                       │
│                                                         │
│  User Query ──► Retrieval Node ──► Generation Node      │
│                      │                    │             │
│               [Fireworks Embed]    [Fireworks LLM]      │
│               nomic-embed-text     llama-v3p1-70b       │
│                      │                    │             │
│               ┌──────▼──────┐             │             │
│               │   Qdrant    │             │             │
│               │  (768-dim)  │─── docs ───►│             │
│               └─────────────┘             │             │
│                                           ▼             │
│                                    Final Response       │
└─────────────────────────────────────────────────────────┘

Cost model:
  - Fireworks LLM:        ~$0.20–0.90 / 1M tokens (model-dependent)
  - Fireworks Embeddings: ~$0.008 / 1M tokens (vs. $0.10 for ada-002)
  - Qdrant Cloud:         ~$0.05 / GB / hour for 1M 768-dim vectors
  
Savings vs. full OpenAI stack at 100M tokens/month:
  LLM:        60–80% depending on Llama vs. GPT-4o comparison
  Embeddings: ~12× cheaper with Nomic vs. ada-002
```

### Responsible Endpoint Management — Scheduled Start/Stop Pattern

```python
import httpx
import os

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
ENDPOINT_ID = os.environ["FIREWORKS_ENDPOINT_ID"]

BASE = "https://api.fireworks.ai/v1"
HEADERS = {
    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    "Content-Type": "application/json",
}

def set_endpoint_state(active: bool) -> dict:
    """Toggle dedicated endpoint on/off — use in scheduled jobs for cost control."""
    resp = httpx.patch(
        f"{BASE}/accounts/me/deployedModels/{ENDPOINT_ID}",
        headers=HEADERS,
        json={"active": active},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def get_endpoint_status() -> str:
    resp = httpx.get(
        f"{BASE}/accounts/me/deployedModels/{ENDPOINT_ID}",
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("state", "unknown")

# Usage in a cron job (e.g., run at 08:55 AM and 06:00 PM)
# 08:55 AM: set_endpoint_state(active=True)  → warm before business hours
# 06:00 PM: set_endpoint_state(active=False) → stop burning money overnight
```

---

*Study Guide generated for AIE9 Session 16 — LLM Servers & Dedicated Endpoints (Fireworks AI)*
