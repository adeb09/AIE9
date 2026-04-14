# Session 12: Industry Use Cases & Capstone System Design
## Interview-Ready Study Guide — Senior/Staff AI Engineer

---

## A. Core Concept Summary

AI Engineering at the staff level is as much about _when not to build_ as it is about knowing how to build. The session-12 capstone synthesizes every layer of the AI Engineering stack — scoping, ingestion, retrieval, agent orchestration, evaluation, and stakeholder communication — into a single coherent system design discipline. The key mental model is the **AI Engineering Project Lifecycle**: a staged delivery process where each phase produces a falsifiable artifact (a benchmark, a latency p99, a precision@k score) that gates the next phase, rather than a waterfall toward a vague "deploy AI" goal. Production-grade agentic RAG is not a chatbot bolted onto a vector store; it is a distributed, async, observable pipeline where the retrieval contract, the agent loop, and the eval loop are first-class engineering concerns. A staff engineer must be able to articulate _why_ every architectural choice was made — embedding model, chunk size, retrieval strategy, orchestration framework — in terms of latency, cost, quality trade-offs, and failure modes, and must be able to translate those choices into stakeholder language that builds trust without overpromising.

---

## B. Key Terms & Definitions

- **"What Not to Build" Framework**: A decision gate applied before committing to RAG or agent architectures. Asks: does a keyword search, a deterministic DB query, or a rule engine already solve the problem at lower latency, lower cost, and zero hallucination risk? RAG/agents are justified only when the answer space is dynamic, knowledge is unbounded, or multi-step reasoning is genuinely required.

- **Vibe Check Prototype**: The first phase of the AI Engineering lifecycle — a minimal, non-production implementation whose sole purpose is to establish whether an LLM can solve the core task at all. It produces a gut-check pass/fail signal before any engineering investment in infrastructure.

- **Structured Eval Loop**: A repeatable, automated evaluation harness — typically using a framework like Ragas, LangSmith, or a custom pytest suite — that measures retrieval quality (precision@k, recall@k, MRR) and generation quality (faithfulness, answer relevance, context precision) against a fixed golden dataset. It is the gate between prototype and production.

- **Hybrid Vector Store**: A retrieval backend that combines dense vector similarity search (ANN via HNSW or IVF) with sparse BM25/keyword search, merging results via Reciprocal Rank Fusion (RRF) or a learned reranker. Required when queries span both semantic intent and exact token matching (e.g., code identifiers, version numbers).

- **LangGraph Agent**: A stateful, graph-structured agent orchestration pattern where each node is a deterministic computation step (tool call, LLM inference, branching logic) and edges encode conditional transitions. Enables reproducible, debuggable multi-step reasoning with explicit state management, unlike chain-based agents.

- **Async Ingestion Pipeline**: A Python `asyncio`-based pipeline that concurrently fetches, preprocesses, chunks, embeds, and upserts documents without blocking on I/O. Critical for large-scale document corpora where synchronous ingestion would be the wall-clock bottleneck.

- **Semaphore Rate Limiting**: A concurrency primitive (`asyncio.Semaphore`) used to cap the number of in-flight requests to a rate-limited external API (e.g., GitHub API at 5000 req/hr, OpenAI embeddings at N RPM). Prevents 429 storms without sacrificing parallelism.

- **Stamina Retry-with-Backoff**: The `stamina` library provides structured, observable retry logic with exponential backoff and jitter for transient failures (network timeouts, 503s, rate limit spikes). Preferable to bare `tenacity` because it integrates with structured logging and OpenTelemetry spans.

- **Reciprocal Rank Fusion (RRF)**: A rank aggregation algorithm that merges result lists from heterogeneous retrievers (dense + sparse) without requiring score normalization. `score = Σ 1/(k + rank_i)` where `k=60` is standard. Simple, effective, and parameter-insensitive compared to learned fusion.

- **Faithfulness (Ragas)**: An evaluation metric measuring whether every claim in a generated answer is attributable to the retrieved context. A faithfulness score < 0.8 is a signal that the model is confabulating beyond what retrieval supports — a hard stop before production deployment in high-stakes domains.

---

## C. How It Works — Technical Mechanics

### The "What Not to Build" Decision Framework

Before architecting any AI system, apply this decision tree in order:

```
1. Is the answer a deterministic lookup?
   → Yes: DB query / API call. Full stop.

2. Is the query intent fully expressible as a keyword pattern?
   → Yes: Elasticsearch / OpenSearch BM25. Faster, cheaper, no hallucination.

3. Is the task a structured classification or regression over a fixed schema?
   → Yes: Train a fine-tuned classifier or use a rules engine.

4. Is a hallucinated answer worse than no answer?
   → Yes (medical, legal, financial compliance): add strict grounding constraints,
     citation verification, or reject RAG entirely in favor of retrieval-only.

5. Does the task require dynamic, multi-hop reasoning over a large, evolving corpus?
   → Now RAG/agents are justified.
```

The most common mistake senior engineers see from mid-level candidates: jumping to RAG for problems that are fundamentally keyword search with a semantic wrapper. RAG adds latency (50–200ms retrieval + 500ms–2s LLM), cost (embedding inference + LLM tokens), and a new failure mode (retrieval misses) for zero benefit when BM25 would answer the query with 99% precision.

---

### The AI Engineering Project Lifecycle

```
Phase 0: Problem Scoping
  ├── Define the task contract: input format, output format, acceptance criteria
  ├── Identify the "oracle": how would a human verify correctness?
  ├── Apply the "What Not to Build" framework
  └── Output: signed-off problem statement + evaluation criteria

Phase 1: Vibe Check Prototype
  ├── Minimal RAG: naive chunking, OpenAI embeddings, in-memory FAISS, GPT-4o
  ├── 10–20 golden Q&A pairs, manual evaluation
  └── Output: binary signal — "LLM can do this task" vs. "task requires rethink"

Phase 2: Structured Eval
  ├── Build golden dataset (50–200 Q&A pairs + source citations)
  ├── Instrument Ragas: faithfulness, answer_relevance, context_precision, context_recall
  ├── Baseline metrics established — this is your regression floor
  └── Output: eval harness that runs in CI on every retrieval change

Phase 3: Retrieval Upgrade
  ├── Move from naive chunking → semantic/hierarchical chunking
  ├── Add hybrid retrieval (BM25 + dense) + RRF fusion
  ├── Add reranker (Cohere Rerank, cross-encoder) for top-k refinement
  └── Output: +∆ on context_recall and context_precision vs. baseline

Phase 4: Production Hardening
  ├── Async ingestion pipeline with semaphore + retry
  ├── Persistent vector store (Qdrant, Weaviate, Pinecone) with versioned collections
  ├── LangGraph agent with explicit state, tool calling, fallback edges
  ├── Input/output guardrails (content filtering, citation verification)
  └── Output: p99 latency SLA, cost-per-query budget, error budget

Phase 5: Monitoring
  ├── Trace every agent run with LangSmith or OpenTelemetry
  ├── Online eval: thumbs-up/down + implicit signals (follow-up questions = retrieval miss)
  ├── Embedding drift detection: monitor query embedding distribution over time
  └── Output: alerting on faithfulness degradation, retrieval latency spikes
```

---

### Capstone Architecture: Production Agentic RAG System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ASYNC INGESTION LAYER                            │
│                                                                         │
│  GitHub API (gidgethub/async)                                           │
│    └── Semaphore(10) → aiohttp fetch → stamina retry                    │
│         └── Document queue (asyncio.Queue)                              │
│              └── Preprocessing workers (N concurrent)                  │
│                   ├── Markdown → text extraction                        │
│                   ├── Code → AST-aware chunking (by function/class)     │
│                   ├── Prose → semantic chunking (sentence-window)       │
│                   └── Metadata attachment (repo, path, commit SHA)      │
│                        └── Embedding batch (text-embedding-3-large)     │
│                             └── Upsert → Qdrant (versioned collection)  │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────────┐
│                        HYBRID VECTOR STORE                              │
│                                                                         │
│  Qdrant                                                                 │
│    ├── Dense index: text-embedding-3-large (3072d, cosine)              │
│    └── Sparse index: BM25 (SPLADE or Qdrant native sparse vectors)      │
│                                                                         │
│  Retrieval: RRF(dense_results, sparse_results, k=60)                   │
│  Reranking: Cohere Rerank v3 on top-20 → top-5                         │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────────┐
│                        LANGGRAPH AGENT LOOP                             │
│                                                                         │
│  State: { query, retrieved_docs, tool_calls, answer, citations }        │
│                                                                         │
│  Nodes:                                                                 │
│    query_analyzer → route_intent → [retrieve | sql_query | api_call]   │
│         └── synthesize → citation_verifier → output_guardrail          │
│                                                                         │
│  Edges:                                                                 │
│    citation_verifier → FAIL → re_retrieve (max 2 hops)                 │
│    output_guardrail → FAIL → refuse_with_explanation                   │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────────┐
│                        EVALUATION LOOP                                  │
│                                                                         │
│  Offline: Ragas on golden dataset (CI gate)                             │
│    ├── faithfulness ≥ 0.85                                              │
│    ├── answer_relevance ≥ 0.80                                          │
│    ├── context_precision ≥ 0.75                                         │
│    └── context_recall ≥ 0.70                                            │
│                                                                         │
│  Online: LangSmith traces + implicit feedback signals                   │
│    ├── Latency p50/p95/p99 per node                                     │
│    ├── Tool call frequency distribution                                 │
│    └── Citation verification pass rate                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Key Design Decisions at Each Layer

**Embedding Model Selection**
- `text-embedding-3-large` (3072d) vs. `text-embedding-3-small` (1536d): the large model yields ~5–8% improvement on MTEB retrieval benchmarks at 2× the cost and latency. For code, `voyage-code-2` or `text-embedding-3-large` with code-specific fine-tuning outperforms generic models by 15–20% on function retrieval tasks.
- Matryoshka embeddings (MRL): `text-embedding-3-large` supports truncation to 256d for speed/cost when full precision is not required.

**Vector DB Selection**
- Qdrant: strong choice for on-prem/self-hosted, native sparse+dense hybrid, Rust performance, versioned collections for zero-downtime re-indexing. Weakness: operational burden.
- Pinecone: managed, serverless, minimal ops. Weakness: no native sparse index until recently, less control over HNSW parameters.
- pgvector: acceptable for < 1M vectors with existing Postgres infra. Falls off above that due to sequential scan at high recall targets.

**Chunk Strategy Selection**
- Fixed-size (512 tokens, 50-token overlap): baseline. Cheap, predictable. Fails on documents where logical units span variable lengths.
- Semantic chunking: split on embedding cosine similarity drops between sentences. Better context coherence, variable chunk size.
- Hierarchical (parent-child): store small chunks for retrieval precision, retrieve large parent chunk for synthesis context. Best for long-form documents where a small passage anchors a large reasoning unit.
- AST-aware (code): chunk by function/class boundaries, not token count. Critical for code repositories — token-boundary chunking splits function signatures from bodies.

**Retrieval Strategy Selection**
- Dense-only: high recall on semantic queries, poor on exact identifiers (function names, error codes).
- Sparse-only (BM25): precise on keywords, poor on paraphrase/synonym queries.
- Hybrid + RRF: best of both. Default choice for mixed corpora.
- Reranker (cross-encoder): adds 30–100ms but can recover 10–15% on precision@5. Worth it when top-k quality is critical (e.g., answer synthesis relies on top-3).

---

## D. Common Interview Questions (with Strong Answers)

---

### Q1: "Walk me through how you would decide whether to use RAG vs. fine-tuning vs. a deterministic system for a new use case."

**A:** The decision hinges on three axes: knowledge locality, answer determinism, and hallucination tolerance. If the knowledge is static and task-specific (e.g., a domain classifier), fine-tuning a smaller model is cheaper and more reliable than RAG — you get a deterministic, low-latency inference path without retrieval overhead. If the knowledge is dynamic, large, or frequently updated (e.g., a codebase that changes weekly), RAG wins because you can re-index without retraining. If the answer can be derived from a structured query — a SQL join, an API call — you should not involve an LLM at all; the hallucination risk is never zero and the determinism of a DB query is strictly better. Fine-tuning and RAG are not mutually exclusive: a fine-tuned model that is better at following citation-grounded instructions, combined with RAG retrieval, is the production pattern for high-stakes domains. The staff-level answer is that this decision should be made against a measured baseline — run BM25 first, measure precision@k on your golden set, and only justify the incremental complexity of RAG if the delta is significant.

> **Staff-Level Extension:** "How does this decision change when the use case has zero-shot queries that span multiple internal data sources — e.g., a question that requires joining GitHub code context with Confluence documentation?" Now you're in multi-source RAG territory, which introduces retrieval routing and source fusion as first-class problems. The interviewer wants to hear about intent-based query routing, collection-specific retrieval, and how you merge results from heterogeneous corpora without score normalization artifacts.

---

### Q2: "Your RAG system has high answer relevance but low faithfulness scores. What do you do?"

**A:** Low faithfulness with high answer relevance means the model is generating fluent, on-topic answers that are _not grounded_ in the retrieved documents — it's confabulating from parametric knowledge. First, I'd inspect the trace: are the retrieved chunks actually relevant to the query, or is retrieval missing the grounding context entirely? If retrieval is the problem, faithfulness is a downstream symptom — fix context_recall first. If retrieval is correct but the model is still hallucinating, the issue is in the synthesis prompt: the model needs an explicit "answer only from the provided context, cite the source for every claim, say 'I don't know' if the context is insufficient" instruction. Adding a citation verifier node in the LangGraph pipeline — which checks that every claim in the output maps to a span in the retrieved context — catches this at inference time. If faithfulness stays low after both fixes, it's a signal that the task requires a model with stronger instruction-following for grounded generation (GPT-4o vs. GPT-3.5-turbo, or a fine-tuned grounding model).

> **Staff-Level Extension:** "How would you detect faithfulness degradation in production without running Ragas on every query?" Online proxies: monitor the citation verification pass rate (the node already exists), track follow-up clarification rates (a high follow-up rate signals the answer didn't resolve the query), and sample 1–5% of traces for offline Ragas evaluation on a rolling basis. Set alerting on the citation pass rate dropping below a threshold — this is a leading indicator for faithfulness degradation.

---

### Q3: "Design an async Python ingestion pipeline for a GitHub repository corpus at scale. What are the failure modes?"

**A:** The pipeline has three I/O-bound stages: fetching from the GitHub API, calling the embedding API, and upserting to the vector store. Each is rate-limited by a different constraint — GitHub at 5000 req/hr authenticated, OpenAI embeddings at 3000 RPM (model-dependent), and the vector store at write throughput. I'd model this as a producer-consumer pipeline using `asyncio.Queue`: a GitHub fetcher with a `Semaphore(10)` produces raw documents, a preprocessing pool transforms them, and an embedding batcher with a `Semaphore(5)` consumes chunks in batches of 100. Stamina handles transient 429s and 503s on both the GitHub and embedding API calls with exponential backoff + jitter. The key failure modes are: (1) pagination bugs — GitHub's link-header pagination must be followed explicitly, and a missed page means silent data loss; (2) embedding batch size limits — OpenAI's embeddings API has a token limit per batch, not just a request limit, so batching by token count not item count; (3) vector store write amplification — upserting individual vectors is ~10× slower than batch upsert, so always batch; (4) partial failures mid-ingestion — without a checkpoint mechanism, a crash at 80% ingestion requires a full re-run, so write a progress manifest keyed on document hash.

> **Staff-Level Extension:** "How would you make this pipeline idempotent and resumable?" Hash each document (repo + path + commit SHA) and check existence in the vector store before re-embedding. Store ingestion state in a lightweight checkpoint store (Redis or a SQLite manifest). On restart, skip already-ingested document hashes. This also enables incremental re-indexing on push events without full corpus re-ingestion.

---

### Q4: "How do you communicate the reliability limitations of an agentic RAG system to a non-technical stakeholder who wants a guarantee that the system won't hallucinate?"

**A:** I frame this in terms of the stakeholder's actual risk, not LLM internals. The honest message is: "This system is designed to answer only from verified sources, and every claim in its output is linked to a source document you can audit. Our citation verification gate rejects answers that can't be grounded — so the system will say 'I don't have enough information' rather than guess. In our evaluation on 200 representative questions, 91% of answers were fully grounded in the source material. The 9% where it struggled were questions that required information we don't have in the corpus yet — not hallucinations, but knowledge gaps." I never promise zero hallucination — that's not achievable with current LLMs — but I can promise observable, auditable grounding with a measured failure rate. I also separate "the system is wrong" from "the system is uncertain and says so" — the latter is a feature, not a failure. For high-stakes domains (legal, medical), I'd recommend a human-in-the-loop review step for answers below a confidence threshold, and make that part of the product design from day one.

---

### Q5: "What are the trade-offs between using a managed vector database (Pinecone) vs. self-hosting (Qdrant) for a production RAG system?"

**A:** Managed (Pinecone) gives you zero operational burden, automatic scaling, and a serverless pricing model that's favorable at low-to-medium QPS. The downsides: you lose control over HNSW parameters (`m`, `ef_construction`, `ef_search`), which matters when you need to tune the precision-recall-latency trade-off for your specific query distribution; you're subject to the vendor's pricing model as you scale (can become expensive at high vector counts); and data residency/compliance constraints may be a blocker. Self-hosted Qdrant gives you full control over indexing parameters, on-prem deployment for compliance, and a cost structure that scales better at high volume (you pay for the compute, not per-vector). The operational burden is real: you own upgrades, backups, and failure recovery. My default for a greenfield project with a small team: start with Pinecone to move fast, instrument the query latency and recall, and migrate to self-hosted Qdrant when either cost or compliance forces the issue — by which point you have real query distribution data to tune the self-hosted configuration against.

> **Staff-Level Extension:** "How do you handle zero-downtime re-indexing when your embedding model changes?" Qdrant's versioned collections are the cleanest solution: create a new collection with the new model, run parallel ingestion, flip the read pointer atomically (or use a feature flag to route a canary percentage to the new index), validate eval metrics, then decommission the old collection. This is a standard blue-green deployment pattern applied to vector indexes.

---

### Q6: "You've shipped a RAG system that performs well in offline eval but gets poor user feedback in production. How do you diagnose this?"

**A:** Distribution shift between the golden evaluation set and real production queries is the most common cause. I'd start by pulling a sample of production query traces and running Ragas metrics on them — if offline eval scores are 0.85+ and production sample scores are 0.60, the golden dataset doesn't represent the real query distribution. Next, I'd cluster production queries (embed them, HDBSCAN cluster) and identify query types that are underrepresented in the golden set — these are the failure modes the eval harness wasn't covering. Common culprits: (1) multi-hop queries that require synthesizing across more than 2 documents — the system was only tested on single-hop; (2) queries about very recent documents that aren't yet indexed; (3) queries using domain jargon that differs from the indexing vocabulary, causing retrieval misses; (4) ambiguous queries where the top retrieved chunk is technically relevant but not the _most relevant_ — a reranker gap. The fix is a living golden dataset that is continuously seeded with production query samples, reviewed for correctness, and incorporated into the eval harness on a weekly cadence.

---

## E. Gotchas, Trade-offs & Best Practices

- **Chunk size is a retrieval-vs-synthesis trade-off, not a hyperparameter to tune blindly.** Small chunks (128–256 tokens) maximize retrieval precision (you retrieve exactly the relevant sentence) but degrade synthesis quality (the model lacks surrounding context). Large chunks (1024+ tokens) improve synthesis but dilute retrieval signal — the relevant sentence is buried in noise. The hierarchical parent-child pattern resolves this: retrieve on small child chunks, synthesize on large parent chunks. This is almost always the right default for prose documents.

- **Re-ranking is the highest-ROI retrieval upgrade.** A cross-encoder reranker applied to top-20 retrieved candidates and returning top-5 typically yields 10–15% improvement in context_precision with 30–100ms latency overhead — comparable ROI to upgrading the embedding model at a fraction of the cost. Many practitioners skip this step in favor of larger embedding models; they should do it in the reverse order.

- **Async ingestion pipelines silently fail without end-to-end reconciliation.** A Semaphore + asyncio pipeline looks correct but has a class of bugs where exceptions inside coroutines are swallowed if not explicitly awaited/gathered. Always use `asyncio.gather(*tasks, return_exceptions=True)` and inspect the results list for exception instances. Additionally, track ingestion completeness via a counter reconciled against the expected document count from the source — a 98% completion rate with no errors visible in logs is a common silent failure mode.

- **LLM-as-judge eval metrics are useful but biased toward fluency, not factuality.** Ragas `answer_relevance` uses an LLM to score answers, which means a fluent hallucination can score high. Always pair LLM-as-judge metrics with reference-based metrics (ROUGE, exact match on key entities) for high-stakes domains, and periodically audit a random sample of high-scoring answers manually. The "faithfulness" metric is the most trustworthy in the Ragas suite because it grounds evaluation against retrieved context rather than relying on the LLM's parametric knowledge.

- **Rate limit compliance is a systems problem, not just a code problem.** A single asyncio Semaphore is sufficient for one process, but production ingestion jobs often run in multiple parallel workers (Kubernetes pods, Celery workers). A process-local Semaphore does not enforce a global rate limit — you need a distributed rate limiter (Redis token bucket, or a centralized rate-limit proxy like an API gateway) to prevent 429 storms at scale. Design this into the ingestion architecture from the start; retrofitting a distributed rate limiter into an existing pipeline is painful.

---

## F. Code Pattern — Async Ingestion with Rate Limiting, Retry, and Vector Store Upsert

```python
"""
Production-grade async document ingestion pipeline.
Demonstrates: aiohttp fetching, Semaphore rate limiting, stamina retry,
batched embedding, and vector store upsert into Qdrant.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

import aiohttp
import stamina
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


COLLECTION_NAME = "codebase_v2"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
EMBED_BATCH_SIZE = 100
GITHUB_CONCURRENCY = 10   # Max concurrent GitHub API requests
EMBED_CONCURRENCY = 5     # Max concurrent embedding API calls


# ---------------------------------------------------------------------------
# Retry-decorated fetch — stamina wraps the coroutine with exponential backoff
# and emits structured log events on each retry attempt.
# ---------------------------------------------------------------------------
@stamina.retry(on=aiohttp.ClientError, attempts=5, wait_initial=1.0, wait_max=30.0)
async def fetch_document(
    session: aiohttp.ClientSession,
    url: str,
    sem: asyncio.Semaphore,
    headers: dict,
) -> bytes:
    async with sem:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            return await resp.read()


# ---------------------------------------------------------------------------
# GitHub document stream — yields raw Document objects.
# Handles Link-header pagination explicitly to avoid silent data loss.
# ---------------------------------------------------------------------------
async def stream_github_documents(
    repo_files: list[dict],
    github_token: str,
    sem: asyncio.Semaphore,
) -> AsyncIterator[Document]:
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3.raw",
    }
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_document(session, f["download_url"], sem, headers)
            for f in repo_files
        ]
        # return_exceptions=True ensures one failed fetch doesn't cancel the batch
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for file_meta, result in zip(repo_files, results):
        if isinstance(result, Exception):
            logger.error(
                "fetch_failed",
                path=file_meta["path"],
                error=str(result),
            )
            continue
        doc_id = hashlib.sha256(file_meta["path"].encode()).hexdigest()[:16]
        yield Document(
            id=doc_id,
            content=result.decode("utf-8", errors="replace"),
            metadata={
                "path": file_meta["path"],
                "repo": file_meta["repo"],
                "sha": file_meta["sha"],
            },
        )


def chunk_document(doc: Document, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    """
    Token-aware sliding window chunking.
    For production: replace with semantic chunking or AST-aware chunking
    depending on content type (prose vs. code).
    """
    words = doc.content.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        window = words[i : i + chunk_size]
        if not window:
            break
        chunk_text = " ".join(window)
        chunk_id = f"{doc.id}_{i}"
        chunks.append(
            Chunk(
                doc_id=doc.id,
                chunk_id=chunk_id,
                text=chunk_text,
                metadata={**doc.metadata, "chunk_index": i},
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Batched embedding with semaphore — respects RPM limits across concurrent callers.
# ---------------------------------------------------------------------------
@stamina.retry(on=Exception, attempts=4, wait_initial=2.0, wait_max=60.0)
async def embed_batch(
    client: AsyncOpenAI,
    texts: list[str],
    sem: asyncio.Semaphore,
) -> list[list[float]]:
    async with sem:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
    return [item.embedding for item in response.data]


async def embed_chunks_in_batches(
    chunks: list[Chunk],
    openai_client: AsyncOpenAI,
    embed_sem: asyncio.Semaphore,
) -> list[tuple[Chunk, list[float]]]:
    results: list[tuple[Chunk, list[float]]] = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        vectors = await embed_batch(openai_client, [c.text for c in batch], embed_sem)
        results.extend(zip(batch, vectors))
    return results


async def upsert_to_qdrant(
    qdrant: AsyncQdrantClient,
    embedded_chunks: list[tuple[Chunk, list[float]]],
) -> None:
    points = [
        PointStruct(
            id=abs(hash(chunk.chunk_id)) % (2**63),  # Qdrant requires uint64
            vector=vector,
            payload={
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                **chunk.metadata,
            },
        )
        for chunk, vector in embedded_chunks
    ]
    await qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("upserted", count=len(points))


# ---------------------------------------------------------------------------
# Orchestrator — wires all stages together with independent semaphores
# to enforce separate rate limits per external service.
# ---------------------------------------------------------------------------
async def ingest_repository(
    repo_files: list[dict],
    github_token: str,
    openai_api_key: str,
    qdrant_url: str,
) -> None:
    github_sem = asyncio.Semaphore(GITHUB_CONCURRENCY)
    embed_sem = asyncio.Semaphore(EMBED_CONCURRENCY)

    openai_client = AsyncOpenAI(api_key=openai_api_key)
    qdrant = AsyncQdrantClient(url=qdrant_url)

    # Ensure collection exists with correct vector config
    existing = {c.name for c in await qdrant.get_collections().then(lambda r: r.collections)}
    if COLLECTION_NAME not in existing:
        await qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    all_chunks: list[Chunk] = []
    async for doc in stream_github_documents(repo_files, github_token, github_sem):
        all_chunks.extend(chunk_document(doc))

    logger.info("chunking_complete", total_chunks=len(all_chunks))

    embedded = await embed_chunks_in_batches(all_chunks, openai_client, embed_sem)
    await upsert_to_qdrant(qdrant, embedded)

    logger.info(
        "ingestion_complete",
        documents=len(repo_files),
        chunks=len(all_chunks),
        vectors_upserted=len(embedded),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    sample_files = [
        {"path": "src/agent.py", "download_url": "https://...", "repo": "myorg/myrepo", "sha": "abc123"},
        {"path": "docs/architecture.md", "download_url": "https://...", "repo": "myorg/myrepo", "sha": "def456"},
    ]

    asyncio.run(
        ingest_repository(
            repo_files=sample_files,
            github_token=os.environ["GITHUB_TOKEN"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        )
    )
```

### Key Design Notes on the Code Pattern

- **Separate semaphores per external service**: GitHub and OpenAI have independent rate limits. A single shared semaphore would either over-throttle one or under-throttle the other. Model each rate limit independently.
- **`return_exceptions=True` in `asyncio.gather`**: Without this, one failed coroutine raises and cancels the entire gather. With it, you get a mixed list of results and exceptions, which you handle explicitly — the correct pattern for batch I/O where partial failures are expected.
- **`stamina` over bare `tenacity`**: stamina emits structured log events on every retry attempt (attempt number, wait duration, exception type) with zero configuration, which is invaluable for debugging production ingestion failures. It also integrates with OpenTelemetry spans if you're tracing ingestion jobs.
- **Hash-based document IDs**: Using `sha256(path)` as the document ID makes ingestion idempotent — re-ingesting the same path with the same content is a no-op at the vector store level (upsert semantics). Changing the commit SHA in metadata signals a content change that should trigger re-embedding.
- **Collection existence check before upsert**: In a production pipeline that may run in multiple parallel workers, the collection creation call should be idempotent (check-before-create or use `recreate_collection` with `if_not_exists=True` in Qdrant ≥ 1.8).

---

*Study guide generated for Session 12: Industry Use Cases & Capstone System Design. Covers the full AI Engineering project lifecycle, capstone agentic RAG architecture, async Python patterns, and staff-level interview preparation.*
