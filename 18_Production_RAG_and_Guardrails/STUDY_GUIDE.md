# Session 18: Production RAG, Guardrails & Caching
### Interview-Ready Study Guide — Senior / Staff AI Engineer

---

## A. Core Concept Summary

Moving a RAG system from prototype to production is not an incremental engineering task — it is a systems redesign. The happy path (clean queries, well-formed documents, compliant users) accounts for maybe 20% of real traffic; the other 80% includes adversarial inputs, schema-drifted documents, long-tail latency spikes, and users probing for jailbreaks or IP leakage. The mental model shift is from "does it work?" to "does it fail safely, predictably, and cheaply at scale?"

Three orthogonal concerns dominate production RAG architecture: **correctness** (guardrails ensure outputs are faithful, scoped, and policy-compliant), **cost** (caching collapses repeated or semantically similar LLM calls into sub-millisecond lookups), and **observability** (every request must produce a trace rich enough to debug failures post-hoc without replaying production traffic). A practitioner who holds all three simultaneously — and understands their interactions — is operating at staff level.

---

## B. Key Terms & Definitions

- **Guard (Guardrails AI):** A composable validation wrapper that applies one or more `Validator` objects to an LLM input or output. Guards are stateful objects that carry the validation schema and produce a `ValidationOutcome` with pass/fail status and optionally a corrected value.

- **Hub Guard:** A pre-built, community-contributed validator hosted at `hub.guardrailsai.com`. Instantiated with `hub.DetectJailbreak()` or similar; abstracts the detection logic behind a uniform `Validator` interface so you don't implement from scratch.

- **Input Guard vs. Output Guard:** Input guards validate the user prompt *before* the LLM call (blocking prompt injection, topic restriction, profanity). Output guards validate the model's response *after* generation (checking faithfulness, competitor mentions, hallucinations). They compose into a single `Guard` object but execute at different pipeline positions.

- **Exact-Match Cache:** A deterministic cache keyed on a hash (typically SHA-256) of `(prompt, model_id, temperature, max_tokens, ...)`. Returns a cached response only on bitwise-identical inputs. Highest precision, lowest recall; effective for repeated API calls like health checks or fixed-template queries.

- **Semantic Cache:** Embeds the incoming query and performs a nearest-neighbor search against a cache index. Returns a cached response if the cosine similarity exceeds a configured threshold (e.g., `≥ 0.92`). Higher recall than exact-match but introduces false-positive risk when semantically close queries have meaningfully different correct answers.

- **Cache Poisoning:** An attack (or accidental condition) where a malicious or incorrect response is written to the cache and subsequently served to other users. Particularly dangerous in semantic caches because a poisoned entry can match many future queries within its similarity radius.

- **RAG Faithfulness Guard:** A post-generation validator that cross-checks each claim in the generated answer against the retrieved context passages. Typically uses an entailment or NLI model, or a secondary LLM call, to score whether the answer is grounded. Catches hallucination that retrieval alone cannot prevent.

- **Prompt Injection:** A class of adversarial input where the user embeds instructions inside their query designed to override the system prompt or manipulate the agent's behavior. Particularly dangerous in agentic RAG where the LLM has tool-calling authority.

- **TTL (Time-to-Live):** A cache expiry policy that invalidates entries after a fixed duration. In LLM caching, TTL must be tuned to the update cadence of source documents and model version changes — not just to traffic patterns.

- **P99 Latency:** The 99th-percentile response time. In LLM systems, P99 is dominated by outlier token counts and guard chain overhead, making it a more actionable SLA metric than mean latency for production contracts.

---

## C. How It Works — Technical Mechanics

### Production RAG Pipeline — Layered Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│     INPUT GUARDRAIL LAYER   │  ← Topic restriction, jailbreak detection,
│  (pre-LLM validation)       │    profanity filter, PII detection
└────────────┬────────────────┘
             │ (blocked → return policy violation response)
             ▼
┌─────────────────────────────┐
│     SEMANTIC CACHE LOOKUP   │  ← Embed query → ANN search in cache index
│  (pre-retrieval)            │    (Redis / Qdrant / Chroma)
└────────────┬────────────────┘
             │ cache hit → skip retrieval + generation → CACHE HIT RESPONSE
             │ cache miss ↓
             ▼
┌─────────────────────────────┐
│     RETRIEVAL LAYER         │  ← Dense retrieval (ANN on vector DB)
│  (vector DB + reranker)     │    + optional sparse/hybrid, reranking
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│     GENERATION LAYER        │  ← LLM call with retrieved context injected
│  (LLM + prompt template)    │    into the system/user prompt
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│     OUTPUT GUARDRAIL LAYER  │  ← RAG faithfulness, competitor mention,
│  (post-LLM validation)      │    hallucination scoring, output profanity
└────────────┬────────────────┘
             │ failed → noop response / retry / corrected value
             ▼
┌─────────────────────────────┐
│     CACHE WRITE             │  ← Write (query_embedding, response) to
│  (async, post-response)     │    semantic cache with TTL
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│     OBSERVABILITY EMIT      │  ← LangSmith trace, guard violations,
│  (async side-channel)       │    latency, token count, cost metadata
└─────────────────────────────┘
             │
             ▼
        Final Response → User
```

### Guard Mechanics in Detail

**Topic Restriction** embeds both the user query and a set of forbidden topic descriptions. At runtime, it computes cosine similarity between the query embedding and each forbidden topic vector. If any similarity exceeds the configured threshold, the guard rejects the input. The embedding model used for restriction should match the one used for retrieval to avoid embedding space misalignment.

**Jailbreak Detection** uses a two-layer approach: (1) a fast regex/pattern layer that catches known jailbreak templates ("ignore previous instructions", "DAN mode", etc.) and (2) a slower LLM-based classifier for novel jailbreaks. The pattern layer acts as a pre-filter to avoid paying LLM costs on obvious violations.

**Competitor Mention** uses named entity recognition (NER) to identify organization names in the output, then checks against a configured deny-list of competitor names. More robust than regex because NER handles variations ("AWS" vs. "Amazon Web Services").

**RAG Faithfulness** extracts atomic claims from the generated response and, for each claim, checks whether it is entailed by any retrieved passage. An NLI model (e.g., a fine-tuned DeBERTa) returns `entailment`, `neutral`, or `contradiction`. Responses with a contradiction score above threshold or faithfulness below threshold are flagged. At high throughput this is expensive — a secondary LLM call per response is ~2–3× the generation cost — so many teams use it as a sampling guard (e.g., evaluate 10% of traffic) rather than inline per request.

### Caching — Mechanics and Trade-offs

| Strategy | Key | Backend | Hit Rate | False Positive Risk | Latency |
|---|---|---|---|---|---|
| Exact-match | SHA-256(prompt+params) | Redis / DynamoDB | Low | None | <1ms |
| Semantic | ANN(query_embedding) | Qdrant / Pinecone / Redis VSS | High | Medium-High | 5–20ms |
| Hybrid | Exact first, semantic fallback | Layered | Highest | Low-Medium | 1–20ms |

**Cache Invalidation** is genuinely hard in LLM systems. Unlike a database record, a cached LLM response can go stale for multiple independent reasons: (1) the source documents it was generated from were updated, (2) the underlying model was rotated (GPT-4o → GPT-4o-mini), (3) the system prompt changed, or (4) organizational policy changed. TTL-based invalidation is a blunt instrument — a 24-hour TTL may be correct for static knowledge bases and catastrophically wrong for real-time data products. The right approach is event-driven invalidation: when a document is re-ingested, compute the set of cached queries whose retrieved context included that document and tombstone them.

---

## D. Common Interview Questions (with Strong Answers)

---

**Q: How do you decide between exact-match and semantic caching, and what are the failure modes of each?**

A: The choice is fundamentally about the query distribution of your application. Exact-match caching is appropriate when your input space is constrained — templated queries, fixed UI flows, API health checks — because the collision rate under SHA-256 is effectively zero, meaning you never serve a wrong cached result. Semantic caching is appropriate when you have a large, varied user query space and you've measured empirically that query paraphrases have identical correct answers. The critical failure mode of semantic caching is threshold calibration: too low and you serve stale or wrong answers to semantically adjacent but meaningfully different queries ("what's the refund policy for orders over $100" hitting the cache for "what's the refund policy for orders under $50"); too high and your cache hit rate approaches zero. In production I set threshold conservatively (≥0.95 cosine) and A/B test cache hit rate vs. user satisfaction metrics to find the sweet spot. I also log every cache hit with the similarity score so I can audit for false positives post-hoc.

> **Staff-Level Extension:** How do you handle cache poisoning in a semantic cache — specifically, what happens when a malicious user crafts a query that writes a poisoned response which then matches many future benign queries?

The answer is to never write directly on the hot path. Cache writes should go through the same output guardrail pipeline before being committed. Additionally, write operations should be scoped by user/session isolation — cache entries from untrusted anonymous users should be quarantined and reviewed before being promoted to the shared cache namespace.

---

**Q: Walk me through how you'd wrap a LangGraph agent with Guardrails AI without breaking the streaming interface.**

A: LangGraph agents return either full responses or streamed token chunks. Guardrails AI's `guard.parse()` operates on complete strings, so naively wrapping streaming output means you can't validate until the stream is fully consumed — which defeats the UX purpose of streaming. The practical resolution is a two-phase approach: stream tokens to the UI for responsiveness, but hold back the "commit" (e.g., persisting to conversation history, triggering downstream actions) until the full output has been validated asynchronously. For truly latency-sensitive paths where even that is unacceptable, I move guardrails to asynchronous side-channel evaluation — every response is validated post-hoc, violations are flagged for human review, and the user's session is flagged for elevated monitoring. The synchronous inline guard is reserved for high-risk surfaces (e.g., the agent has tool-call authority that could exfiltrate data).

> **Staff-Level Extension:** What's the latency cost of a full guardrail chain in practice, and where do you put it in the SLA budget?

A full chain (jailbreak + topic + faithfulness) can add 200–800ms at P50 and 1–2s at P99 depending on whether faithfulness uses a local NLI model or a secondary LLM call. At P99 this is often the dominant cost, not the primary LLM call. I budget guardrails as a fixed overhead in the SLA and use model-tier selection — NLI-based for inline faithfulness, LLM-based for async deep audit — to keep the inline path under 300ms.

---

**Q: How do you handle context window overflow in a production RAG system under load, and what are the failure modes if you don't?**

A: Context window overflow happens when the retrieval system returns more chunks than the prompt template can accommodate, typically because top-k is set too aggressively or document chunk sizes weren't calibrated to the model's context budget. Without explicit handling, the LLM silently truncates input (right-truncation by default in most tokenizers), which means the system appears to work but is generating answers from incomplete context — often the worst failure mode because it's invisible to the user. My mitigation is a context budget enforcer in the prompt assembly layer: compute token counts for system prompt + retrieved chunks + generation budget, and truncate/rerank chunks to fit within `context_window - generation_budget - buffer`. I also set retrieval top-k conservatively (k=5 typically rather than k=20) and use a reranker to ensure the highest-quality chunks survive truncation rather than just the first-retrieved.

> **Staff-Level Extension:** How does this interact with your caching strategy?

Cached responses were generated from a specific retrieved context at a specific top-k. If the retrieval or chunking strategy changes, cached responses may be stale in a non-obvious way — the answer is technically correct for the old document set but wrong for the updated one. This is another argument for document-event-driven cache invalidation rather than pure TTL.

---

**Q: What are the security properties of input guardrails, and why are they not a complete defense against prompt injection?**

A: Input guardrails operating on the raw user query can catch known patterns and topic violations, but they fundamentally cannot defend against prompt injection that is embedded in *retrieved documents* rather than in the user query. If a user controls or influences source documents (e.g., a public web page that gets indexed), they can embed instructions in the document content that are then injected into the LLM's context at retrieval time. This is sometimes called indirect prompt injection and bypasses all input-layer guards entirely. A more complete defense requires output-layer behavioral monitoring — checking whether the LLM's response shows signs of instruction following that weren't in the user's query — and architectural isolation (the LLM should operate in a permission-minimal environment where even a successfully injected instruction cannot exfiltrate data or trigger dangerous tool calls).

---

**Q: How do you design an observability stack for a production RAG system, and what metrics are actually actionable?**

A: I instrument at three layers. At the request layer: end-to-end latency, token counts (input + output), cost-per-request (critical for budget forecasting), and guard outcome (pass/fail/corrected). At the retrieval layer: retrieved chunk relevance scores, top-k hit rate, embedding model latency. At the generation layer: LangSmith traces for every request, tagged with user cohort, query intent (if classifiable), and cache hit/miss status. The actionable metrics are: P99 latency by pipeline stage (tells you where to optimize), guard violation rate by type (a spike in jailbreak violations often precedes a security incident), cache hit rate (directly maps to cost reduction), and faithfulness score distribution (leading indicator for hallucination regression when you rotate models or update retrieval). Mean latency and raw accuracy metrics are vanity metrics at staff level — the distributions and correlations are what matter.

---

**Q: How do you test guardrails in CI/CD without relying on the production LLM?**

A: Guardrail tests should be hermetic — no live LLM calls in the fast CI loop. I maintain an adversarial test suite: a curated set of (input, expected_guard_outcome) pairs that cover known jailbreak patterns, edge cases from past production incidents, and boundary conditions around threshold values. Unit tests mock the LLM call and validate that the guard chain produces the expected `ValidationOutcome`. Integration tests against a local LLM (Ollama, vLLM) run in a slower nightly pipeline. I also track guard precision/recall over the adversarial suite as a metric in the CI report — a drop in recall (more jailbreaks passing) or a spike in false positives (legit queries being blocked) is a deploy gate. The adversarial suite is a living artifact owned by the security team and updated whenever a new bypass is discovered in production.

---

## E. Gotchas, Trade-offs & Best Practices

- **Threshold miscalibration in semantic caches is a silent correctness bug.** A similarity threshold that's slightly too low doesn't cause visible errors — it causes subtly wrong answers that are hard to detect without a faithfulness guardrail. Always instrument cache hits with similarity scores and run periodic audits against a sample of cached responses. In domains with high consequence (medical, legal, financial), the threshold should be validated domain-specifically, not just with general embedding benchmarks.

- **Guardrails AI's hub validators are convenient but not calibrated for your domain.** A hub jailbreak detector trained on generic adversarial prompts may have high false-positive rates on legitimate technical queries in your domain (e.g., security research, red teaming tooling). Always benchmark hub validators on a representative sample of your production query distribution before deploying. Fine-tuned custom validators are almost always necessary for high-stakes production systems.

- **The faithfulness guard and the semantic cache interact in a non-obvious way.** If you write to the semantic cache *before* output guardrails run (e.g., for performance), you can cache an unfaithful response and serve it to all future similar queries. The cache write *must* occur after all output guards pass. This adds latency to the cache write path but is non-negotiable for correctness.

- **Docker image size for AI apps bloats fast.** A naive single-stage build of a LangGraph + Guardrails + embedding model app can easily exceed 8GB due to model weights bundled in the image. Use multi-stage builds to separate the dependency installation stage from the runtime stage, mount model weights as a volume or pull from object storage at startup, and use `python:3.11-slim` as the base. Keep the deployable image under 2GB; weights in S3/GCS are cheaper and more maintainable than fat images in ECR.

- **Async guardrail evaluation is not a silver bullet for latency.** Moving guardrails off the critical path improves user-facing P99 but means users can receive policy-violating responses before the violation is detected. This is acceptable for content policy (output profanity) where the consequence of a single violation is low, but unacceptable for data exfiltration scenarios (PII leakage, IP output). The sync/async decision for each guard type must be an explicit risk-based engineering decision, not a default performance optimization.

---

## F. Code & Architecture Pattern

### Wrapping a LangChain Chain with Guardrails AI (Input + Output) + Semantic Cache

```python
"""
Production RAG pipeline with:
- Input guardrails (jailbreak detection, topic restriction)
- Output guardrails (RAG faithfulness, competitor mention)
- Semantic cache layer (Redis + vector similarity)

Requires:
  pip install guardrails-ai langchain-openai redis langchain-community
  guardrails hub install hub://guardrails/detect_jailbreak
  guardrails hub install hub://guardrails/competitor_check
  guardrails hub install hub://guardrails/toxic_language
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

import numpy as np
import redis
from guardrails import Guard
from guardrails.hub import CompetitorCheck, DetectJailbreak, ToxicLanguage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard definitions
# ---------------------------------------------------------------------------

INPUT_GUARD = Guard().use_many(
    DetectJailbreak(on_fail="exception"),
    ToxicLanguage(threshold=0.7, validation_method="sentence", on_fail="exception"),
)

OUTPUT_GUARD = Guard().use_many(
    CompetitorCheck(
        competitors=["CompetitorA", "CompetitorB"],
        on_fail="fix",          # redact competitor names rather than hard-fail
    ),
    ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="exception"),
)


# ---------------------------------------------------------------------------
# Semantic cache backed by Redis (stores embeddings + responses)
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Embedding-based cache. Stores (query_embedding, response) pairs in Redis.
    Uses brute-force cosine similarity for simplicity; swap for Redis VSS /
    Qdrant at production scale (>100k cached entries).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        similarity_threshold: float = 0.94,   # conservative; tune per domain
        ttl_seconds: int = 86_400,             # 24h — tune to source update cadence
    ):
        self.client = redis.from_url(redis_url, decode_responses=False)
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self._cache_key_prefix = "semcache:"

    def _embed(self, text: str) -> np.ndarray:
        return np.array(self.embedder.embed_query(text), dtype=np.float32)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def get(self, query: str) -> Optional[str]:
        query_vec = self._embed(query)
        # Scan all cache entries — replace with ANN at scale
        for key in self.client.scan_iter(f"{self._cache_key_prefix}*"):
            entry = json.loads(self.client.get(key))
            cached_vec = np.array(entry["embedding"], dtype=np.float32)
            similarity = self._cosine(query_vec, cached_vec)
            if similarity >= self.threshold:
                logger.info(
                    "semantic_cache_hit",
                    extra={"similarity": round(similarity, 4), "key": key},
                )
                return entry["response"]
        return None

    def set(self, query: str, response: str) -> None:
        query_vec = self._embed(query)
        cache_key = f"{self._cache_key_prefix}{hashlib.sha256(query.encode()).hexdigest()}"
        payload = json.dumps({"embedding": query_vec.tolist(), "response": response})
        self.client.setex(cache_key, self.ttl, payload)

    def invalidate_by_prefix(self, prefix: str) -> int:
        """
        Event-driven invalidation: call when source documents update.
        In production, prefix would encode the document namespace.
        """
        keys = list(self.client.scan_iter(f"{self._cache_key_prefix}{prefix}*"))
        if keys:
            self.client.delete(*keys)
        return len(keys)


# ---------------------------------------------------------------------------
# RAG chain with integrated guards + semantic cache
# ---------------------------------------------------------------------------

class ProductionRAGChain:
    """
    Wraps a LangChain retrieval chain with:
    1. Input guardrails (pre-LLM)
    2. Semantic cache lookup (pre-LLM, post-input-guard)
    3. LLM generation
    4. Output guardrails (post-LLM)
    5. Cache write (post-output-guard — critical ordering)

    Guard failures raise GuardrailsValidationError; callers should catch and
    return a policy violation response to the user.
    """

    def __init__(self, retriever, llm: ChatOpenAI, cache: SemanticCache):
        self.retriever = retriever
        self.cache = cache

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a helpful assistant. Answer based only on the provided context. "
                "If the context does not contain enough information, say so explicitly."
            )),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])

        self._chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, query: str) -> dict:
        # ── 1. Input guardrails ──────────────────────────────────────────────
        validation = INPUT_GUARD.validate(query)
        # Guard raises on_fail="exception" validators; on_fail="fix" returns corrected value
        validated_query: str = validation.validated_output or query

        # ── 2. Semantic cache lookup ─────────────────────────────────────────
        cached = self.cache.get(validated_query)
        if cached is not None:
            return {"response": cached, "source": "cache"}

        # ── 3. LLM generation ────────────────────────────────────────────────
        raw_response: str = self._chain.invoke(validated_query)

        # ── 4. Output guardrails ─────────────────────────────────────────────
        out_validation = OUTPUT_GUARD.validate(raw_response)
        final_response: str = out_validation.validated_output or raw_response

        # ── 5. Cache write (only after guards pass) ──────────────────────────
        self.cache.set(validated_query, final_response)

        return {"response": final_response, "source": "llm"}


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from langchain_community.vectorstores import Chroma

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    cache = SemanticCache(similarity_threshold=0.94, ttl_seconds=86_400)

    rag = ProductionRAGChain(retriever=retriever, llm=llm, cache=cache)

    try:
        result = rag.invoke("What is our return policy for international orders?")
        print(f"[{result['source']}] {result['response']}")
    except Exception as e:
        # Guard violation — return policy message to user, log for audit
        logger.warning("guard_violation", extra={"error": str(e)})
        print("I'm unable to process that request.")
```

### LangGraph Agent Integration Pattern

```python
"""
Wrapping a LangGraph agent node with guardrails.
The guard is applied at the node boundary, not inside the LLM call,
so it composes cleanly with the graph's state management.
"""

from guardrails.errors import ValidationError as GuardrailsValidationError
from langgraph.graph import END, StateGraph
from typing import TypedDict


class AgentState(TypedDict):
    query: str
    response: str
    guard_violation: bool


def input_guard_node(state: AgentState) -> AgentState:
    """Pre-LLM node: validate and potentially transform the user query."""
    try:
        result = INPUT_GUARD.validate(state["query"])
        return {**state, "query": result.validated_output or state["query"]}
    except GuardrailsValidationError as e:
        logger.warning("input_guard_violation", extra={"reason": str(e)})
        return {**state, "guard_violation": True, "response": "Policy violation detected."}


def output_guard_node(state: AgentState) -> AgentState:
    """Post-LLM node: validate and optionally correct the agent response."""
    try:
        result = OUTPUT_GUARD.validate(state["response"])
        return {**state, "response": result.validated_output or state["response"]}
    except GuardrailsValidationError as e:
        logger.warning("output_guard_violation", extra={"reason": str(e)})
        return {**state, "guard_violation": True, "response": "Response failed policy check."}


def should_continue(state: AgentState) -> str:
    return END if state.get("guard_violation") else "llm_node"


graph = StateGraph(AgentState)
graph.add_node("input_guard", input_guard_node)
graph.add_node("llm_node", lambda s: s)   # replace with actual LLM node
graph.add_node("output_guard", output_guard_node)

graph.set_entry_point("input_guard")
graph.add_conditional_edges("input_guard", should_continue)
graph.add_edge("llm_node", "output_guard")
graph.add_edge("output_guard", END)

app = graph.compile()
```

### Dockerfile for a LangGraph RAG Service (Multi-Stage)

```dockerfile
# ── Stage 1: dependency builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security hardening
RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

# Model weights are NOT baked in — pulled from object storage at startup
# or mounted as a volume. Keeps image under 2GB.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Secrets injected via environment — never baked into image
# OPENAI_API_KEY, LANGCHAIN_API_KEY, REDIS_URL set in k8s secret / ECS task def

USER appuser
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
```

---

*Study guide generated for AIE9 Session 18 — Production RAG, Guardrails & Caching.*
