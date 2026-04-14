# Session 11: Advanced Retrieval Strategies — Interview-Ready Study Guide

> **Audience:** Senior / Staff AI Engineer candidates with a background in production search, recommendation systems, and information retrieval. Parallels to Elasticsearch, FAISS/ScaNN, two-tower models, and ANN search are drawn throughout.

---

## A. Core Concept Summary

Naive top-k dense retrieval — embed a query, ANN-search a vector index, return the top-k hits — is the "Hello World" of RAG. In production it fails in predictable, well-understood ways: vocabulary mismatch when the query uses exact terminology the embedding model never saw, domain gaps when a general-purpose encoder is applied to specialized corpora, and brittleness on long-tail or high-precision queries where a single character difference changes the answer. The practitioner's mental model must shift from *single-stage retrieval* to a *retrieval pipeline*: a first-stage that maximises recall across multiple modalities (sparse + dense), followed by second-stage reranking that trades latency for precision. This is identical in spirit to the two-stage architecture used in production recommendation and web search systems — candidate generation at scale, then a slower, higher-capacity scorer over a small candidate set. Evaluation must be decoupled from generation; measuring context precision and recall independently is the only way to attribute quality improvements to the retrieval layer rather than the LLM.

---

## B. Key Terms & Definitions

- **BM25 (Best Match 25):** A probabilistic sparse retrieval function that extends TF-IDF with term saturation (parameter `k1`) and document-length normalization (parameter `b`). It scores documents based on lexical overlap, making it highly effective for exact-match and keyword-heavy queries.

- **Reciprocal Rank Fusion (RRF):** A rank-aggregation method that combines ranked lists from multiple retrievers using the formula `score = 1 / (k + rank)`. It is robust to absolute score scale differences because it operates only on ordinal rank positions, not raw scores.

- **Hybrid Search:** A retrieval strategy that merges results from at least one sparse retriever (BM25) and one dense retriever (vector ANN) to exploit the complementary strengths of lexical precision and semantic coverage.

- **Multi-Query Retrieval:** An LLM-powered technique that expands a single user query into N semantically diverse reformulations, executes all N retrievals independently, and deduplicates the union of results. It improves recall on ambiguous or under-specified queries.

- **Parent Document Retriever:** A two-tier chunking strategy where small child chunks are embedded and indexed for precise retrieval scoring, but the larger parent document they belong to is returned for generation context. It decouples indexing granularity from generation context size.

- **Semantic Chunking:** An embedding-driven method of splitting text at natural semantic boundaries — typically detected by a spike in cosine distance between adjacent sentence embeddings — rather than at fixed token counts. LangChain's `SemanticChunker` implements this.

- **RAG-Fusion:** A pipeline that combines multi-query expansion with RRF: N query variants are generated, N ranked lists are retrieved, and RRF fuses them into a single ranked list before generation. The full pipeline compounds the recall gains of both techniques.

- **Cross-Encoder Reranker:** A transformer model that takes a (query, document) pair as a single input and outputs a relevance score. Unlike bi-encoders (two-tower), cross-encoders model fine-grained query–document interaction, yielding much higher precision at the cost of O(n) inference over the candidate set.

- **Context Precision@k:** A retrieval evaluation metric measuring what fraction of the top-k retrieved chunks are actually relevant. Analogous to precision@k in classical IR; it penalizes noise in the context window.

- **Context Recall:** A retrieval evaluation metric measuring what fraction of the ground-truth relevant information is covered by the retrieved set. Analogous to recall in IR; it captures retrieval completeness independently of LLM answer quality.

---

## C. How It Works — Technical Mechanics

### BM25: Mechanics Beyond TF-IDF

BM25 scores a document `d` for query `q` as:

```
BM25(d, q) = Σ_t IDF(t) · [ tf(t,d) · (k1 + 1) ] / [ tf(t,d) + k1 · (1 - b + b · |d|/avgdl) ]
```

- **`k1` (term frequency saturation):** Controls how quickly additional occurrences of a term stop contributing. `k1 ∈ [1.2, 2.0]` is the standard range. As `k1 → ∞`, BM25 degenerates toward raw TF. As `k1 → 0`, TF is ignored entirely (pure IDF). For technical documentation where term repetition matters, push `k1` higher; for short snippets, lower it.
- **`b` (length normalization):** `b = 1.0` fully penalizes long documents; `b = 0.0` ignores length. The sweet spot for most corpora is `b ≈ 0.75`. If your corpus has wildly variable document lengths (API docs mixed with white-papers), tuning `b` is non-negotiable.
- The key difference from TF-IDF: BM25's saturation curve means the marginal gain of the 10th occurrence of a term is much smaller than the 2nd — exactly the behaviour you want for noisy repetitive documents.

### Why Dense Retrieval Fails at Scale

Dense retrieval failure modes map directly to production search failure modes:

| Failure Mode | Dense Retrieval | Equivalent Search Failure |
|---|---|---|
| Vocabulary mismatch | OOV terms not in embedding space | Missing inverted index entry |
| Domain gap | General encoder on specialized corpus | Relevance model trained on wrong domain |
| Long-tail queries | Low-frequency patterns underrepresented in training data | Cold-start in collaborative filtering |
| Exact-match needs | Paraphrases cluster near; identical strings may not be nearest neighbor | BM25 term boost not modeled |

### Score Normalization Problem in Hybrid Search

BM25 scores are unbounded and corpus-dependent; cosine similarities are bounded `[-1, 1]`. Naively summing them gives BM25 total dominance for anything but small, clean corpora. Options:

1. **Min-max normalization per query** — sensitive to outliers; breaks when the score distribution shifts.
2. **Z-score normalization** — better but requires computing statistics over the candidate set, adding latency.
3. **RRF (no score normalization needed)** — convert to ranks first, fuse ranks. This is why RRF dominates in practice.

### RRF Formula

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```

`k = 60` is the canonical default (from the original Cormack et al. paper). The `k` constant prevents a rank-1 document from dominating — it acts as a floor that dampens the advantage of top-ranked documents. When a document appears in multiple ranked lists, its scores accumulate additively. Documents absent from a list get no contribution from that retriever.

**RRF is essentially a Borda count variant with a smoothing constant.** For engineers familiar with ensemble ranking in recommender systems, this is exactly a rank-based ensemble without the distributional assumptions of score-based fusion.

### Multi-Query Retrieval: When It Helps and When It Hurts

**Helps:**
- Ambiguous queries where the user intent could be interpreted multiple ways.
- Queries with entity aliases (e.g., "ML" vs. "machine learning" vs. "artificial intelligence").
- Queries where the user's vocabulary differs from the corpus vocabulary.

**Hurts:**
- High-precision factual queries where the first retrieval is already correct — N extra LLM calls add latency with zero recall gain.
- Cost-sensitive deployments: N queries = N × (embedding cost + ANN query cost) + 1 LLM call.
- Short context window models: deduplication may still leave you with 3N–5N candidates, overwhelming the context window.

### Parent Document Retriever: Two-Tier Store Design

```
                ┌─────────────────────────────────────┐
                │            Full Documents            │
                │         (InMemoryStore / Redis)      │
                └──────────────────┬──────────────────┘
                                   │  parent_id pointer
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
        [child chunk 1]    [child chunk 2]       [child chunk 3]
        (128 tokens)       (128 tokens)          (128 tokens)
         embedded +         embedded +            embedded +
         indexed in         indexed in            indexed in
         vector store       vector store          vector store
```

Child chunk embeddings are dense and precise — they represent a narrow semantic slice, which improves recall for specific questions. But the returned context for the LLM is the full parent, preserving discourse coherence. This directly mirrors how production IR systems return full documents after scoring on passage-level signals.

### Semantic Chunking: Mechanics

LangChain's `SemanticChunker` computes embeddings for each sentence, then measures cosine distance between adjacent sentence pairs. A chunk boundary is inserted wherever the distance exceeds a percentile threshold (configurable: percentile-based, standard-deviation-based, or interquartile-range-based). The result is chunks that respect topical coherence at the cost of variable chunk sizes and unpredictable memory/latency behavior at inference time.

**vs. Fixed-Size Chunking:**
- Fixed-size is predictable, fast, and easy to batch; semantic chunking is content-adaptive but slow (requires N embedding calls during ingestion) and can produce degenerate chunk distributions on repetitive or poorly-structured documents.
- For structured corpora (API docs, FAQs), semantic chunking adds cost with minimal benefit. For long-form heterogeneous content (research papers, legal documents), it consistently improves recall.

### Reranking: Cross-Encoder Mechanics

A cross-encoder takes `[CLS] query [SEP] document [SEP]` as input and produces a scalar relevance score. The key difference from a bi-encoder: attention layers can model direct query-term–document-term interactions, catching non-compositional relevance signals that distance metrics miss.

**Pipeline placement:**
```
Query → First-stage retrieval (BM25 + dense, top-100) → Reranker (cross-encoder, top-100 → top-5) → LLM generation
```

The first stage must recall broadly (target recall@100 ≥ 0.95); the reranker then precision-filters. If your first-stage recall is poor, reranking cannot recover documents that were never retrieved.

**Latency budget:** Cohere Rerank API adds ~200–400ms for 100 candidates; a locally-run BGE reranker on a T4 GPU adds ~50–150ms. For real-time RAG, this is often acceptable; for sub-100ms SLA systems, you need to be ruthless about candidate count or cache reranker scores.

### Decision Tree: Retrieval Strategy Selection

```
Query arrives
    │
    ├─► Exact-match / keyword / entity lookup?
    │       └─► BM25 first, dense as fallback
    │
    ├─► Semantic / conceptual question?
    │       └─► Dense retrieval; add BM25 if recall is insufficient
    │
    ├─► Ambiguous or multi-intent?
    │       └─► Multi-query retrieval → RRF fusion
    │
    ├─► Long-tail / rare topic?
    │       └─► Hybrid (BM25 + dense) → reranker
    │
    ├─► High-precision required (narrow factual)?
    │       └─► Dense → cross-encoder rerank
    │
    └─► Broad research / synthesis task?
            └─► RAG-Fusion (multi-query + RRF) → reranker
```

---

## D. Common Interview Questions (with Strong Answers)

---

**Q1: You've deployed a RAG system with dense retrieval and users are reporting that it misses results for exact product names and SKU codes. How do you diagnose and fix this?**

**A:** This is a textbook vocabulary mismatch failure — exact identifiers like SKUs rarely appear in the pretraining data of general-purpose embedding models, so their representations are not reliably clustered in the embedding space. The fix is to introduce a BM25 retriever running in parallel over the same corpus, fuse the ranked lists with RRF, and measure context recall on a golden set of SKU-heavy queries before and after. In my experience, hybrid search with RRF on this class of queries produces a recall lift of 15–30% over dense-only without any embedding retraining. If the SKU queries are high-frequency, I'd also consider a dedicated exact-match lookup layer (Elasticsearch term query) upstream of both retrievers as a short-circuit — this mirrors how production e-commerce search systems handle product ID lookups.

**Staff-Level Extension:** What if the SKU corpus updates in real-time (new products added hourly)? Now you have a freshness problem: the dense index requires re-embedding and re-indexing, while BM25 only needs an inverted index update. This asymmetry is a strong argument for maintaining BM25 as the primary retriever for structured metadata and the dense index for semantic content — a separation of concerns that also simplifies partial index updates.

---

**Q2: Walk me through the RRF formula and explain why `k=60` is the default. When would you change it?**

**A:** RRF scores a document as the sum of `1 / (k + rank)` across all ranked lists. The `k` constant acts as a smoothing term: without it, the document ranked first by any single retriever would receive a score of `1/1 = 1.0`, which is arbitrarily large relative to rank-2 documents. With `k=60`, the rank-1 document scores `1/61 ≈ 0.0164` versus rank-2's `1/62 ≈ 0.0161` — a much smaller gap, making the fusion more democratic across retrievers. Cormack et al. set `k=60` empirically across TREC benchmarks; it's a sensible default but not sacred. I'd lower `k` (e.g., `k=10`) when I want to amplify the signal from a high-confidence retriever — for instance, when the dense retriever is fine-tuned on-domain and I trust its top-1 result more than the BM25 top-1. I'd raise `k` (e.g., `k=100`) when I want uniform treatment of all retrieved candidates, effectively reducing the variance contribution from rank position.

**Staff-Level Extension:** RRF assumes all retrievers are equally informative. In a system where retriever A has recall@10 of 0.9 and retriever B has recall@10 of 0.6, uniform fusion suboptimally weights B. A learned rank aggregation model (e.g., a linear combination of RRF scores calibrated via grid search on a validation set, or a learned sparse combiner) can outperform vanilla RRF when you have enough labeled data to tune it.

---

**Q3: Describe the failure modes of multi-query retrieval and how you'd mitigate them.**

**A:** The three core failure modes are: (1) **Query drift** — the LLM generates reformulations that shift the semantic intent, retrieving topically adjacent but irrelevant content that poisons the context window; (2) **Cost amplification** — N queries means N embedding calls and N ANN searches; at GPT-4 token prices, generating 5 reformulations can 3–4x the per-query cost; (3) **Context overload** — post-deduplication you may have 4N chunks competing for a fixed context window, which can actually degrade answer quality compared to a focused top-5. Mitigations: constrain the reformulation prompt to preserve the core entity/intent; set a hard cap on N (3 is usually sufficient); apply a reranker over the union to force a ranked truncation before feeding the LLM. I'd also A/B test multi-query against single-query + reranker on production traffic, as the latter often achieves comparable precision with lower cost.

**Staff-Level Extension:** How do you detect query drift in production? You can embed all N reformulations and compute pairwise cosine distances; if the centroid distance from the original query exceeds a threshold, flag the expansion as potentially drifted and fall back to single-query retrieval. This is analogous to query diversification detection in recommendation systems.

---

**Q4: How do you evaluate retrieval quality independently of generation quality, and why does the distinction matter for production debugging?**

**A:** The key metrics are context precision@k (fraction of retrieved chunks that are relevant) and context recall (fraction of ground-truth relevant content that was retrieved). In Ragas, these are computed against a golden dataset of (question, ground_truth_contexts, ground_truth_answer) triples — crucially, the retrieval metrics are evaluated before the LLM ever sees the context. The distinction matters enormously for production debugging: if the LLM answer quality degrades after a corpus update, you need to know whether recall dropped (the right chunks aren't being retrieved) or whether precision dropped (noisy chunks are confusing the LLM). Without this separation you're stuck tuning prompt engineering for a retrieval problem, or vice versa. In practice I maintain separate evaluation pipelines that checkpoint retrieved contexts and LLM outputs independently, with dashboards for both. This mirrors how production recommendation systems separate click-through rate (generation quality) from recall@N (retrieval quality).

**Staff-Level Extension:** Context recall requires knowing what the ground-truth contexts are, which is expensive to label. Synthetic golden dataset generation using an LLM (generate questions from known documents, use those documents as ground-truth contexts) is the standard shortcut — but it introduces a labeling bias where the LLM's generation biases align with its retrieval biases. How do you detect and correct for this?

---

**Q5: The Parent Document Retriever solves the chunk-size dilemma, but it introduces its own failure modes. What are they?**

**A:** The two primary failure modes are: (1) **Parent boundary misalignment** — if the parent chunks are too large (e.g., full 10-page documents), the LLM receives massive context where the relevant content is a small fraction, degrading answer faithfulness and increasing cost. The child-to-parent ratio needs to be tuned empirically for the corpus; I typically start at a 4:1 ratio (128-token children, 512-token parents) and measure context precision at both granularities. (2) **Cross-parent aggregation** — for synthesis questions that require content from multiple parent chunks, the two-tier store returns multiple full parents, potentially exhausting the context window. This is structurally similar to the passage retrieval vs. document retrieval tradeoff in classical IR. A hybrid approach — return the most relevant parent plus child-level snippets from other parents — can reduce this pressure. There's also an operational issue: the parent store (usually `InMemoryStore`) is not persistent by default in LangChain, so production deployments must wire it to a durable store (Redis, DynamoDB) with appropriate TTL and invalidation logic.

---

**Q6: Where in the retrieval pipeline would you place a cross-encoder reranker, and what are the latency/quality trade-offs vs. the ColBERT-style late interaction model?**

**A:** The reranker sits at the junction between first-stage retrieval and generation — it scores all first-stage candidates but only the top-k reranked results proceed to the LLM. Placement decision: the first stage must over-retrieve (top-50 to top-100) to ensure high recall, since the reranker cannot recover documents that were never retrieved. Cross-encoders (Cohere Rerank, BGE reranker) achieve high precision because they model direct query–document token interactions, but at O(n · sequence_length²) cost — 100 candidates at 512 tokens each on a T4 is ~150ms per query. ColBERT-style late interaction (pre-compute document token embeddings, compute MaxSim at query time) is a middle ground: it can be applied at first-stage scale (tens of thousands of candidates) with much lower latency than a cross-encoder, but typically lower precision than a full cross-encoder. For a production RAG system with a 500ms end-to-end SLA, I'd use a cross-encoder on ≤50 candidates and accept the latency; for a sub-200ms SLA, I'd use ColBERT at first-stage or skip the reranker entirely and invest in better first-stage recall via hybrid search.

---

## E. Gotchas, Trade-offs & Best Practices

- **RRF is not a free lunch for heterogeneous retriever quality.** If one retriever has consistently higher recall than the other (e.g., a fine-tuned dense retriever vs. a vanilla BM25 on a specialized domain), equal-weight RRF dilutes the signal from the better retriever. Validate retriever quality individually before fusing; if one consistently underperforms, consider weighting its contribution or dropping it.

- **Semantic chunking at ingestion is a hidden latency bottleneck.** Calling an embedding model for every sentence during ingestion can be 10–50x slower than fixed-size splitting on large corpora. If you're processing millions of documents, this needs to run asynchronously and at high concurrency. Also, the chunk size distribution with semantic chunking can be bimodal (many tiny chunks for structured text, a few massive chunks for dense prose), which breaks assumptions in your vector index and context window budget.

- **Multi-query retrieval degrades under tight context window budgets.** After deduplication you may still have 3–5x more candidates than a single-query retrieval. Without a downstream reranker to truncate aggressively, you risk filling the context window with marginally relevant content that reduces answer faithfulness on precise factual questions. Always pair multi-query with a reranker or a hard candidate cap.

- **First-stage recall is the ceiling; reranking only rearranges what you already have.** Engineers commonly over-invest in reranker quality while neglecting first-stage retrieval. A reranker operating on a recall@50 of 0.70 cannot outperform a simpler reranker operating on a recall@50 of 0.95. Instrument and alert on first-stage recall before optimizing the second stage.

- **In-memory parent stores in LangChain are silent production killers.** The default `InMemoryDocstore` for the Parent Document Retriever is not serializable, not persistent, and not shared across process replicas. In a multi-replica deployment, each replica rebuilds its store independently, meaning retrieval results are non-deterministic across requests. Production deployments must use Redis or a persistent key-value store with a documented serialization format and TTL policy.

---

## F. Code & Architecture Patterns

### Pattern 1: Hybrid BM25 + Dense Ensemble Retriever with RRF

```python
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from typing import List
from collections import defaultdict

# --- First-stage retrievers ---
docs: List[Document] = [...]  # your corpus

bm25_retriever = BM25Retriever.from_documents(docs, k=20)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embedding=embedding_model)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# --- Option A: LangChain EnsembleRetriever (weighted score fusion) ---
# Note: EnsembleRetriever uses RRF internally when weights are equal
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5],  # equal weight; tune based on retriever quality
)

# --- Option B: Custom RRF retriever (explicit, tunable k) ---
class RRFFusionRetriever(BaseRetriever):
    retrievers: list = Field(default_factory=list)
    k: int = 60
    top_n: int = 10

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Collect ranked lists from each retriever
        ranked_lists = [r.get_relevant_documents(query) for r in self.retrievers]

        # Compute RRF scores: score(d) = sum over lists of 1 / (k + rank)
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                doc_id = doc.metadata.get("id") or doc.page_content[:64]
                rrf_scores[doc_id] += 1.0 / (self.k + rank)
                doc_map[doc_id] = doc

        # Sort by RRF score descending, return top_n
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids[: self.top_n]]

rrf_retriever = RRFFusionRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    k=60,
    top_n=10,
)
```

---

### Pattern 2: Parent Document Retriever with Persistent Store

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore  # swap for RedisStore in prod
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Child splitter: small, precise — optimized for embedding recall
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=128,
    chunk_overlap=20,
)

# Parent splitter: large, contextual — optimized for LLM coherence
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
)

# Vector store holds child chunk embeddings only
child_vectorstore = Chroma(
    collection_name="child_chunks",
    embedding_function=embedding_model,
)

# Docstore holds parent documents — use RedisStore in production:
# from langchain_community.storage import RedisStore
# parent_store = RedisStore(redis_url="redis://localhost:6379", ttl=86400)
parent_store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore,
    docstore=parent_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 10},  # retrieve top-10 child chunks, return their parents
)

docs: list = [...]  # your source documents
retriever.add_documents(docs, ids=None)  # auto-generates parent IDs

# At query time: scores over child chunks, returns parent documents
results = retriever.get_relevant_documents("What is the refund policy?")
```

---

### Pattern 3: RAG-Fusion Pipeline (Multi-Query + RRF)

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

query_expansion_prompt = ChatPromptTemplate.from_template(
    """You are an expert at query reformulation for document retrieval.
    Generate {n} diverse reformulations of the following question.
    Preserve the core intent. Output one query per line, no numbering.

    Original question: {question}"""
)

def expand_queries(inputs: dict, n: int = 3) -> list[str]:
    chain = query_expansion_prompt | llm | StrOutputParser()
    raw = chain.invoke({"question": inputs["question"], "n": n})
    variants = [q.strip() for q in raw.strip().split("\n") if q.strip()]
    return [inputs["question"]] + variants[:n]  # always include the original

def rrf_fuse(ranked_lists: list[list[Document]], k: int = 60) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc.metadata.get("id") or doc.page_content[:64]
            scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[i] for i in sorted_ids[:10]]

def rag_fusion_retrieve(query: str, retriever) -> list[Document]:
    queries = expand_queries({"question": query}, n=3)
    ranked_lists = [retriever.get_relevant_documents(q) for q in queries]
    return rrf_fuse(ranked_lists)

# Usage
fused_docs = rag_fusion_retrieve("What are the late payment penalties?", rrf_retriever)
```

---

### Pattern 4: Retrieval Evaluation with Ragas (Isolated from Generation)

```python
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from datasets import Dataset

# Golden dataset: (question, ground_truth_contexts, ground_truth_answer)
eval_data = {
    "question": ["What is the refund policy?", ...],
    "contexts": [
        ["Full refunds are available within 30 days...", ...],  # retrieved chunks
        ...
    ],
    "ground_truth": ["Customers can get a full refund within 30 days...", ...],
}

dataset = Dataset.from_dict(eval_data)

# Evaluate retrieval metrics ONLY — no LLM answer quality metrics
results = evaluate(
    dataset=dataset,
    metrics=[context_precision, context_recall],
)

print(results)
# context_precision: fraction of retrieved chunks that are relevant
# context_recall: fraction of ground-truth info covered by retrieved chunks
# These run without generating an LLM answer — pure retrieval signal
```

---

*End of Study Guide — Session 11: Advanced Retrieval Strategies*
