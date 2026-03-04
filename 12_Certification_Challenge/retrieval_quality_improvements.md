# Retrieval Quality Improvements for AskMyBookmark

Notes from diagnosing and fixing retrieval quality issues in the hybrid search RAG pipeline.

---

## The Core Problem: Curated List Repo Contamination

### What was happening

When a user asked *"What are some top deep learning libraries I have starred?"*, the retrieved
contexts were repos like:

- `lukasmasuch/best-of-ml-python` — *"A ranked list of awesome machine learning Python libraries"*
- `SkalskiP/courses` — *"A curated collection of AI courses and resources"*
- `ritchieng/the-incredible-pytorch` — *"A curated list of PyTorch tutorials"*

The LLM then read the READMEs of those curated list repos, found library names (fastai, Keras,
PyTorch, TensorFlow, JAX) within them, and presented those as if they were directly starred repos.

### Why it happened

**Both retrievers had the same bias toward curated list repos:**

- **BM25** scores curated list repos extremely high because their READMEs contain every relevant
  keyword in high frequency. A 30,000-character list of libraries beats a focused 500-character
  library README every time — BM25 rewards term frequency.
- **Dense (semantic)** had the same bias — embedding a README that is literally a ranked list of
  deep learning libraries produces a vector very close to *"what are the top deep learning
  libraries?"*.

Shifting weights from `[0.5, 0.5]` to `[0.4, 0.6]` or `[0.3, 0.7]` does not fix this. Both
signals point in the wrong direction for the same reason.

---

## Why Chunking Would Make Things Worse

Chunking curated list READMEs would split them into small, highly targeted chunks — e.g.:

> "fastai — high-level deep learning on PyTorch. Keras — neural network API. TorchVision —
> datasets and transforms for CV."

Each chunk would score *higher* against relevant queries than the full README does today, because
the noise from unrelated sections is removed. Chunking makes curated list repos more precise and
harder to filter.

The retrieval unit for this use case should be the **repository**, not a passage from a README.
The primary relevance signal (description + topics) is already short — chunking adds no value
there and only amplifies the curated list problem in the README content.

---

## Solution: Three-Layer Defense

### Layer 1 — Metadata-only retrieval track (index time)

Create a separate set of documents containing only `description` and `topics` — no README.
This is the right "chunking strategy" for this use case: extract the most semantically dense,
noise-free signal into a lean document.

A curated list repo's description is just *"A ranked list of awesome machine learning Python
libraries"* — it will not match *"top deep learning libraries"* nearly as strongly as a repo
whose description is *"A deep learning framework"*.

Also classify every repo at index time so the label is stored in metadata and available for
filtering downstream:

```python
CURATED_KEYWORDS = {"curated", "awesome", "ranked list", "collection of links", "resources", "list of"}

def classify_repo(meta: dict) -> str:
    desc = (meta.get("description") or "").lower()
    return "curated_list" if any(kw in desc for kw in CURATED_KEYWORDS) else "library"

def is_curated_list(meta: dict) -> bool:
    return classify_repo(meta) == "curated_list"

# Metadata-only documents
metadata_docs = [
    Document(
        page_content=(
            f"{meta.get('description') or ''}\n"
            f"Topics: {' '.join(meta.get('topics', []))}"
        ),
        metadata={**meta, "id": id_, "repo_type": classify_repo(meta)}
    )
    for text, id_, meta in zip(docs, ids, metadata)
]

# Full-content documents (description + README)
full_docs = [
    Document(
        page_content=text,
        metadata={**meta, "id": id_, "repo_type": classify_repo(meta)}
    )
    for text, id_, meta in zip(docs, ids, metadata)
]
```

### Layer 2 — Three-way ensemble with Qdrant pre-filter (retrieval time)

Replace the original two-retriever ensemble with three retrievers:

| Retriever | Content | Weight | Notes |
|---|---|---|---|
| `bm25_metadata` | Metadata-only docs | 0.25 | Short text → clean BM25 signal |
| `dense_metadata` | Metadata-only docs | 0.45 | Pre-filtered: curated repos excluded |
| `dense_full` | Full-content docs | 0.30 | Recall fallback; MMR for diversity |

The metadata tracks dominate at 0.70 combined weight. The Qdrant pre-filter on `dense_metadata`
completely excludes curated repos before RRF fusion — the highest-weight signal never sees them.

**BM25 `k=5`** on metadata docs (down from 10): metadata text is very short; fewer candidates
are needed, and BM25 on short text behaves cleanly without the term-frequency inflation that
plagues it on long READMEs.

**MMR `lambda_mult=0.7`** on both dense retrievers: slightly favours relevance over diversity —
you still want topically close repos, just not five identical curated list repos crowding everything
else out.

```python
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# BM25 over metadata-only docs
bm25_metadata_retriever = BM25Retriever.from_documents(metadata_docs)
bm25_metadata_retriever.k = 5

# Dense over metadata-only docs — Qdrant pre-filters curated repos out
dense_metadata_retriever = meta_vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 25,
        "lambda_mult": 0.7,
        "filter": Filter(
            must_not=[
                FieldCondition(
                    key="metadata.repo_type",
                    match=MatchValue(value="curated_list")
                )
            ]
        ),
    },
)

# Dense over full-content docs — curated repos allowed here for recall
dense_full_retriever = full_vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 25, "lambda_mult": 0.7},
)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_metadata_retriever, dense_metadata_retriever, dense_full_retriever],
    weights=[0.25, 0.45, 0.30],
)
```

### Layer 3 — Pre-Cohere rerank pool cap (query time)

Cohere's reranker scores documents **independently of their input order** — moving curated repos
to the back of the list does nothing. The fix is to hard-cap how many curated repos are even
eligible to be reranked.

Replace `ContextualCompressionRetriever` with a manual retrieve → filter → rerank pipeline in
the `retrieve` function:

```python
MAX_CURATED_IN_RERANK = 2

def retrieve(state):
    candidates = ensemble_retriever.invoke(state["question"])

    focused = [doc for doc in candidates if not is_curated_list(doc.metadata)]
    curated  = [doc for doc in candidates if is_curated_list(doc.metadata)]

    # focused repos fill the pool; curated repos capped at MAX_CURATED_IN_RERANK
    rerank_candidates = focused + curated[:MAX_CURATED_IN_RERANK]

    compressor = CohereRerank(model="rerank-v3.5")
    reranked_docs = compressor.compress_documents(
        documents=rerank_candidates,
        query=state["question"],
    )
    return {"context": reranked_docs}
```

Set `MAX_CURATED_IN_RERANK = 0` to exclude curated repos entirely — useful if users are almost
always asking for specific libraries. `1` or `2` is a safe middle ground for edge cases like
*"do I have any awesome-lists for NLP?"*.

---

## System Prompt Fixes

Two prompt-level guardrails were added alongside the retrieval fixes:

**1. Anchor the LLM to the `repo` field**

> "Only surface repositories that are directly starred by the user. A directly starred repository
> is identified by the `repo` field in the retrieved context (e.g., `owner/repo-name`)."

**2. Detect and contain curated list repos**

> "Do NOT treat tools, libraries, papers, or projects that are merely *mentioned or listed inside*
> a repository's README as starred repositories. If a README is a curated list, awesome list, or
> ranked collection (e.g., its description contains phrases like 'ranked list', 'curated list',
> 'awesome', 'collection of links', or 'resources'), treat that entire repository as a single
> starred item — do not extract or present its listed contents as individual starred repos."

Note: prompt fixes alone are not sufficient. The LLM technically isn't hallucinating — those
library names do appear in the retrieved context. The real fix must happen at the retrieval layer.
Prompt fixes are a useful second line of defence once the retrieval signal is clean.

---

## Summary: Defense in Depth

| Layer | Where | What it does |
|---|---|---|
| **Index-time classification** | Document creation | Tags every repo as `library` or `curated_list` in metadata |
| **Qdrant pre-filter** | Dense metadata retriever | Curated repos never enter the highest-weight (0.45) retrieval signal |
| **Ensemble weighting** | RRF fusion | Metadata tracks (0.70 combined) outweigh full-content track (0.30) |
| **Rerank pool cap** | `retrieve()` function | At most `MAX_CURATED_IN_RERANK` curated repos reach Cohere |
| **System prompt** | LLM generation | Instructs LLM not to extract items from curated list READMEs |
