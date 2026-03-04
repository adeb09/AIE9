# Using a Semantic Chunker with ParentDocumentRetriever (LangChain)

## Short answer

**Yes, you can combine semantic chunking with ParentDocumentRetriever**, but not by passing `SemanticChunker` directly as `parent_splitter` or `child_splitter`. The supported approach is to **pre-split with `SemanticChunker`** and then use `ParentDocumentRetriever` with `parent_splitter=None` so that your semantic chunks become the parent documents.

---

## Why not pass SemanticChunker directly?

In LangChain’s implementation:

1. **`ParentDocumentRetriever`** expects:
   - `child_splitter: TextSplitter` (from `langchain_text_splitters`)
   - `parent_splitter: Optional[TextSplitter] = None`

2. **`SemanticChunker`** (from `langchain_experimental.text_splitter`) inherits from **`BaseDocumentTransformer`**, not from **`TextSplitter`**:
   - `SemanticChunker` MRO: `['SemanticChunker', 'BaseDocumentTransformer', 'ABC', 'object']`
   - So it is not a `TextSplitter` and does not satisfy the type expected by `ParentDocumentRetriever`.

3. The retriever only calls `.split_documents()` on the splitters. So at runtime a `SemanticChunker` would be compatible in terms of behavior, but:
   - The API is typed to accept `TextSplitter` only.
   - If the retriever is validated (e.g. via Pydantic), passing a `SemanticChunker` as `parent_splitter` or `child_splitter` may raise a validation error.

So the **official, type-safe way** to combine them is the pre-split pattern below, not by plugging `SemanticChunker` in as one of the retriever’s splitters.

---

## Recommended: pre-split with SemanticChunker, then use ParentDocumentRetriever

When `parent_splitter=None`, `ParentDocumentRetriever` does **not** split the input documents; they are treated as the **parent documents**. Each parent is then split by `child_splitter` into smaller chunks for the vectorstore. So you can make the “parents” be your semantic chunks.

Flow:

1. Split raw documents with **`SemanticChunker`** → list of semantic-chunk `Document`s.
2. Build **`ParentDocumentRetriever`** with:
   - `parent_splitter=None`
   - `child_splitter=RecursiveCharacterTextSplitter(...)` (or another `TextSplitter`).
3. Call **`retriever.add_documents(semantic_chunks)`**.

Result:

- **Vectorstore (search):** small child chunks (from `child_splitter`) for better semantic matching.
- **Docstore (return):** your semantic parent chunks, so the model gets context that respects semantic boundaries.

### Example (conceptual)

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore  # or your vectorstore
from langchain_openai import OpenAIEmbeddings

# 1) Semantic chunking: raw_docs → semantic parent chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
semantic_docs = semantic_chunker.split_documents(raw_docs)

# 2) ParentDocumentRetriever: semantic docs = parents, child_splitter = small chunks for indexing
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
vectorstore = QdrantVectorStore(...)  # your setup
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=None,  # no further parent splitting; we already have semantic parents
)
retriever.add_documents(semantic_docs)
```

No need to pass `SemanticChunker` as `parent_splitter` or `child_splitter`; the combination is achieved by **pre-processing** with `SemanticChunker` and then using the retriever with `parent_splitter=None`.

---

## Alternative: use SemanticChunker as “parent splitter” via a TextSplitter wrapper

If you really want to use `SemanticChunker` in the “parent” role **inside** the retriever (so that the retriever splits raw docs with it), you must satisfy the `TextSplitter` type. That means implementing a small **wrapper** that subclasses `TextSplitter` and delegates to `SemanticChunker.split_documents()` / `split_text()` (and, if needed, implementing `split_text` in terms of the semantic chunker). This is more involved and not the pattern shown in the assignment notebook or in the official Parent Document Retriever docs, which use `RecursiveCharacterTextSplitter` for both parent and child.

---

## Summary

| Approach | Supported / Typed? | Recommendation |
|----------|--------------------|----------------|
| Pass `SemanticChunker` as `parent_splitter` or `child_splitter` | No (type is `TextSplitter`; `SemanticChunker` is `BaseDocumentTransformer`) | Avoid; may fail validation. |
| Pre-split with `SemanticChunker`, then `ParentDocumentRetriever(parent_splitter=None, child_splitter=...)` and `add_documents(semantic_docs)` | Yes | **Recommended**: semantic parents + small children for retrieval. |
| Wrapper class that extends `TextSplitter` and uses `SemanticChunker` internally | Possible but custom | Only if you need semantic splitting inside `add_documents(raw_docs)`. |

So: **you can use a semantic chunker with ParentDocumentRetriever** by treating semantic chunks as the parent documents (pre-split with `SemanticChunker`, then use the retriever with `parent_splitter=None`). You cannot plug `SemanticChunker` in as the retriever’s `parent_splitter` or `child_splitter` without a `TextSplitter`-compatible wrapper.

---

# SemanticChunker with Other Retrievers (BM25, Cohere Reranker, Multi-Query, Ensemble)

**Summary:** SemanticChunker works with **all** of these retrievers. None of them take a splitter; they operate on **pre-built document lists or retrievers**. So you use SemanticChunker at **indexing time** to create chunks, then build BM25 or a vector store on those chunks and optionally wrap with the Cohere reranker, Multi-query retriever, or Ensemble retriever.

---

## BM25 Retriever

**Compatibility: Yes — use semantic chunks as the document list.**

`BM25Retriever` is built from a **list of documents** via `BM25Retriever.from_documents(documents, k=...)`. It does not accept a splitter; it indexes whatever `Document` list you pass. So you can use SemanticChunker to produce chunks and pass them directly to BM25.

**Flow:**

1. `semantic_docs = semantic_chunker.split_documents(raw_docs)`
2. `bm25_retriever = BM25Retriever.from_documents(semantic_docs, k=10)`

**Effect:** BM25 performs lexical (keyword) search over **semantic chunks** instead of fixed-size character chunks. Chunk boundaries follow semantic breaks, which can improve precision when a query matches one topic and you want the whole coherent block (e.g. a full paragraph or section) rather than an arbitrary mid-sentence split. BM25 still scores by term frequency; only the unit of retrieval changes.

**Example:**

```python
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker

semantic_docs = semantic_chunker.split_documents(raw_docs)
bm25_retriever = BM25Retriever.from_documents(semantic_docs, k=10)
```

---

## Cohere Reranker (Contextual Compression)

**Compatibility: Yes — reranker is agnostic to how chunks were created.**

The Cohere reranker is used as a **wrapper** around another retriever via `ContextualCompressionRetriever(base_compressor=CohereRerank(...), base_retriever=base_retriever)`. It takes the documents returned by the base retriever and re-ranks them. It does not care how those documents were chunked; it only needs a retriever that returns a list of documents.

**Flow:**

1. Create semantic chunks: `semantic_docs = semantic_chunker.split_documents(raw_docs)`.
2. Build a base retriever on those chunks (e.g. vector store or BM25 from `semantic_docs`).
3. Wrap with Cohere: `ContextualCompressionRetriever(base_compressor=CohereRerank(...), base_retriever=base_retriever)`.

**Effect:** You get semantic chunk boundaries **plus** Cohere’s reranking. The reranker improves relevance of the final list; semantic chunking improves the quality and coherence of each retrieved unit.

**Example:**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Base retriever built on semantic chunks (e.g. vector store from semantic_docs)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
compressor = CohereRerank(model="rerank-v3.5")
reranker_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
```

---

## Multi-Query Retriever

**Compatibility: Yes — multi-query wraps any retriever.**

`MultiQueryRetriever` takes an existing retriever and an LLM, generates multiple query reformulations, and merges results from the underlying retriever. It does not take a splitter or document list; it only wraps another retriever.

**Flow:**

1. Create semantic chunks and build a base retriever (vector store or BM25 on `semantic_docs`).
2. Wrap with MultiQuery: `MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)`.

**Effect:** You get semantic chunking (better chunk boundaries) **plus** multi-query (higher recall via multiple phrasings). The underlying retriever continues to return whatever chunks it indexes—so if the base retriever is built on semantic chunks, MultiQuery will retrieve and merge those semantic chunks.

**Example:**

```python
from langchain.retrievers import MultiQueryRetriever

# base_retriever could be from vector_store or BM25 built on semantic_docs
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOpenAI(model="gpt-4.1-nano"),
)
```

---

## Ensemble Retriever

**Compatibility: Yes — ensemble combines retrievers; use semantic chunks in each.**

`EnsembleRetriever` takes a **list of retrievers** and optional weights, then merges and re-ranks their results (e.g. reciprocal rank fusion). It does not take a splitter or document list; it only needs two or more retrievers.

**Flow:**

1. Create semantic chunks: `semantic_docs = semantic_chunker.split_documents(raw_docs)`.
2. Build two (or more) retrievers on the **same** semantic chunks, e.g.:
   - Vector store from `semantic_docs` → `vector_retriever`
   - BM25 from `semantic_docs` → `bm25_retriever`
3. Combine: `EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])`.

**Effect:** Ensemble retrieval over **semantic chunks**: you get both dense (vector) and sparse (BM25) signals while keeping chunk boundaries semantic. You can also wrap one or both of the underlying retrievers with Cohere or MultiQuery before passing them to the ensemble.

**Example:**

```python
from langchain.retrievers import EnsembleRetriever

# Both built on the same semantic_docs
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(semantic_docs, k=10)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)
```

---

## Summary: SemanticChunker + Retrievers

| Retriever | How SemanticChunker fits | Compatible? |
|-----------|---------------------------|-------------|
| **ParentDocumentRetriever** | Pre-split with SemanticChunker; use as parents with `parent_splitter=None`. | Yes (pre-split pattern). |
| **BM25** | Pass semantic chunks to `BM25Retriever.from_documents(semantic_docs, ...)`. | Yes. |
| **Cohere Reranker** | Build base retriever on semantic chunks; wrap with `ContextualCompressionRetriever` + `CohereRerank`. | Yes. |
| **Multi-Query** | Build base retriever on semantic chunks; wrap with `MultiQueryRetriever.from_llm(...)`. | Yes. |
| **Ensemble** | Build each retriever (vector, BM25, etc.) on the same semantic chunks; pass to `EnsembleRetriever`. | Yes. |

**Takeaway:** SemanticChunker is an **indexing-time** choice. Use it to create your chunk list; then feed that list to BM25 or a vector store, and optionally wrap with Cohere, Multi-query, or Ensemble. No special API or wrapper is required for BM25, Cohere, Multi-query, or Ensemble—they all work on documents or retrievers, not on splitters.
