# Vector Stores & Hybrid Search for RAG

Notes on local vector stores, in-memory sizing, and full-text + hybrid search options for the certification challenge RAG application.

---

## 1. SQLite & DuckDB as Vector DB (Minimal Deps, Privacy)

**Goal:** RAG with minimal DB dependencies and true privacy—no data leaving the machine.

### SQLite options

| Option | Install | Notes |
|--------|--------|--------|
| **sqlite-vec** | `pip install sqlite-vec` | C extension, single `.db` file, brute-force search. Good for ~100K–500K vectors. |
| **LangChain SQLiteVec** | `pip install sqlite-vec langchain-community` | Uses sqlite-vec; drop-in vector store with `similarity_search`, local embeddings. |
| **VectorLiteDB** | `pip install vectorlitedb` | Pure Python, simple API, ~100K vectors. No C extension. |

**Fully local RAG with LangChain + SQLite:**

```python
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import SQLiteVec

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = SQLiteVec(
    table="docs",
    db_file="./rag_vectors.db",
    embedding=embedding,
)
vector_store.add_texts(["Your document chunks..."])
results = vector_store.similarity_search("user question", k=4)
```

### DuckDB

- **Vector:** Native `ARRAY` + distance functions; optional **vss** extension (HNSW).
- **Install:** `pip install duckdb`; enable vss per DuckDB docs.
- **Use case:** Single file, in-process; good when you want analytics + vector search in one place.

### Summary

- **SQLite + sqlite-vec:** Lightest; single file; no server. Use with **SentenceTransformerEmbeddings** for 100% local.
- **DuckDB:** Same privacy story; better for larger scale or analytics + vectors.

---

## 2. In-Memory Vector Store: How Much Data Is Reasonable?

For **Qdrant (or any store) in-memory**, approximate sizing:

### Memory per vector (rough)

- **Per vector:** `dimension × 4` bytes (float32) + ~100–500 bytes for IDs, payloads, index.
- **384-dim** (e.g. MiniLM): ~2–3 KB per vector  
- **768-dim:** ~4–5 KB per vector  
- **1536-dim** (e.g. OpenAI): ~7–8 KB per vector  

### Safe ballpark by machine RAM

| Machine RAM | Safe for vector store | ~Vectors (384-dim, ~3 KB/vec) |
|-------------|------------------------|--------------------------------|
| 8 GB        | ~1–2 GB                | ~300K–600K                     |
| 16 GB       | ~2–4 GB                | ~600K–1.2M                     |
| 32 GB       | ~4–8 GB                | ~1.2M–2.5M                     |

**Rule of thumb:** Leave 50–60% of RAM for OS, Python, and the rest of the app. Qdrant in-memory uses roughly **1.5–2×** raw vector+payload size (including HNSW index).

**Reasonable in practice:** hundreds of thousands to low millions of chunks with 384-dim embeddings, as long as total usage fits the safe ballpark above.

---

## 3. Full-Text Search & Hybrid Search: Which Datastores Support Both?

All three support **full-text + vector**; hybrid = keyword + semantic together.

| Datastore | Full-text | Vector | Hybrid in one place? |
|-----------|-----------|--------|----------------------|
| **SQLite** | ✅ FTS5 (built-in) | ✅ sqlite-vec | ✅ Same DB; combine in app (e.g. RRF) |
| **DuckDB** | ✅ FTS extension | ✅ Arrays + vss | ✅ Same DB; combine in SQL or app |
| **Qdrant** | ✅ Full-text index on payloads | ✅ Vector search | ✅ Native: one Query API (dense + full-text) |

### SQLite (FTS5 + sqlite-vec)

- **Full-text:** FTS5 virtual table, `MATCH` queries.
- **Vector:** sqlite-vec.
- **Hybrid:** Run FTS query + vector query; merge in Python with **Reciprocal Rank Fusion (RRF)**. One DB file, no extra service.
- Refs: [Hybrid search with SQLite](https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html), [Simon Willison](https://simonwillison.net/2024/Oct/4/hybrid-full-text-search-and-vector-search-with-sqlite).

### DuckDB

- **Full-text:** Full-Text Search extension (e.g. `duckdb-fts`).
- **Vector:** ARRAY + distance functions; vss for HNSW.
- **Hybrid:** FTS + vector in SQL; combine (e.g. RRF) in SQL or in Python.

### Qdrant

- **Full-text:** Indexed payload fields, configurable tokenizers.
- **Vector:** Dense (and optional sparse) vectors.
- **Hybrid:** Single Query API (v1.10+): one request can combine dense vector + full-text. No manual RRF needed.
- Refs: [Full-text filters and indexes](https://qdrant.tech/articles/qdrant-introduces-full-text-filters-and-indexes), [Hybrid Search](https://qdrant.tech/articles/hybrid-search).

### Recommendation

- **Minimal deps + privacy:** SQLite (FTS5 + sqlite-vec) or DuckDB (FTS + vector)—both allow full-text and vector; you implement RRF (or similar) in app/SQL.
- **Already using Qdrant:** Use Qdrant’s native full-text + vector and hybrid API for less custom code.

---

*Generated from research and discussion on vector stores and hybrid search for the 12 Certification Challenge.*
