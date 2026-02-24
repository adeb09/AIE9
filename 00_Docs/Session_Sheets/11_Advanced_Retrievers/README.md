# Session 11: 🐕 Advanced Retrievers

🎯 Learn best practices for retrieval and a systematic approach for deciding on the best retriever for your AI applications.

📚 **Learning Outcomes**
- Understand how advanced retrieval, chunking, and ranking techniques can enhance the context given to agentic RAG applications
- Understand the fine lines between chunking, retrieval, and re-ranking — and how they all affect context quality
- Learn to systematically compare the performance of retrieval algorithms using metrics (RAGAS), cost, and latency
- Develop intuition for when to use which retriever for your specific data and use case

🧰 **New Tools**
Retrievers: [LangChain Retrievers](https://docs.langchain.com/oss/python/integrations/retrievers)

## 📛 Required Tooling & Account Setup
No new tools or accounts required

## 📜 Recommended Reading

- [BM25](https://www.nowpublishers.com/article/Details/INR-019) (2009)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (2009) — a two-page paper that underpins ensemble retrieval
- [Semantic Chunking](https://x.com/GregKamradt/status/1737921395974430953?s=20) (2023)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — for exploring re-ranking and embedding model options

---

# 📂 **The Role of Advanced Retrieval Methods**

Retrieval in RAG is essential for feeding the right data into a large language model (LLM) to improve the quality of generated responses. It is crucial to avoid hallucinations and redundancy while ensuring the returned reference material is relevant and fact-checkable.

The more relevant the retrieved context, the more accurate the generation — this is the fundamental principle of RAG.

### Chunking → Retrieval → Ranking: The Fine Lines

A key insight from this module is that the boundaries between chunking, retrieval, and ranking are blurry. Both chunking and the retrieval process affect the context that gets returned to the LLM. **Metadata** plays a critical role in bridging these stages — it tells us where to look, which haystack a chunk comes from, and helps the LLM reason through which chunks to pick.

> "Both chunking and the retrieval process both affect the context we return. Metadata is a key thing that we want to be leveraging to optimize what we put in context." — Dr. Greg

We can think of RAG more broadly as **information retrieval** (not just dense vector retrieval), which opens us up to hybrid approaches that combine the old (keyword search) with the new (semantic search).

---

# 🔪 **Chunking Strategies**

Chunks are created when natural language (text) from our source documents gets split. Common methods for text splitting include:

- By number of characters, with overlapping sliding window
- By sentence or paragraph
- A hybrid of the two (e.g., recursive character text splitter)

### Naive (Recursive Character) Chunking

The [recursive character text splitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/) has become the de facto standard for building RAG systems. The underlying hypothesis is that **text that is close together in a document tends to be semantically related** — sentences, paragraphs, and chapters provide natural chunks of context.

The recursive character text splitter is smarter than fixed-size chunking because it splits hierarchically: it first tries to split on double newlines, then single newlines, then spaces, and only as a last resort does it split within a word. This keeps natural text boundaries intact while targeting a specified chunk size.

**Fixed-size chunking**, by contrast, counts a fixed number of characters and splits regardless of whether it lands in the middle of a word — quite too naive for most applications.

[ChunkViz](https://chunkviz.up.railway.app/) is a useful tool for visualizing how different chunk sizes and strategies affect your text.

### Semantic Chunking

Despite the utility of the recursive text splitter, deciding on chunk sizes is still a bit of a black art. **Semantic chunking** takes a fundamentally different approach by using embeddings to determine chunk boundaries:

1. **Split** the document into individual sentences
2. **Group** adjacent sentences (e.g., groups of 3)
3. **Compare** the embeddings of adjacent groups — if they are similar enough, merge them into a single chunk; if they are different enough, split them apart

The result is variable-sized chunks that respect the actual semantic structure of the text rather than arbitrary character counts. For example, when applied to "Alice in Wonderland," semantic chunking might produce chunks that align with natural narrative shifts rather than fixed 200-character windows.

**Key question:** How do you know how big a chunk should be for your data? It depends — leverage whatever structure already exists in the text. If the text lacks natural structure, semantic chunking can help discover it.

> "Semantic chunking is not a retrieval strategy, but it is a way to improve the performance of retrieval." — Chris Alexiuk

---

# 🧰 **Retrieval Methods for Your Toolkit**

The session introduces several retrieval methods, progressing from the simplest to the most sophisticated. In the notebook, each method is implemented by **swapping out the retriever** while keeping the rest of the LCEL chain the same — that's the key pattern.

### 1. Naive Retrieval
The simplest form of retrieval: embed the query, search for similar chunks via cosine similarity, and return the Top K chunks. Easy to set up but can return less relevant or redundant content. This is the baseline that everything else is measured against.

### 2. BM25 (Keyword / Sparse Vector Retrieval)
[BM25](https://www.nowpublishers.com/article/Details/INR-019) (Best Matching 25) is a classic NLP technique that uses a **bag-of-words** retrieval function. It ranks documents based on exact query term matches, considering term frequency, document length, and average document length.

**Why it matters:** Semantic similarity can miss exact matches. For example, searching for an error code like "TS999" is much better suited for keyword search than embedding-based retrieval. BM25 excels in these cases.

> "Before adding LLMs and vector retrieval, ask: have we even implemented keyword search yet? Do we even need LLMs yet?" — Dr. Greg

### 3. Parent Document Retrieval
Instead of returning the small chunks directly, this method returns the **larger parent document** (or section) from which the chunk originates. The approach:

- Create **two sets of chunks**: small (child) chunks for searching and large (parent) chunks for returning
- Track **metadata** linking each child chunk to its parent
- **Search** on the small chunks (for precision), but **return** the parent chunks (for broader context)

This solves the "pile of PDFs" problem — when you retrieve a chunk, you know exactly which document, which page, and which section it comes from. It's especially useful when information related to a concept isn't labeled with that concept's name (e.g., searching for "Bernoulli equation" when the equation itself doesn't contain the word "Bernoulli").

**Implementation:** Build two vector stores — one in your vector DB (e.g., Qdrant) for the child chunks, and one in memory for the parent chunks.

### 4. Multi-Query Retriever
This retriever takes the original user query and uses an LLM to generate **n different queries** from different perspectives. Then it retrieves documents for each query and returns all unique documents as context.

**Why it matters:** People are often bad at formulating queries, especially for systems they're unfamiliar with. Multi-query retrieval expands the search surface area by rephrasing the question in multiple ways — similar to how RAGAS generates alternative questions for evaluation.

### 5. Ensemble Retrieval (Hybrid Search)
Combines multiple retrievers and merges their results using [**Reciprocal Rank Fusion (RRF)**](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). The key idea: if a document appears highly ranked across multiple different retrieval methods, it is probably more relevant.

RRF merges ranked lists from different retrievers regardless of how they are scored — it only cares about relative rank, not the score values. Weights can be assigned to each retriever to control their influence.

**This is often the best approach in practice.** Combining keyword search (BM25) with vector retrieval (semantic) gives the best of both worlds. Ensembles are consistently the answer in ML, and retrieval is no different.

> "If one of these retrievers is good, what if we try all of them at the same time? The sources that appear most frequently across all our retrievers are probably more likely to be ultimately correct." — Chris Alexiuk

### 6. Contextual Compression (Re-ranking)
After initial retrieval returns many chunks, re-ranking **compresses** the results down to the most relevant ones. This is a two-stage process: retrieve many documents cheaply, then use a slower, more expensive, more accurate model to select the top few.

[Cohere Rerank](https://cohere.com/rerank) is an industry standard for this, but there are also many open-source re-ranking models available — check the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for options.

**Important distinction:** Re-ranking is different from reciprocal rank fusion. Re-ranking uses a model to score relevance. RRF merges multiple ranked lists into one.

---

# 📊 **Evaluating Retrieval Methods**

Your assignment this week is to **systematically compare** all of the retrieval methods above using RAGAS. The evaluation goes beyond just RAGAS metric accuracy to also consider:

- **Cost** — How expensive is each retrieval method? (Use LangSmith to track this)
- **Latency** — How long does each method take? (Also trackable via LangSmith)
- **RAGAS Metric Accuracy** — How well does each method actually perform on retrieval and generation quality?

The goal: use metrics to select the best retriever for your specific corpus and use case. There is no universally "best" retriever — it depends on your data, your users, and your constraints.
