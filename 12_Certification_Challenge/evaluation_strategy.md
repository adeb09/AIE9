# AskMyBookmark — Evaluation Strategy

## Overview

AskMyBookmark is a **search and discovery** tool, not a traditional question-answering RAG system.
This distinction fundamentally shapes the evaluation strategy. Most RAGAS metrics were designed for
factual Q&A where a single correct reference answer exists. Queries like
*"show me repos about Bayesian statistics"* have no single correct answer — they are conceptual
retrieval tasks. The evaluation plan therefore separates into two layers:

1. **Retrieval quality** — did the retriever surface the right repos?
2. **Generation quality** — did the LLM respond faithfully and relevantly given what it retrieved?

Both layers are compared across the two pipeline variants built in this project:

- **Pipeline A (Naive):** dense vector retrieval only (`text-embedding-3-small`, k=10)
- **Pipeline B (Hybrid):** BM25 + dense ensemble + Cohere rerank (`rerank-v3.5`, k=10)

---

## Metrics

### Retrieval Metrics

#### Precision@K

**What it measures:** Of the top K retrieved documents, what fraction are actually relevant to the query?

```
Precision@K = (# relevant docs in top K) / K
```

**Why it applies here:** AskMyBookmark retrieves k=10 chunks per query. Not all will be topically
relevant. Precision@K gives a single interpretable number per query that directly measures
retrieval quality. It is also the clearest way to show the improvement from Cohere reranking —
reranking reshuffles the top of the list, so Precision@5 is expected to improve more than
Precision@10.

**Computed at:** K=5 and K=10.

#### ContextEntityRecall (RAGAS)

**What it measures:** Whether key entities (repo names) from the reference answer appear in the
retrieved context. A softer, LLM-judged complement to Precision@K.

**Why it applies here:** For a query like "Bayesian inference repos," the expected entities are
specific repo names (e.g., `pymc-devs/pytensor`). ContextEntityRecall measures whether those
repo names were actually present in what the retriever returned to the model.

#### LLMContextRecall (RAGAS)

**What it measures:** Whether the retrieved context contains sufficient information to answer the
query, as judged by an LLM against a reference answer.

**Why it applies here:** Measures whether the right information reached the generator at all.
Requires a reference answer per query (generated from the ground truth dict — see below).

---

### Generation Metrics

#### Faithfulness (RAGAS)

**What it measures:** Whether every claim in the generated response is grounded in the retrieved
context. A faithfulness score of 1.0 means no hallucinations.

**Why it is the most important metric for this application:** The system prompt explicitly
instructs the model not to suggest repos outside the retrieved context. If faithfulness is low,
the model is inventing repos that were never starred — the single worst failure mode for a
personal bookmark tool.

#### ResponseRelevancy (RAGAS)

**What it measures:** Whether the response addresses the user's query, as judged by generating
candidate questions from the response and checking if they match the original query.

**Why it applies here:** Irrelevant responses (e.g., the model answering a question about agents
with repos about tokenizers) are a core quality failure for a search tool.

#### NoiseSensitivity (RAGAS)

**What it measures:** How much the response degrades when noisy or irrelevant context is present
in the retrieved chunks.

**Why it is particularly important here:** With k=10 retrieved chunks, not all will be topically
relevant. Additionally, many starred repos are "awesome lists" or curated link aggregators whose
READMEs contain hundreds of mentioned-but-not-starred repos. NoiseSensitivity tests whether the
model correctly ignores that noise.

---

### Metric not used: FactualCorrectness

`FactualCorrectness()` is designed for factual Q&A with an objectively correct answer.
*"Show me repos about AI agents"* has no single correct factual answer. This metric produces
meaningless scores for a search/discovery task and is excluded from this evaluation.

---

## Ground Truth Construction with SearchArray

The ground truth test set is built programmatically using
[searcharray](https://github.com/softwaredoug/searcharray) — a pandas-native BM25 library — to
score all starred repos against each test query using the structured `topics`, `description`, and
`language` metadata from the GitHub API. This avoids manual labeling of 2000+ repos.

### Why topics metadata is the right oracle

GitHub topics are user-applied, curated labels (e.g., `bayesian-inference`, `multi-agent`,
`full-text-search`). They are intentionally descriptive and high-precision. BM25 scoring over
topics gives a principled, reproducible way to define relevance without subjective manual review.

### Tokenization decision

Topics like `bayesian-inference` are hyphen-separated compound terms. BM25 tokenizes on
whitespace by default, so `bayesian-inference` would be treated as a single token and would not
match a query of `"bayesian inference"` (two separate tokens). The fix is to replace hyphens with
spaces when building the search field, so each word is indexed independently.

---

## Full Evaluation Pipeline Code

### Step 1 — Install dependency

```bash
pip install searcharray
```

### Step 2 — Build the BM25 search index over repo metadata

```python
import pandas as pd
from searcharray import SearchArray

# metadata is already in memory after organize_repo_data()
df_repos = pd.DataFrame(metadata)

def build_search_field(row: dict) -> str:
    """
    Combine topics (hyphen-split), description, language, and repo name
    into a single BM25-searchable string per repo.
    """
    topics_text = " ".join(t.replace("-", " ") for t in (row["topics"] or []))
    desc_text   = row["description"] or ""
    lang_text   = row["language"] or ""
    # split repo name: "pymc-devs/pytensor" → "pymc devs pytensor"
    repo_text   = row["repo"].replace("/", " ").replace("-", " ").replace("_", " ")
    return f"{topics_text} {desc_text} {lang_text} {repo_text}".lower().strip()

df_repos["searchable"]     = df_repos.apply(build_search_field, axis=1)
df_repos["searchable_idx"] = SearchArray.index(df_repos["searchable"])

print(f"Indexed {len(df_repos)} repos")
```

### Step 3 — Define the relevance scoring function

```python
def get_relevant_repos(df: pd.DataFrame, query: str, min_score: float = 0.5) -> pd.DataFrame:
    """
    BM25-score all repos against a query and return those above min_score,
    sorted by descending relevance. These form the ground truth relevant set
    for Precision@K and RAGAS context recall metrics.

    Tune min_score per query:
      - Niche queries (e.g. "bayesian inference"): 0.1–0.5 (fewer matching repos)
      - Broad queries (e.g. "machine learning"):   1.0–2.0 (many matching repos)
    """
    scores = df["searchable_idx"].array.score(query)
    result = df.copy()
    result["bm25_score"] = scores
    relevant = result[result["bm25_score"] > min_score].sort_values(
        "bm25_score", ascending=False
    )
    return relevant[["repo", "url", "topics", "language", "bm25_score"]]
```

### Step 4 — Define test queries and generate the ground truth dict

```python
# Each query is written in natural language matching how a user would search.
# Queries intentionally span different topic types to stress-test both retrieval
# strategies across varied conceptual searches.
TEST_QUERIES = [
    # probabilistic / statistics
    "bayesian inference statistics probabilistic programming",
    # agent frameworks
    "ai agents multi agent framework autonomous orchestration",
    # LLM inference / serving
    "large language model llm inference serving",
    # RAG / retrieval
    "retrieval augmented generation rag vector search embeddings",
    # evaluation / benchmarks
    "evaluation benchmark llm testing metrics",
    # full text / lexical search
    "full text search bm25 lexical keyword",
    # data science / ML
    "machine learning deep learning neural network",
    # Go language
    "Go golang backend server concurrency",
    # async / Python
    "async asynchronous python concurrent",
    # recommendation systems
    "recommendation system collaborative filtering",
    # data / datasets
    "dataset data collection benchmark research",
    # code agents / developer tools
    "code review coding agent developer tool",
    # graph / workflow
    "graph workflow pipeline orchestration dag",
    # fuzzy / approximate search
    "fuzzy search approximate nearest neighbour similarity",
    # finance
    "finance financial benchmark accounting",
]

# Build ground truth: query → list of relevant repo name strings
ground_truth: dict[str, list[str]] = {}

for query in TEST_QUERIES:
    relevant = get_relevant_repos(df_repos, query, min_score=0.5)
    ground_truth[query] = relevant["repo"].tolist()
    print(f"[{len(relevant):3d} relevant]  {query}")

print(f"\nGround truth generated for {len(TEST_QUERIES)} queries.")
```

> **Important:** Manually inspect the ground truth for 3–4 queries before running the evaluation.
> If a query returns 0 relevant repos (because matching repos have no topics set), lower
> `min_score` to `0.1` or add more description-level keywords to the query string.

### Step 5 — Precision@K implementation

```python
def precision_at_k(retrieved_docs: list, relevant_repos: list, k: int) -> float:
    """
    retrieved_docs : list of LangChain Document objects from invoking a retriever
    relevant_repos : ground_truth[query] — list of relevant repo name strings
    k              : cutoff rank
    """
    top_k       = retrieved_docs[:k]
    top_k_repos = [doc.metadata.get("repo") for doc in top_k]
    hits        = sum(1 for repo in top_k_repos if repo in set(relevant_repos))
    return hits / k if k > 0 else 0.0


def evaluate_retriever_precision(
    retriever,
    ground_truth: dict,
    k_values: list[int] = [5, 10]
) -> dict:
    """
    Compute mean Precision@K across all test queries for one retriever.
    Skips queries with no known relevant repos (avoids dividing by zero ground).
    """
    results = {k: {} for k in k_values}

    for query, relevant_repos in ground_truth.items():
        if not relevant_repos:
            continue
        retrieved = retriever.invoke(query)
        for k in k_values:
            results[k][query] = precision_at_k(retrieved, relevant_repos, k)

    summary = {}
    for k in k_values:
        scores = list(results[k].values())
        summary[f"mean_precision_at_{k}"] = sum(scores) / len(scores) if scores else 0.0
        summary[f"per_query_at_{k}"]      = results[k]

    return summary


# Evaluate both retrievers
print("=== Pipeline A: Naive Dense Retriever ===")
naive_eval = evaluate_retriever_precision(naive_retriever, ground_truth)
print(f"  Mean Precision@5  : {naive_eval['mean_precision_at_5']:.3f}")
print(f"  Mean Precision@10 : {naive_eval['mean_precision_at_10']:.3f}")

print("\n=== Pipeline B: Hybrid (BM25 + Dense + Cohere Rerank) ===")
# compression_retriever is built inside the retrieve node — reconstruct it standalone:
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

compressor          = CohereRerank(model="rerank-v3.5")
standalone_hybrid   = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever,
    search_kwargs={"k": 10}
)

hybrid_eval = evaluate_retriever_precision(standalone_hybrid, ground_truth)
print(f"  Mean Precision@5  : {hybrid_eval['mean_precision_at_5']:.3f}")
print(f"  Mean Precision@10 : {hybrid_eval['mean_precision_at_10']:.3f}")
```

### Step 6 — RAGAS generation metrics

```python
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    NoiseSensitivity,
    LLMContextRecall,
    ContextEntityRecall,
)
from ragas import EvaluationDataset, SingleTurnSample
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ragas_llm        = ChatOpenAI(model="gpt-4.1-mini")
ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def build_ragas_dataset(rag_graph, ground_truth: dict) -> EvaluationDataset:
    """
    Run the full RAG graph for each test query and collect:
      - user_input       : the query
      - retrieved_contexts: list of page_content strings from retrieved docs
      - response         : the generated answer
      - reference        : a simple reference answer built from ground truth repos
    """
    samples = []
    for query, relevant_repos in ground_truth.items():
        if not relevant_repos:
            continue

        result    = rag_graph.invoke({"question": query})
        context   = result.get("context", [])
        response  = result.get("response", "")

        # Reference answer: a plain list of the known relevant repos for this query.
        # Used by LLMContextRecall and ContextEntityRecall.
        reference = "Relevant repositories: " + ", ".join(relevant_repos[:10])

        samples.append(SingleTurnSample(
            user_input          = query,
            retrieved_contexts  = [doc.page_content for doc in context],
            response            = response,
            reference           = reference,
        ))

    return EvaluationDataset(samples=samples)


# Build datasets for both pipelines
print("Building RAGAS evaluation datasets...")
naive_dataset  = build_ragas_dataset(naive_rag_graph, ground_truth)
hybrid_dataset = build_ragas_dataset(hybrid_rag_graph, ground_truth)

metrics = [
    Faithfulness(),
    ResponseRelevancy(),
    NoiseSensitivity(),
    LLMContextRecall(),
    ContextEntityRecall(),
]

print("\nEvaluating Pipeline A: Naive Dense Retriever...")
naive_ragas_results  = evaluate(
    dataset=naive_dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print("\nEvaluating Pipeline B: Hybrid Retriever...")
hybrid_ragas_results = evaluate(
    dataset=hybrid_dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print("\n=== RAGAS Results: Pipeline A (Naive) ===")
print(naive_ragas_results)

print("\n=== RAGAS Results: Pipeline B (Hybrid) ===")
print(hybrid_ragas_results)
```

---

## Summary Table

| Metric | Layer | Pipeline A (Naive) | Pipeline B (Hybrid) | Notes |
|---|---|---|---|---|
| Precision@5 | Retrieval | TBD | TBD | Hybrid expected to win |
| Precision@10 | Retrieval | TBD | TBD | Smaller gap than @5 |
| ContextEntityRecall | Retrieval | TBD | TBD | Repo names in context? |
| LLMContextRecall | Retrieval | TBD | TBD | Needs reference answers |
| Faithfulness | Generation | TBD | TBD | Primary metric — no hallucinations |
| ResponseRelevancy | Generation | TBD | TBD | Is the answer on-topic? |
| NoiseSensitivity | Generation | TBD | TBD | Noise from awesome-lists |
| FactualCorrectness | — | — | — | **Excluded** — wrong task type |

---

## Expected Findings

- **Precision@5 should improve more than Precision@10** from Naive → Hybrid, because Cohere
  reranking reshuffles the top of the list specifically.
- **Faithfulness should be high for both pipelines** if the system prompt grounding rules are
  working. A drop in faithfulness signals the model is ignoring the context instructions.
- **NoiseSensitivity** is the metric most likely to differ between pipelines: Cohere reranking
  filters irrelevant chunks before they reach the generator, which should reduce noise exposure.
- **ResponseRelevancy** should be consistently high since the queries are clear. A low score
  would indicate the model is being overly cautious ("I don't know") rather than using the
  retrieved context.

---

*Generated as part of the AskMyBookmark Certification Challenge project.*
