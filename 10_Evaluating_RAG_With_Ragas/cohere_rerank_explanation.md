# CohereRerank Explanation

## Overview

**CohereRerank** is a reranking model from Cohere that improves retrieval quality by re-ordering documents based on their relevance to a query.

## How Reranking Works (High Level)

The reranking process follows a three-stage approach:

1. **Initial Retrieval**: First, the system retrieves a larger set of potentially relevant documents from your vector store (in the notebook example, 20 documents with `k=20`)

2. **Reranking**: The CohereRerank model then analyzes each document in the context of the specific query and assigns more accurate relevance scores. This is a more sophisticated (but slower) similarity measure than the initial vector similarity search.

3. **Compression**: Finally, the top N most relevant documents are selected based on the reranking scores (in the notebook example, the top 5 documents)

## Why Use Reranking?

The two-stage approach is beneficial because:

- **Vector similarity** (first stage) is fast but can miss semantic nuances
- **Reranking** (second stage) provides more accurate relevance scoring but is computationally expensive
- By using vector search to narrow down candidates and reranking to refine, you get both speed and accuracy

## Technical Architecture

### CohereRerank is a Machine Learning Model

Cohere's reranking model is a **cross-encoder** architecture that predicts the relevance score between a query and each document.

### How It Differs From Initial Vector Search

The key difference is in the **architecture**:

#### Initial Vector Search (Bi-Encoder)
- Query and documents are encoded **separately** into embeddings
- Similarity is computed via simple math (cosine similarity, dot product)
- Fast because embeddings are pre-computed and stored
- **Limitation**: query and document never "see" each other during encoding

#### Reranking (Cross-Encoder)
- Query and document are **concatenated together** and fed into the model simultaneously
- The model processes: `[QUERY] + [DOCUMENT]` as a single input
- Outputs a relevance score (typically 0-1)
- Much more accurate because the model can see the full interaction between query and document
- **Trade-off**: slower because you must run inference for every query-document pair

### Why Cross-Encoders Are Better

Cross-encoders can capture:
- **Semantic relationships** between specific words in the query and document
- **Context dependencies** - how the query terms relate to document terms
- **Nuanced relevance** that simple vector similarity misses

#### Example

If your query is "exercises for lower back pain" and you have documents about "Bird Dog exercise" vs "neck rolls":
- A cross-encoder can better understand that Bird Dog is more relevant because it processes the relationship: "Bird Dog" → "core strengthening" → "lower back pain relief" by analyzing both the query and document together
- The bi-encoder only compares pre-computed embeddings and may miss these nuanced connections

## Implementation in the Notebook

In `Evaluating_RAG_Assignment.ipynb` (cell 55), CohereRerank is used within a `ContextualCompressionRetriever`:

```python
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

def retrieve_adjusted(state):
  compressor = CohereRerank(model="rerank-v3.5")
  compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=adjusted_example_retriever,
    search_kwargs={"k": 5}
  )
  retrieved_docs = compression_retriever.invoke(state["question"])
  return {"context" : retrieved_docs}
```

This takes 20 initially retrieved documents and compresses them down to the 5 most relevant ones for the question.

## Performance Impact

The results in the notebook demonstrate the effectiveness of reranking:
- **Context Recall**: improved from 0.2024 to 0.9630
- **Faithfulness**: improved from 0.4258 to 0.7518
- **Factual Correctness**: improved from 0.4783 to 0.7267
- **Answer Relevancy**: improved from 0.5376 to 0.9521

This dramatic improvement shows how effective reranking can be for RAG applications!
