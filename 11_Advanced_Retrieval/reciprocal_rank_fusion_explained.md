# Reciprocal Rank Fusion (RRF) Algorithm

## Overview

Reciprocal Rank Fusion (RRF) is a data fusion algorithm used to combine results from multiple retrieval systems or rankers.

**Reference**: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. SIGIR.

## What Problem Does It Solve?

When combining results from multiple retrieval systems (e.g., BM25 + vector search + reranker), each system produces scores on different scales:
- BM25 might give scores from 0-100
- Vector similarity might give scores from 0-1
- Rerankers might use completely different scoring mechanisms

**The Challenge**: How do you fairly combine these different scoring systems?

**RRF's Solution**: Ignore the actual scores and only use the **rank positions** of documents in each retriever's results.

---

## The Algorithm

### Formula

```
RRF_score(document) = Σ [1 / (k + rank_i(document))]
```

**Where:**
- `rank_i(document)` = the position of the document in retriever `i`'s results (1, 2, 3, ...)
- `k` = a constant (typically 60) that prevents top-ranked items from dominating too heavily
- `Σ` = sum across all retrievers that returned this document

### Key Properties

1. **Scale-independent**: Only uses rank positions, not raw scores
2. **Simple**: No complex normalization or parameter tuning required
3. **Effective**: Documents that rank highly across multiple systems get higher RRF scores
4. **Fair**: No single retriever can dominate the final ranking

---

## Example

Suppose you have 3 retrievers and Document A appears in their results:

| Retriever | Document A's Rank Position |
|-----------|---------------------------|
| BM25 | 2nd |
| Vector Search | 5th |
| Reranker | 1st |

**Calculation** (with k=60):
```
RRF_score(A) = 1/(60+2) + 1/(60+5) + 1/(60+1)
             = 1/62 + 1/65 + 1/61
             ≈ 0.0161 + 0.0154 + 0.0164
             = 0.0479
```

If Document B only appears in one retriever at rank 1:
```
RRF_score(B) = 1/(60+1) = 1/61 ≈ 0.0164
```

**Result**: Document A (appearing in all 3 retrievers) gets a higher score than Document B (appearing in only 1), even though B was ranked 1st in its retriever.

---

## Use in Ensemble Retrieval (Task 9)

In the LangChain Ensemble Retriever, RRF is used to combine results from multiple retrievers:

```python
retriever_list = [
    bm25_retriever,
    naive_retriever,
    parent_document_retriever,
    compression_retriever,
    multi_query_retriever
]

ensemble_retriever = EnsembleRetriever(
    retrievers=retriever_list,
    weights=equal_weighting  # Optional weights for each retriever
)
```

**How it works:**
1. Each retriever returns its ranked list of documents
2. RRF calculates a score for each unique document based on its positions across all retrievers
3. Documents are re-ranked by their RRF scores
4. The top-k documents by RRF score are returned as the final result

**Benefits:**
- Leverages strengths of different retrieval methods (keyword-based, semantic, reranked, etc.)
- Documents that appear in multiple retrievers' results are promoted
- No need to tune score normalization parameters

---

## Why k=60?

The constant `k` (typically set to 60) serves to:
- Reduce the impact of small rank differences at the top
- Prevent a single high-ranking result from dominating
- Empirically found to work well across various domains

The choice of 60 is somewhat arbitrary but has been validated through experiments in the original paper.

---

## Advantages

1. **No score calibration needed**: Works with rankings alone
2. **Robust**: Less sensitive to individual retriever failures
3. **Improved recall**: Combines diverse retrieval strategies
4. **Simple implementation**: Easy to understand and implement

## Limitations

1. **Requires multiple retrievers**: Only useful when combining 2+ systems
2. **Computational cost**: Must run multiple retrievers
3. **Equal treatment**: Doesn't inherently weight retrievers by quality (though weights can be added)
