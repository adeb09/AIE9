# Question 2: Multi-Query Retrieval and Recall

## Question:
Explain how generating multiple reformulations of a user query can improve recall.

## Answer:

Generating multiple reformulations of a user query improves recall by casting a "wider net" during retrieval. Here's why:

### 1. Vocabulary Mismatch Problem
A single query might miss relevant documents that use different terminology. For example:
- **Original**: "What exercises help with lower back pain?"
- **Reformulation 1**: "Which physical activities reduce lumbar discomfort?"
- **Reformulation 2**: "Back strengthening movements for pain relief"

Each reformulation uses different keywords (exercises vs. activities vs. movements, pain vs. discomfort) that might match different relevant documents in the corpus.

### 2. Semantic Coverage
Each reformulation can emphasize different aspects of the same question, capturing documents that focus on those specific angles. This helps retrieve documents that might be relevant but weren't semantically similar to the original query phrasing.

### 3. Increased Retrieval Pool
Since each query retrieves its own set of documents, and then all unique documents are combined, you're more likely to retrieve ALL relevant documents that exist in the corpus.

**Impact on Recall:**
```
Recall = (Number of relevant documents retrieved) / (Total number of relevant documents in corpus)
```

Multi-query retrieval increases the numerator (relevant documents retrieved) because:
- Query 1 might find documents A, B, C
- Query 2 might find documents B, D, E
- Query 3 might find documents C, E, F
- **Combined result**: Documents A, B, C, D, E, F (6 documents instead of 3)

### 4. Compensating for Embedding Limitations
Single embedding vectors can only capture one representation of a concept. By generating multiple reformulations, we explore different semantic neighborhoods in the embedding space, increasing the likelihood of finding all relevant documents.

## Conclusion

In essence, multiple query reformulations compensate for the limitations of embedding similarity by exploring different ways to express the same information need, thereby increasing the likelihood of finding all relevant documents in the corpus and improving overall recall performance.
