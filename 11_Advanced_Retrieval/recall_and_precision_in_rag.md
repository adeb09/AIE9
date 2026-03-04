# Recall and Precision in RAG

## Recall

**Recall** measures the proportion of relevant documents that were successfully retrieved out of all the relevant documents that exist in your corpus.

**Formula:**
```
Recall = (Number of relevant documents retrieved) / (Total number of relevant documents in corpus)
```

**Example:**
- Your corpus has 10 documents that are relevant to a query
- Your retrieval system retrieves 7 of those relevant documents (and maybe some irrelevant ones too)
- **Recall = 7/10 = 0.7 or 70%**

**Key Point:**
- Recall measures **completeness** - of all relevant docs that exist, how many did I find?

---

## Precision

**Precision** measures the proportion of retrieved documents that are actually relevant out of all the documents you retrieved.

**Formula:**
```
Precision = (Number of relevant documents retrieved) / (Total number of documents retrieved)
```

**Example:**
- Your retrieval system returns 10 documents for a query
- Out of those 10, only 6 are actually relevant to the query
- **Precision = 6/10 = 0.6 or 60%**

**Key Point:**
- Precision measures **quality** - of everything I retrieved, how much of it was actually useful?

---

## Side-by-Side Comparison

```
Recall:     Relevant Retrieved / All Relevant in Corpus
            (Did I find everything I should have?)

Precision:  Relevant Retrieved / All Retrieved
            (Is what I found actually useful?)
```

---

## Complete Example

**Scenario:**
- Corpus has 10 relevant docs for your query
- You retrieve 15 docs total
- 7 of those 15 are relevant

**Metrics:**
- **Recall = 7/10 = 70%** (found 7 out of 10 relevant docs)
- **Precision = 7/15 = 47%** (only 7 out of 15 retrieved docs were relevant)

---

## Trade-offs in RAG Systems

- **High Precision, Low Recall**: You retrieve only very relevant docs, but miss many others (too conservative)
- **High Recall, Low Precision**: You retrieve all relevant docs, but also lots of noise (too aggressive)

**Goal:** In RAG systems, you need to balance both - you want to retrieve enough relevant context (recall) without overwhelming the LLM with irrelevant information (precision).
