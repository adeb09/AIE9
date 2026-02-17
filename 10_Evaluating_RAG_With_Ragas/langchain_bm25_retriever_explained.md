# LangChain BM25Retriever: BM25Plus and Preprocessing Explained

## Overview

This document explains two important features of LangChain's `BM25Retriever`:
1. **BM25Plus**: An enhanced variant that handles short documents better
2. **preprocess_func**: Custom preprocessing/tokenization for better retrieval

Based on: https://docs.langchain.com/oss/python/integrations/retrievers/bm25

---

## What is BM25Plus?

### The Problem BM25Plus Solves

Standard BM25 (also called **BM25Okapi**) has a known issue: **it can be biased against short documents**.

#### Why Standard BM25 Penalizes Short Documents

Recall the BM25 formula's length normalization component:
```
Length_factor = (1 - b + b × |D| / avgdl)
```

Where:
- `|D|` = document length
- `avgdl` = average document length
- `b` = length normalization parameter (typically 0.75)

**Problem scenario**:
```
Corpus has average document length: 500 words

Document A (short): 50 words, contains query term "mitochondria" 2 times
Document B (long): 500 words, contains query term "mitochondria" 2 times

Standard BM25 with b=0.75:
- Doc A length factor: 1 - 0.75 + 0.75 × (50/500) = 0.325 → STRONG BOOST
- Doc B length factor: 1 - 0.75 + 0.75 × (500/500) = 1.0 → NEUTRAL

So far so good - short doc gets boosted!

But here's the issue with VERY SHORT documents:
Document C (very short): 10 words, contains NO query terms
- Length factor: 1 - 0.75 + 0.75 × (10/500) = 0.265
- Even with NO matches, the denominator is very small
- This can cause instability in scoring
```

Additionally, in standard BM25, if the IDF is negative (term appears in most documents), the contribution can be negative, effectively penalizing documents for containing common terms.

### BM25Plus: The Solution

**BM25Plus** makes two key improvements:

#### 1. **Matched Terms Always Contribute Positively**

Standard BM25:
```
Score contribution can be negative if IDF is negative (common terms)
```

BM25Plus:
```
Score contribution is ALWAYS ≥ 0, even for common terms
Adds a small positive delta (δ) to ensure non-negative contributions
```

#### 2. **Better Handling of Short Documents**

The BM25Plus formula adds a **delta (δ)** parameter that provides a baseline positive score:

```
BM25Plus(D, Q) = Σ IDF(qᵢ) × ((f(qᵢ, D) × (k₁ + 1)) / (f(qᵢ, D) + k₁ × (1 - b + b × |D| / avgdl)) + δ)
                                                                                                    ↑
                                                                                           Added delta term
```

The **delta (δ)** parameter (typically 0.5 or 1.0):
- Ensures every matched term contributes at least δ to the score
- Reduces bias against very short documents
- Improves recall for short, relevant documents

---

## When to Use BM25 vs BM25Plus

### Use Standard BM25 (BM25Okapi) When:

✅ **Long-form documents**
- Blog posts, articles, research papers
- Documents are typically 500+ words

✅ **Uniform document lengths**
- All documents are roughly the same length
- Length normalization works consistently

✅ **Precision-focused**
- You want to prioritize highly relevant long documents
- False positives are more costly than false negatives

✅ **Traditional search engines**
- Web page search where documents are substantial

### Use BM25Plus When:

✅ **Short documents or chunks**
- RAG systems with chunked text (100-500 words per chunk)
- Social media posts, tweets, comments
- Product descriptions, FAQ entries

✅ **Variable document lengths**
- Corpus has mix of short and long documents
- Want to avoid unfairly penalizing short documents

✅ **Recall-focused**
- You want to catch all potentially relevant documents
- Using a reranker in a second stage (like in your assignment!)
- False negatives are more costly than false positives

✅ **Chunked RAG systems** (Most Common Use Case!)
- When you split documents into chunks for retrieval
- This is exactly your use case in the assignment!

---

## BM25Plus in Your RAG Assignment Context

In your assignment, you had:
```python
# Baseline
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
# Very short chunks!

# Improved
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
# Better sized chunks
```

If you were using BM25 retrieval instead of dense embeddings, **BM25Plus would be the better choice** because:
1. You're working with chunks, not full documents
2. Chunks have variable lengths (50-500 characters)
3. You want high recall (retrieve 20, rerank to 5)
4. Short but relevant chunks shouldn't be penalized

---

## The `preprocess_func` Parameter

### What It Does

The `preprocess_func` parameter allows you to specify **how to tokenize and preprocess text** before BM25 scoring.

**Default behavior** (if not specified):
```python
# Simple whitespace splitting
"Lower back pain exercises".split()
# → ["Lower", "back", "pain", "exercises"]
```

**Problem**: This is very naive and doesn't handle:
- Punctuation: "exercises." vs "exercises"
- Case sensitivity: "Lower" vs "lower"
- Contractions: "don't" as one token
- Special characters: "COVID-19" handling

### Why Preprocessing Matters

```
Query: "What are lower-back pain exercises?"
Document: "Lower back pain: exercises, stretches & treatments."

Without preprocessing:
Query tokens:    ["What", "are", "lower-back", "pain", "exercises?"]
Document tokens: ["Lower", "back", "pain:", "exercises,", "stretches", "&", "treatments."]

Matches:
- "lower-back" ≠ "Lower" ❌
- "lower-back" ≠ "back" ❌
- "pain" ≠ "pain:" ❌
- "exercises?" ≠ "exercises," ❌

Result: Very few matches, poor score!

With preprocessing (word_tokenize + lowercase):
Query tokens:    ["what", "are", "lower", "back", "pain", "exercises"]
Document tokens: ["lower", "back", "pain", "exercises", "stretches", "treatments"]

Matches:
- "lower" = "lower" ✅
- "back" = "back" ✅
- "pain" = "pain" ✅
- "exercises" = "exercises" ✅

Result: Much better matching!
```

---

## Common Preprocessing Functions

### 1. NLTK's word_tokenize (Example from Documentation)

```python
from nltk.tokenize import word_tokenize
from langchain_community.retrievers import BM25Retriever

retriever = BM25Retriever.from_documents(
    docs,
    k=2,
    preprocess_func=word_tokenize
)
```

**What word_tokenize does**:
```python
text = "Don't worry about COVID-19's impact. Dr. Smith's advice: exercise 30min/day."

word_tokenize(text)
# Output:
['Do', "n't", 'worry', 'about', 'COVID-19', "'s", 'impact', '.',
 'Dr.', 'Smith', "'s", 'advice', ':', 'exercise', '30min/day', '.']
```

**Features**:
- ✅ Handles contractions ("don't" → "Do" + "n't")
- ✅ Preserves important punctuation in terms (COVID-19, Dr.)
- ✅ Separates punctuation as tokens
- ⚠️ Still case-sensitive (you'd need to lowercase separately)

### 2. Custom Preprocessing Function

You can create your own preprocessing function:

```python
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def custom_preprocess(text):
    """
    Custom preprocessing for health/wellness domain
    """
    # Lowercase
    text = text.lower()

    # Remove special characters but keep important ones
    # Keep hyphens in medical terms like "COVID-19"
    text = re.sub(r'[^\w\s-]', ' ', text)

    # Tokenize on whitespace
    tokens = text.split()

    # Optional: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    # Optional: Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens

# Use it
retriever = BM25Retriever.from_documents(
    docs,
    k=5,
    preprocess_func=custom_preprocess
)
```

**Example**:
```python
text = "Running exercises help with lower back pain relief."

custom_preprocess(text)
# Output:
['run', 'exercis', 'help', 'lower', 'back', 'pain', 'relief']

# Note:
# - "Running" → "run" (stemmed)
# - "exercises" → "exercis" (stemmed)
# - "with" removed (stopword)
```

### 3. Simple Lowercase + Split

```python
def simple_preprocess(text):
    return text.lower().split()

retriever = BM25Retriever.from_documents(
    docs,
    k=5,
    preprocess_func=simple_preprocess
)
```

**When to use**:
- Quick prototyping
- Clean text without much punctuation
- When you want full control over tokenization

---

## Complete LangChain Code Examples

### Example 1: Basic BM25Retriever

```python
from langchain_community.retrievers import BM25Retriever

# Simple text list
texts = ["foo", "bar", "world", "hello", "foo bar"]

retriever = BM25Retriever.from_texts(texts)

# Query
result = retriever.invoke("foo")
print(result)
# Returns documents containing "foo"
```

### Example 2: With Documents and Preprocessing

```python
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from nltk.tokenize import word_tokenize

# Create documents (like from your TextLoader)
docs = [
    Document(page_content="Lower back pain exercises include cat cow stretch"),
    Document(page_content="Neck pain relief through gentle stretches"),
    Document(page_content="Lower back strengthening with planks"),
]

# Create retriever with preprocessing
retriever = BM25Retriever.from_documents(
    docs,
    k=2,
    preprocess_func=word_tokenize
)

# Query
results = retriever.invoke("lower back pain")
for doc in results:
    print(doc.page_content)
```

**Output**:
```
Lower back pain exercises include cat cow stretch
Lower back strengthening with planks
```

### Example 3: BM25Plus with Custom Parameters

```python
from langchain_community.retrievers import BM25Retriever

# Using BM25Plus variant
retriever = BM25Retriever.from_documents(
    docs,
    k=5,
    bm25_variant="plus",
    bm25_params={"delta": 0.5}  # Delta parameter for BM25Plus
)

results = retriever.invoke("exercise")
```

**Parameters explained**:
- `k=5`: Return top 5 results
- `bm25_variant="plus"`: Use BM25Plus instead of standard BM25
- `bm25_params={"delta": 0.5}`: Set delta value for BM25Plus

### Example 4: Complete RAG Pipeline with BM25Plus

```python
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from nltk.tokenize import word_tokenize

# 1. Load documents
loader = TextLoader("data/HealthWellnessGuide.txt")
docs = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=30
)
split_docs = text_splitter.split_documents(docs)

# 3. Create BM25Plus retriever with preprocessing
retriever = BM25Retriever.from_documents(
    split_docs,
    k=5,
    bm25_variant="plus",
    bm25_params={"delta": 1.0},
    preprocess_func=word_tokenize
)

# 4. Retrieve relevant documents
query = "What exercises help with lower back pain?"
retrieved_docs = retriever.invoke(query)

# 5. Generate answer with LLM
llm = ChatOpenAI(model="gpt-4o-mini")
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.

Context: {context}

Question: {question}
""")

response = llm.invoke(prompt.format(context=context, question=query))
print(response.content)
```

---

## Comparing Retrieval Approaches in Your Assignment

### What You Used (Dense Embeddings)

```python
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = QdrantVectorStore(...)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

**Pros**:
- Semantic understanding (synonyms, paraphrasing)
- Works well with natural language queries
- No preprocessing needed

**Cons**:
- Requires embedding model (cost, API calls)
- Less interpretable
- Slower than BM25

### Alternative: BM25Plus (Sparse)

```python
from langchain_community.retrievers import BM25Retriever

retriever = BM25Retriever.from_documents(
    split_docs,
    k=3,
    bm25_variant="plus",
    bm25_params={"delta": 0.5},
    preprocess_func=word_tokenize
)
```

**Pros**:
- No API calls or embedding costs
- Very fast (inverted index)
- Interpretable (see which terms matched)
- Good for keyword-based queries

**Cons**:
- No semantic understanding
- Needs preprocessing
- Vocabulary mismatch problem

### Best: Hybrid Approach

```python
from langchain.retrievers import EnsembleRetriever

# BM25Plus retriever
bm25_retriever = BM25Retriever.from_documents(
    split_docs,
    k=10,
    bm25_variant="plus",
    preprocess_func=word_tokenize
)

# Dense retriever
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Combine both
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.3, 0.7]  # 30% BM25, 70% dense
)

# Get best of both worlds
results = ensemble_retriever.invoke(query)
```

This gives you:
- Keyword precision from BM25Plus
- Semantic recall from dense embeddings
- Better overall performance

---

### Understanding the Weights Parameter in EnsembleRetriever

The `weights` parameter controls **how much each retriever contributes to the final ranking**. This is a critical parameter for tuning hybrid retrieval performance.

#### How It Works

1. **Each retriever produces scores** for documents:
   - BM25 retriever: Returns documents with BM25 scores
   - Dense retriever: Returns documents with similarity scores

2. **Scores are normalized** (typically to 0-1 range)

3. **Weighted combination** is computed:
   ```
   Final_Score(doc) = weight[0] × BM25_score(doc) + weight[1] × Dense_score(doc)
   ```

4. **Documents are re-ranked** by the final combined score

#### Practical Example

```python
Query: "lower back pain exercises"

BM25 retriever (weight=0.3) returns:
- Doc A: BM25 score = 8.5 → normalized = 1.0
- Doc B: BM25 score = 6.2 → normalized = 0.73
- Doc C: BM25 score = 4.1 → normalized = 0.48

Dense retriever (weight=0.7) returns:
- Doc B: similarity = 0.92
- Doc D: similarity = 0.88
- Doc A: similarity = 0.65

Final scores:
- Doc A: (0.3 × 1.0) + (0.7 × 0.65) = 0.30 + 0.46 = 0.76
- Doc B: (0.3 × 0.73) + (0.7 × 0.92) = 0.22 + 0.64 = 0.86  ← Winner!
- Doc C: (0.3 × 0.48) + (0.7 × 0) = 0.14
- Doc D: (0.3 × 0) + (0.7 × 0.88) = 0.62

Final Ranking: B > A > D > C
```

**Key insight**: Doc B wins because it ranks well in BOTH retrievers, even though it's not #1 in either individually. This is the power of ensemble retrieval!

#### Choosing Weights: A Guide

**Equal Weight: [0.5, 0.5]**
```python
weights=[0.5, 0.5]  # Equal importance
```
- Use when both retrievers are equally important
- Good starting point for experimentation
- Neither keyword nor semantic has priority

**Favor Dense (Semantic): [0.3, 0.7]**
```python
weights=[0.3, 0.7]  # 30% BM25, 70% dense
```
- Use when semantic understanding matters more
- Good for natural language questions
- Example: "What are some ways to improve sleep quality?"
- Dense embeddings handle paraphrasing and synonyms better

**Favor BM25 (Keyword): [0.7, 0.3]**
```python
weights=[0.7, 0.3]  # 70% BM25, 30% dense
```
- Use when exact keyword matching is important
- Good for technical/domain-specific terms
- Example: "CBT-I treatment protocol" (specific medical term)
- Example: "COVID-19 symptoms" (exact term matching)

**Heavy Skew: [0.2, 0.8] or [0.8, 0.2]**
```python
weights=[0.2, 0.8]  # Heavily favor dense
```
- When one retriever is clearly superior
- Still gets benefit from the other retriever's perspective
- Use if evaluation shows one method dominates

**Multiple Retrievers: [0.2, 0.3, 0.5]**
```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever, hybrid_retriever],
    weights=[0.2, 0.3, 0.5]
)
```
- Can combine more than 2 retrievers
- Weights should sum to 1.0 for interpretability

#### Weight Selection Strategy

**Step 1: Start with equal weights**
```python
weights=[0.5, 0.5]
```

**Step 2: Run evaluation on your test set**
```python
# Evaluate with different weight configurations
weight_configs = [
    [0.5, 0.5],
    [0.3, 0.7],
    [0.7, 0.3],
    [0.4, 0.6],
    [0.6, 0.4],
]

for weights in weight_configs:
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=weights
    )
    # Run your Ragas evaluation
    results = evaluate(dataset, retriever=ensemble_retriever)
    print(f"Weights {weights}: {results}")
```

**Step 3: Optimize based on your metrics**
- If **Context Recall** is low: Try favoring the retriever with better coverage
- If **Answer Relevancy** is low: Try favoring the retriever with better precision
- If **Factual Correctness** is low: Try favoring BM25 for exact term matches

**Step 4: Domain-specific tuning**
For health/wellness domain:
- Medical terms (like "CBT-I", "mitochondria") → Favor BM25 (exact matching)
- General questions (like "how to sleep better") → Favor dense (semantic)
- Consider: `[0.4, 0.6]` as a balanced compromise

#### Key Points to Remember

✅ **Weights should sum to 1.0** for interpretability (though not strictly required)

✅ **Order matters**: `weights[i]` corresponds to `retrievers[i]`
```python
retrievers=[bm25_retriever, dense_retriever]
weights=[0.3, 0.7]
         ↑    ↑
    BM25: 30%  Dense: 70%
```

✅ **Tune based on evaluation metrics**: Use your Ragas metrics to guide weight selection

✅ **Different queries may benefit from different weights**: Some queries are better for keyword matching, others for semantic matching

❌ **Don't use extreme weights like [0.01, 0.99]**: You lose the benefit of ensemble at that point

❌ **Don't assume default weights are optimal**: Always experiment with your specific dataset

#### Analogy to Recommender Systems

This is similar to **ensemble methods in recommender systems**:

| Recommender Systems | Hybrid Retrieval |
|---------------------|------------------|
| Collaborative filtering | BM25 (term co-occurrence patterns) |
| Content-based filtering | Dense embeddings (semantic content) |
| Hybrid recommender | EnsembleRetriever |
| Blend weights | weights parameter |

Just like you might do:
```python
final_score = 0.7 × collaborative_score + 0.3 × content_score
```

EnsembleRetriever does:
```python
final_score = 0.3 × bm25_score + 0.7 × dense_score
```

#### Example: Experimenting with Weights in Your Assignment

```python
from ragas import evaluate

# Test different weight configurations
results_comparison = []

for bm25_weight, dense_weight in [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]:
    # Create ensemble with these weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[bm25_weight, dense_weight]
    )

    # Run evaluation
    for test_row in dataset:
        response = graph.invoke(
            {"question": test_row.eval_sample.user_input},
            retriever=ensemble_retriever
        )
        test_row.eval_sample.response = response["response"]
        test_row.eval_sample.retrieved_contexts = [
            context.page_content for context in response["context"]
        ]

    # Evaluate
    eval_result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), AnswerRelevancy()],
        llm=evaluator_llm
    )

    results_comparison.append({
        "weights": f"BM25:{bm25_weight}, Dense:{dense_weight}",
        "context_recall": eval_result["context_recall"],
        "faithfulness": eval_result["faithfulness"],
        "answer_relevancy": eval_result["answer_relevancy"]
    })

# Compare results
import pandas as pd
pd.DataFrame(results_comparison)
```

**Expected insights**:
- **[0.3, 0.7]**: May have better answer relevancy (semantic understanding)
- **[0.7, 0.3]**: May have better faithfulness (exact term matching)
- **[0.5, 0.5]**: May offer best balance for your specific domain

#### Summary

The `weights` parameter is a powerful tuning knob that lets you:
1. **Balance keyword precision with semantic recall**
2. **Adapt to your specific domain and query patterns**
3. **Optimize for the metrics that matter most to your use case**

Think of it as a dial where:
- **More weight to BM25**: Emphasizes exact term matching, technical precision
- **More weight to dense**: Emphasizes semantic understanding, natural language
- **Balanced weights**: Gets benefits of both approaches

Always evaluate with your specific dataset to find the optimal weights!

---

## How Your Assignment Could Use BM25Plus

### Current Architecture (Dense Only)

```
Query → OpenAI Embeddings → QDrant Vector Search (k=20)
      → Cohere Rerank → Top 5 → LLM Generation
```

### Alternative: Hybrid with BM25Plus

```
Query → Branch:
        ├─ BM25Plus (k=10, fast keyword matching)
        └─ Dense Embeddings (k=10, semantic matching)
      → Merge results (20 total)
      → Cohere Rerank → Top 5
      → LLM Generation
```

**Benefits**:
1. Catches keyword matches that embeddings might miss
2. Catches semantic matches that BM25 might miss
3. Reranker gets best candidates from both approaches
4. Often improves metrics like Context Recall and Answer Relevancy

---

## Key Parameters Summary

### BM25Retriever Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 4 | Number of documents to return |
| `bm25_variant` | str | "okapi" | "okapi" for standard BM25, "plus" for BM25Plus |
| `bm25_params` | dict | {} | Variant-specific parameters (e.g., {"delta": 0.5}) |
| `preprocess_func` | callable | None | Custom tokenization/preprocessing function |

### BM25Plus Specific Parameters

| Parameter | Type | Typical Value | Description |
|-----------|------|---------------|-------------|
| `delta` | float | 0.5 - 1.0 | Baseline positive contribution for matched terms |
| `k1` | float | 1.5 | Term frequency saturation (inherited from BM25) |
| `b` | float | 0.75 | Length normalization (inherited from BM25) |

---

## Preprocessing Function Best Practices

### 1. Match Query and Document Preprocessing

**CRITICAL**: Your preprocessing function will be applied to BOTH queries and documents.

```python
# This will be called on:
# - Each document during indexing
# - Each query during retrieval

def preprocess(text):
    # Must be consistent!
    return text.lower().split()
```

### 2. Consider Your Domain

**Health/Wellness**:
```python
def health_preprocess(text):
    text = text.lower()
    # Keep medical terms intact
    # "COVID-19" should stay as one token
    # "CBT-I" should stay as one token
    return word_tokenize(text)
```

**Code/Technical**:
```python
def code_preprocess(text):
    # Keep camelCase, snake_case intact
    # Don't lowercase (case matters in code)
    return text.split()
```

### 3. Test Your Preprocessing

```python
def test_preprocessing():
    test_cases = [
        "What is COVID-19?",
        "Dr. Smith's advice",
        "lower-back pain",
    ]

    for text in test_cases:
        tokens = preprocess_func(text)
        print(f"{text} → {tokens}")
```

### 4. Balance Preprocessing Aggressiveness

**Aggressive preprocessing** (stemming, stopword removal):
- ✅ Better recall (more matches)
- ❌ Less precision (matches may be less relevant)

**Light preprocessing** (just lowercase + tokenize):
- ✅ Better precision (exact matches)
- ❌ Less recall (might miss relevant documents)

For RAG with reranking (like your assignment): **Prefer aggressive preprocessing** → High recall in first stage → Reranker handles precision

---

## Practical Recommendations

### For Your Assignment Context

1. **Use BM25Plus** if switching to sparse retrieval:
   - You have chunked documents (500 chars)
   - Variable chunk sizes
   - Want high recall for reranking

2. **Use word_tokenize as preprocess_func**:
   - Good balance of sophistication and speed
   - Handles punctuation well
   - Works well for health/wellness text

3. **Consider Hybrid Retrieval**:
   - Combine BM25Plus + Dense embeddings
   - 30% BM25 + 70% Dense is a good starting point
   - Then rerank with Cohere

4. **Tune delta parameter**:
   - Start with delta=0.5
   - Increase to 1.0 if short chunks are under-represented
   - Decrease to 0.25 if getting too many irrelevant short chunks

---

## Code Template for Your Assignment

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from nltk.tokenize import word_tokenize

# Load and split documents (your existing code)
split_documents = ...  # Your chunks

# Create BM25Plus retriever
bm25_retriever = BM25Retriever.from_documents(
    split_documents,
    k=10,
    bm25_variant="plus",
    bm25_params={"delta": 0.5, "k1": 1.5, "b": 0.75},
    preprocess_func=word_tokenize
)

# Create dense retriever (your existing code)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Combine with ensemble
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.3, 0.7]
)

# Use in your graph
def retrieve_hybrid(state):
    retrieved_docs = ensemble_retriever.invoke(state["question"])
    return {"context": retrieved_docs}
```

---

## Summary

### BM25Plus
- **What**: Enhanced BM25 variant for short documents
- **Key feature**: Matched terms always contribute positively
- **When to use**: Chunked RAG systems, variable document lengths
- **Parameter**: `delta` (typically 0.5-1.0) ensures positive contributions

### preprocess_func
- **What**: Custom tokenization/preprocessing for BM25
- **Key feature**: Applied to both queries and documents
- **Common choice**: `word_tokenize` from NLTK
- **Purpose**: Better term matching through normalization

### Recommendation for RAG
Use hybrid retrieval with:
- BM25Plus for keyword precision
- Dense embeddings for semantic recall
- Preprocessing for better matching
- Reranking for final precision

This combination often outperforms either approach alone!
