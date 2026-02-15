# Ragas Metrics Explained

## Overview

This document explains the six Ragas metrics used to evaluate RAG (Retrieval-Augmented Generation) systems in the assignment. Each metric measures a different aspect of RAG quality.

## Your Results Comparison

**Baseline System** (k=3, no reranking, chunk_size=50):
- Context Recall: 0.2024
- Faithfulness: 0.4258
- Factual Correctness: 0.4783
- Answer Relevancy: 0.5376
- Context Entity Recall: 0.3313
- Noise Sensitivity: 0.0394

**Reranked System** (k=20→5 with Cohere rerank, chunk_size=500):
- Context Recall: 0.7837
- Faithfulness: 0.6673
- Factual Correctness: 0.6367
- Answer Relevancy: 0.9422
- Context Entity Recall: 0.4778
- Noise Sensitivity: 0.1313

---

## 1. Context Recall (Retrieval Quality)

**What it measures**: Whether your retriever found all the relevant information needed to answer the question.

**How it works**: Compares the retrieved contexts against the ground truth reference contexts. It checks: "Did the retriever bring back all the necessary chunks?"

**Formula concept**:
```
Context Recall = (Relevant information in retrieved contexts) / (All relevant information needed)
```

**Score interpretation**:
- **1.0 = Perfect**: Retriever found all necessary information
- **0.5 = Partial**: Retriever found only half the relevant content
- **0.0 = Failed**: Retriever missed all relevant information

**Your results analysis**:
- Baseline: **0.2024** - Only ~20% of needed information was retrieved
- Reranked: **0.7837** - Retrieval improved to ~78% coverage
- **Why the improvement?**: Retrieving k=20 documents then reranking to top 5 gave much better coverage than just retrieving k=3 directly

**Example from your data**:
Question: "What exercises help with lower back pain?"
- Baseline retrieved k=3 chunks with chunk_size=50 (very small chunks)
- Response: "The provided context does not specify any particular exercises"
- **Low context recall** - didn't retrieve the relevant sections

- Reranked retrieved k=20, reranked to top 5, with chunk_size=500
- Response included: "Cat-Cow Stretch...Bird Dog..."
- **High context recall** - found the specific exercise descriptions

---

## 2. Faithfulness (Hallucination Detection)

**What it measures**: Whether the generated answer is grounded in the retrieved context, or if the LLM "hallucinated" information.

**How it works**: Breaks down the answer into individual claims, then checks if each claim can be verified from the retrieved contexts.

**Formula concept**:
```
Faithfulness = (Number of claims supported by context) / (Total number of claims in answer)
```

**Score interpretation**:
- **1.0 = Perfect**: Every claim in the answer comes from the context
- **0.5 = Concerning**: Half the claims are unverified/hallucinated
- **0.0 = Failed**: Nothing in the answer is supported by context

**Your results analysis**:
- Baseline: **0.4258** - About 58% of claims were unsupported (high hallucination)
- Reranked: **0.6673** - About 33% of claims were unsupported (still some hallucination)
- **Why the improvement?**: Better context = less need for LLM to "fill in gaps" with made-up info

**Example scenario**:
If the LLM says: "Exercise improves sleep quality and can cure insomnia"
- "Exercise improves sleep quality" ✓ (in context)
- "Can cure insomnia" ✗ (not in context, too strong a claim)
- Faithfulness = 1/2 = 0.5

**Important note**: This is critical for healthcare domains where hallucinated medical advice could be dangerous.

---

## 3. Factual Correctness (Accuracy)

**What it measures**: Whether the facts in the generated answer match the ground truth reference answer.

**How it works**: Compares the generated answer against the synthetic "reference" answer, checking for:
- Correct facts (TP: true positives)
- Incorrect facts (FP: false positives)
- Missing facts (FN: false negatives)

**Formula concept**:
```
Factual Correctness = F1 score of facts
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Score interpretation**:
- **1.0 = Perfect**: All facts correct, no errors, nothing important missing
- **0.5 = Moderate**: Some facts correct but has errors or omissions
- **0.0 = Failed**: No correct facts, or all facts wrong

**Your results analysis**:
- Baseline: **0.4783** - Many facts missing or incorrect
- Reranked: **0.6367** - More accurate but still room for improvement
- **Why the improvement?**: Better context → more accurate information → more factually correct answers

**Difference from Faithfulness**:
- **Faithfulness**: "Is the answer grounded in context?" (hallucination check)
- **Factual Correctness**: "Is the answer factually accurate?" (accuracy check)

An answer can be faithful (all claims from context) but factually incorrect (if the context itself is wrong or the LLM misinterprets it).

---

## 4. Answer Relevancy (Response Quality)

**What it measures**: How well the generated answer addresses the specific question asked.

**How it works**: Uses an LLM to judge whether the answer is:
- Directly relevant to the question
- Complete (not partial)
- Not verbose with unnecessary information
- On-topic

**Formula concept**:
```
Answer Relevancy = Semantic similarity between question and answer
(Also checks for completeness and conciseness)
```

**Score interpretation**:
- **1.0 = Perfect**: Answer directly and completely addresses the question
- **0.5 = Partial**: Answer is somewhat relevant but incomplete or verbose
- **0.0 = Failed**: Answer is off-topic or doesn't address the question

**Your results analysis**:
- Baseline: **0.5376** - Answers were somewhat relevant but incomplete
- Reranked: **0.9422** - Dramatic improvement, answers very relevant and complete
- **Why the improvement?**: Better context = more complete information = more relevant answers

**Example from your data**:
Question: "What exercises help with lower back pain?"

Baseline answer: "The provided context does not specify any particular exercises"
- Relevancy: LOW - Doesn't answer the question

Reranked answer: "Cat-Cow Stretch...alternate arching and sagging...Bird Dog...extending opposite arm and leg..."
- Relevancy: HIGH - Directly answers with specific exercises

---

## 5. Context Entity Recall (Entity Coverage)

**What it measures**: What proportion of important entities (names, concepts, techniques, etc.) from the ground truth are present in the retrieved context.

**How it works**:
1. Extracts named entities from the reference answer (e.g., "Cat-Cow Stretch", "Cognitive Behavioral Therapy")
2. Checks how many of those entities appear in the retrieved contexts
3. Calculates the percentage of entities found

**Formula concept**:
```
Context Entity Recall = (Entities in retrieved context) / (Entities in ground truth)
```

**Score interpretation**:
- **1.0 = Perfect**: All important entities from ground truth are in retrieved context
- **0.5 = Partial**: Only half the entities were retrieved
- **0.0 = Failed**: No relevant entities retrieved

**Your results analysis**:
- Baseline: **0.3313** - Missing ~67% of important entities
- Reranked: **0.4778** - Still missing ~52% of entities (moderate improvement)
- **Why modest improvement?**: Even with better retrieval, some specific entity names might be scattered across chunks

**Example**:
Question: "What is Cognitive Behavioral Therapy for Insomnia?"
Ground truth entities: ["Cognitive Behavioral Therapy", "CBT-I", "insomnia", "sleep restriction", "stimulus control"]

If retrieved context only mentions: ["Cognitive Behavioral Therapy", "insomnia"]
- Context Entity Recall = 2/5 = 0.4

**Difference from Context Recall**:
- **Context Recall**: "Did we retrieve the relevant semantic information?"
- **Context Entity Recall**: "Did we retrieve the specific named entities/concepts?"

---

## 6. Noise Sensitivity (Robustness)

**What it measures**: How well the system handles irrelevant or noisy information in the retrieved context.

**How it works**:
1. Tests if the LLM gets distracted by irrelevant chunks in the context
2. Checks if answer quality degrades when noise is present
3. Measures the model's ability to focus on relevant information

**Formula concept**:
```
Noise Sensitivity = Impact of irrelevant context on answer quality
(Lower is generally better, but interpretation depends on implementation)
```

**Score interpretation** (can vary by Ragas version):
- **Higher score may indicate**: System is sensitive to noise (gets distracted)
- **Lower score may indicate**: System filters noise well

**Your results analysis**:
- Baseline: **0.0394** - Very low, possibly because context was so sparse there wasn't much noise
- Reranked: **0.1313** - Higher, because retrieving k=20 introduces more potential noise
- **Trade-off**: More documents = better coverage BUT more noise to filter through

**Example scenario**:
Question: "What helps with lower back pain?"

Retrieved contexts include:
1. "Cat-Cow Stretch helps back pain" (RELEVANT)
2. "Progressive muscle relaxation for sleep" (NOISE - about sleep, not back pain)
3. "Bird Dog exercise for back pain" (RELEVANT)

A noise-sensitive system might incorrectly incorporate the sleep information. A robust system ignores irrelevant chunks.

---

## Metric Categories Summary

### **Retrieval Metrics** (measure retriever quality)
1. **Context Recall**: Did we retrieve all needed information?
2. **Context Entity Recall**: Did we retrieve all important entities?

### **Generation Metrics** (measure LLM output quality)
3. **Faithfulness**: Is the answer grounded in context (no hallucination)?
4. **Factual Correctness**: Is the answer factually accurate?
5. **Answer Relevancy**: Does the answer address the question?

### **Robustness Metrics** (measure system resilience)
6. **Noise Sensitivity**: Can the system handle irrelevant information?

---

## How These Metrics Work Together

A good RAG system needs:

1. **High Context Recall** + **High Context Entity Recall**
   - Retriever brings back all necessary information

2. **High Faithfulness**
   - LLM doesn't hallucinate beyond the context

3. **High Factual Correctness** + **High Answer Relevancy**
   - Answer is accurate and addresses the question

4. **Low/Managed Noise Sensitivity**
   - System handles irrelevant information gracefully

---

## Key Insights from Your Results

### What Worked
Reranking improved **Answer Relevancy** most dramatically (0.54 → 0.94):
- Retrieving more documents (k=20) then filtering to best 5
- Larger chunks (500 vs 50) preserved more context
- Chunk overlap (30) maintained continuity

### What Still Needs Work
**Faithfulness** (0.67) and **Factual Correctness** (0.64) have room for improvement:
- Consider prompt engineering to emphasize "only use context"
- May need better chunking strategy to preserve complete facts
- Could benefit from citation mechanisms

### The Trade-off
**Noise Sensitivity** increased (0.04 → 0.13):
- More documents = more potential noise
- But the benefits (better coverage) outweighed this cost
- Reranker helped filter, but some noise remains

---

## Practical Recommendations

1. **For healthcare/wellness apps**: Prioritize **Faithfulness** and **Factual Correctness** (can't have hallucinated medical advice)

2. **For user satisfaction**: Optimize **Answer Relevancy** (users want direct answers)

3. **For retrieval improvement**: Focus on **Context Recall** (can't answer well without right info)

4. **For production robustness**: Monitor **Noise Sensitivity** (real-world data is messy)

5. **Always track together**: A high score on one metric doesn't mean overall success - need balanced performance across all metrics.
