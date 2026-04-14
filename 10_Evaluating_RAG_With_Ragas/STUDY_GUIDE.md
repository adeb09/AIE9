# Session 10: Evaluating RAG & Agents with Ragas
## Interview-Ready Study Guide — Senior/Staff AI Engineer

---

## A. Core Concept Summary

RAG evaluation is a multi-dimensional problem: you must independently assess retrieval quality, generation faithfulness, and end-to-end answer relevance — because failures compound in non-obvious ways. Ragas operationalizes this by decomposing evaluation into orthogonal metrics, each targeting a distinct failure mode in the pipeline, analogous to how NDCG, Precision@K, and Recall@K each isolate different aspects of a ranker's behavior. The critical mental model is that **a high aggregate score can mask a catastrophically failing subsystem** — a pipeline with perfect faithfulness but poor context_recall is hallucination-safe but retrieval-broken, and these require completely different fixes. For agent evaluation, the problem space expands to include tool selection correctness and goal completion, which require reference trajectories rather than reference answers. LLM-as-judge patterns underpin most of these metrics, making calibration, cost, and circular evaluation failure modes first-class concerns in production systems.

---

## B. Key Terms & Definitions

- **Faithfulness**: A Ragas metric measuring whether every factual claim in the generated answer is grounded in the retrieved context. Computed via LLM decomposition of the answer into atomic claims, followed by NLI-style verification of each claim against the context.

- **Answer Relevancy**: A Ragas metric measuring how well the answer addresses the user's question, regardless of factual accuracy. Computed by using an LLM to reverse-generate `n` synthetic questions from the answer, then computing mean cosine similarity between those synthetic questions and the original question embedding.

- **Context Recall**: The fraction of ground-truth answer sentences attributable to at least one retrieved context chunk. Requires a `reference` (ground truth answer) and measures retrieval completeness — the recall side of the retrieval IR curve.

- **Context Precision**: A ranked metric assessing whether relevant context chunks appear early in the retrieved list. Computed as a weighted precision-at-K across retrieved chunks, analogous to Average Precision in search evaluation.

- **LLM-as-Judge**: A pattern where a capable LLM (e.g., GPT-4) serves as an automated evaluator, scoring or comparing model outputs on rubrics. Subject to position bias, verbosity bias, and circular evaluation failure when the judge and the system under test share the same model family.

- **Cross-Encoder (Reranker)**: A transformer that jointly encodes a query and a document in a single forward pass to produce a relevance score. More accurate than bi-encoder (embedding) retrieval because it models token-level query-document interactions, but O(n) at inference time rather than ANN-indexed O(log n).

- **EvaluationDataset (Ragas)**: The structured input format for `ragas.evaluate()`. Requires `user_input` (query), `retrieved_contexts` (list of strings), and `response` (generated answer). The `reference` field is optional but required for metrics needing ground truth (context_recall, answer_correctness).

- **ToolCallAccuracy**: An agent evaluation metric measuring whether the agent invoked the correct tools with correct arguments, relative to a reference tool-call sequence. Combines tool name matching and argument similarity.

- **AgentGoalAccuracy**: A binary or graded metric assessing whether an agent achieved the intended end-state goal, evaluated by LLM-as-judge against a stated goal description.

- **Circular Evaluation**: The failure mode where the same model (or model family) is used both to generate outputs and to judge them, producing systematically inflated scores. Structurally equivalent to using a ranker's own scores as relevance labels for its offline evaluation.

---

## C. How It Works — Technical Mechanics

### Ragas Metric Mechanics

| Metric | What It Measures | Computation Method | Low Score Implies | How to Fix |
|---|---|---|---|---|
| **faithfulness** | Are all answer claims grounded in retrieved context? | LLM decomposes answer → atomic claims; LLM judges each claim as supported/not-supported by context; score = supported_claims / total_claims | Answer contains hallucinated or extrapolated content not present in retrieved docs | Improve generation prompt to constrain to context; add citation-forcing; use smaller, more instruction-following model |
| **answer_relevancy** | Does the answer address the question asked? | LLM generates `n` reverse questions from the answer; mean cosine similarity of reverse question embeddings vs. original question embedding | Answer is technically accurate but off-topic, verbose, or tangential | Improve generation prompt with explicit instruction to answer the question directly; filter retrieved context for relevance pre-generation |
| **context_recall** | Did retrieval surface all content needed to answer? | Each sentence in the ground-truth reference is classified as attributable to a retrieved chunk or not; score = attributable_sentences / total_reference_sentences | Retrieval missed key chunks; chunking is too fine-grained or embedding model has low recall on this domain | Increase top-K retrieval; improve chunking strategy; retrain/swap embedding model; add HyDE or query expansion |
| **context_precision** | Are relevant chunks ranked above irrelevant ones? | Weighted precision-at-K: for each position k, computes whether chunk k is relevant (via LLM judge), weighted by 1/k; averaged across all positions | Relevant chunks exist but are buried deep in retrieved list; reranker is absent or miscalibrated | Add or retune reranker (e.g., Cohere Rerank); tune retrieval scoring; use MMR to diversify and promote relevant chunks |
| **answer_correctness** | Is the answer factually correct vs. ground truth? | F1-style blend of semantic similarity and factual overlap between response and reference | End-to-end pipeline correctness is low regardless of retrieval quality | Diagnose via faithfulness + context_recall decomposition to isolate whether fault is in retrieval or generation |

### Faithfulness — Deep Mechanics

```
answer → [LLM] → [claim_1, claim_2, ..., claim_n]
for each claim_i:
    [LLM] → verdict: {supported, not_supported} given context
faithfulness = count(supported) / n
```

The decomposition step is critical: without it, the judge tends to score holistically, conflating fluency with groundedness. This is an NLI-style task (textual entailment) framed as a prompted LLM call — Ragas can also be configured with a dedicated NLI model for cost reduction.

### Answer Relevancy — Deep Mechanics

```
response → [LLM] → [q̂_1, q̂_2, ..., q̂_n]  # reverse-generated questions
embeddings: e_original = embed(user_input)
            e_i = embed(q̂_i) for each i
answer_relevancy = mean(cosine(e_original, e_i))
```

Key insight: this is a **coverage proxy**, not a factuality measure. An answer that is highly specific and on-topic will produce reverse questions that closely match the original. A verbose, hedged, or tangential answer will produce dispersed reverse questions with lower similarity.

### Cohere Rerank — Cross-Encoder Mechanics

Bi-encoder retrieval (ANN search over pre-computed embeddings) runs in O(log n) but encodes query and document **independently** — the model never sees token-level query-document interactions. Cross-encoders concatenate `[CLS] query [SEP] document [SEP]` and run a full transformer forward pass, producing a calibrated relevance score that can model exact keyword match, negation, and semantic overlap simultaneously.

The standard pipeline:

```
query → bi-encoder ANN → top-K candidates (e.g., K=50)
         → co.rerank(query, top-K) → reranked top-k (e.g., k=5)
         → LLM generation
```

The rerank step is O(K) cross-encoder inference calls, which is why K must be bounded. Cohere Rerank v3 operates at ~100ms for K=50, making it viable for <500ms SLA pipelines when parallelized.

### Agent Evaluation Metrics

**ToolCallAccuracy**: Requires a `reference_tool_calls` field containing the expected sequence of tool invocations with expected arguments. Ragas computes:
- Tool name match: exact string match or fuzzy
- Argument similarity: semantic or exact, depending on argument type
- Sequence order: partial credit possible for correct tools in wrong order

**AgentGoalAccuracy**: LLM-as-judge given the conversation trajectory and a stated goal; returns binary or 0-1 graded score. The judge prompt must include the goal description and the full trajectory.

**TopicAdherence**: Measures whether the agent stayed on-topic and didn't drift into out-of-scope behavior. Useful for guardrail evaluation in production agents.

### LLM-as-Judge Design

A well-calibrated LLM judge requires:
1. **Rubric specificity**: vague criteria (e.g., "is this a good answer?") produce high variance; structured rubrics with anchor examples (1/3/5 with definitions) reduce variance
2. **Reference-based vs. reference-free**: reference-based judges (comparing to a gold answer) outperform reference-free judges on factuality tasks
3. **Calibration**: sample 200–500 examples, collect human labels, compute Spearman ρ or Cohen's κ between judge scores and human scores; production threshold ρ > 0.7 is a reasonable bar
4. **Anti-bias prompting**: randomize answer ordering in pairwise comparisons; use chain-of-thought before verdict to reduce position bias

---

## D. Common Interview Questions (with Strong Answers)

---

**Q: How would you diagnose a RAG pipeline where user satisfaction is low but your BLEU/ROUGE scores look fine?**

**A:** BLEU and ROUGE are surface-form metrics that measure n-gram overlap with a reference — they're fundamentally inadequate for evaluating free-form RAG outputs because a semantically correct answer using different phrasing scores near zero, while a fluent but hallucinated answer paraphrasing the reference scores high. I'd decompose the evaluation using Ragas: first check faithfulness (are we hallucinating?) and context_recall (are we retrieving the right chunks?). Low user satisfaction with high BLEU typically indicates the retrieval is surfacing plausible but wrong content, and the generation is faithfully reproducing that wrong content — which faithfulness won't catch because it only measures groundedness *to retrieved context*, not correctness of the context itself. The fix there is context_recall against a curated reference corpus. This is the exact same pathology as optimizing a recommender for click-through rate without measuring session satisfaction — the metric you're optimizing is misaligned with the objective.

> **Staff-Level Extension**: How would you handle the evaluation when you don't have ground-truth answers for context_recall? I'd use a judge-based approach: sample 100–200 queries, have domain experts annotate the retrieved chunks as relevant/not-relevant, then compute a human-labeled context recall. That becomes your offline eval benchmark. You can then use a cheaper LLM judge calibrated against those human labels for continuous evaluation at scale.

---

**Q: Walk me through how Ragas faithfulness is computed and where it can fail.**

**A:** Faithfulness decomposes the answer into atomic claims using an LLM, then has the same or different LLM judge each claim against the retrieved context as supported or unsupported — the score is the fraction of supported claims. The failure modes are: (1) **claim decomposition errors** — complex sentences may be split poorly, merging two claims into one so a partially-supported compound claim gets credit; (2) **judge model agreement with generator** — if you use GPT-4 as both generator and faithfulness judge, the judge tends to interpret ambiguous claims charitably, inflating scores; (3) **implicit knowledge leakage** — a powerful judge model may mark claims as "supported" because it knows them to be true from pretraining, not because the context actually supports them. In production I'd use a judge model from a different family than the generator, and run a calibration pass where I intentionally inject hallucinations and verify the judge catches them.

> **Staff-Level Extension**: How would you make faithfulness evaluation scale to millions of evaluations per day? Replace the LLM judge with a fine-tuned NLI model (e.g., DeBERTa fine-tuned on NLI datasets + RAG-specific annotation). You trade some accuracy for 100x cost reduction and sub-10ms latency. Track drift between the NLI model and periodic LLM judge samples to detect distribution shift.

---

**Q: What are the failure modes of LLM-as-judge, and when would you explicitly NOT use it?**

**A:** The canonical failure modes are: (1) **circular evaluation** — using the same model or model family to both generate and judge produces correlation-inflated scores; GPT-4 judging GPT-4 outputs is the AI equivalent of a student grading their own exam; (2) **verbosity bias** — judges systematically prefer longer, more hedged answers even when a shorter answer is more correct; (3) **position bias** — in pairwise comparisons, judges prefer the first option ~60% of the time; (4) **sycophancy** — if the prompt reveals which answer was generated by the "better" system, scores shift. I would NOT use LLM-as-judge in: high-stakes decisions (medical, legal) where judge errors compound into liability; latency-sensitive pipelines where a judge call adds 500ms+; or any evaluation where the ground truth can be deterministically verified (code execution, SQL correctness, factual lookup against a structured KB). In those cases, use execution-based evaluation or structured comparison.

> **Staff-Level Extension**: How do you detect and measure verbosity bias empirically? Run an A/B experiment: for a fixed set of queries, generate a concise correct answer and a verbose correct answer covering the same content. If the judge scores the verbose answer higher at a statistically significant rate, you've confirmed verbosity bias. Mitigate by adding explicit rubric language: "prefer concise answers unless detail is warranted by the question complexity."

---

**Q: How does Cohere Rerank improve retrieval quality, and what are the latency/cost trade-offs in production?**

**A:** Bi-encoder ANN retrieval encodes query and document independently — the model compresses each into a single vector with no token-level interaction, so it can miss exact keyword matches, negation (retrieving a doc that says "X is false" for a query about X), and complex semantic dependencies. Cohere Rerank uses a cross-encoder that concatenates query and document and runs full attention across both, producing a calibrated relevance score that captures these interactions. The trade-off: bi-encoder retrieval is O(log n) via ANN index; cross-encoder reranking is O(K) sequential inference calls where K is your candidate pool size. For K=50 at ~2ms/call that's 100ms added latency. In practice I bound K conservatively, run reranking async where possible, and cache rerank scores for popular queries. The quality gain is typically 5–15% improvement in NDCG@5 vs. bi-encoder alone, which is significant enough to justify the latency cost in most non-streaming pipelines.

> **Staff-Level Extension**: When would you skip the reranker entirely? When the query distribution is narrow and well-defined (e.g., internal tooling where queries are templated), a fine-tuned bi-encoder often matches reranker quality at a fraction of the cost. The reranker earns its latency budget on tail queries and long-tail semantic matching.

---

**Q: How would you design an offline evaluation dataset for an agent that uses tool calls?**

**A:** The key challenge is that agent evaluation requires trajectory supervision, not just answer supervision. I'd construct a reference dataset with: (1) `user_input` — the task description; (2) `reference_tool_calls` — the expected ordered sequence of tool invocations with argument schemas; (3) `reference` — the final answer or goal state. For ToolCallAccuracy, the reference_tool_calls must capture not just tool names but argument structure — e.g., `search(query="Q3 2024 revenue")` not just `search`. I'd collect ground truth by having domain experts manually trace the optimal tool-call path for a diverse sample of tasks, stratified by complexity. For AgentGoalAccuracy, I'd write goal descriptions at the task level (e.g., "Schedule a meeting for next Tuesday between Alice and Bob") and evaluate the conversation end-state. The hardest part is handling valid alternative trajectories — an agent that achieves the goal via a different but correct tool sequence should not be penalized, so the judge must evaluate goal achievement, not just trajectory replication.

> **Staff-Level Extension**: How do you handle evaluation of agents with non-deterministic tool outputs (e.g., live web search)? Snapshot the tool outputs at dataset creation time, inject them as mocked responses during eval, and evaluate the agent's decision-making given those fixed inputs. This decouples agent reasoning quality from tool reliability, which is the signal you actually want.

---

**Q: Answer relevancy uses embedding-based reverse question generation — what are the implicit assumptions, and when does this break?**

**A:** The metric assumes that a relevant answer to question Q should, when used as a prompt to generate questions, produce questions whose embeddings are close to Q in embedding space. This breaks in several ways: (1) **ambiguous questions** — if the original question is ambiguous, the answer may correctly address one interpretation, but the embedding similarity will be low because the reverse-generated question matches the answered interpretation, not the original surface form; (2) **embedding model domain mismatch** — if the embedding model wasn't trained on the domain's vocabulary, the similarity scores are unreliable; (3) **tautological answers** — an answer that essentially repeats the question verbatim will produce high similarity but provides no value; (4) **multi-part questions** — an answer addressing only one sub-question will score high if the addressed sub-question dominates the original embedding. I treat answer_relevancy as a necessary but not sufficient signal — I always pair it with faithfulness and a human evaluation sample to catch these failure modes.

---

## E. Gotchas, Trade-offs & Best Practices

- **Metric orthogonality is not guaranteed in practice.** Faithfulness and answer_relevancy can move in opposite directions under prompt changes — a more constrained generation prompt may improve faithfulness (fewer unsupported claims) while reducing answer_relevancy (the answer becomes too terse and misses sub-questions). Always track the full metric vector, not a single aggregate, and define your optimization objective explicitly before prompt engineering. This is identical to the precision/recall trade-off in search — you need to know which side of the curve you're optimizing for.

- **Context_recall requires ground truth, which is expensive and often unavailable at scale.** In production, you typically have ground-truth answers only for a curated seed set (e.g., your evaluation benchmark). Treat context_recall as a benchmark metric (run on seed set) rather than a continuous online metric. For continuous retrieval monitoring, proxy metrics like retrieved chunk click-through rate or user correction signals are more practical. Similarly, context_precision can be estimated with an LLM judge, but that judge's accuracy on your domain must be validated against human labels first.

- **The Ragas LLM judge is a single point of failure.** Ragas defaults to using the same LLM (configurable) for multiple metrics in a single evaluation run. If the judge model has an outage, rate limit, or produces degenerate outputs (e.g., malformed JSON), the entire evaluation pipeline silently fails or returns NaN. In production, wrap `ragas.evaluate()` in retries, validate output distributions (a sudden spike to 1.0 or 0.0 across all samples is a red flag), and compare current run averages to rolling baselines with anomaly detection.

- **Agent evaluation with ToolCallAccuracy penalizes valid alternative trajectories.** Reference tool-call sequences represent one expert path, but agents may achieve the goal via a different valid sequence. If ToolCallAccuracy is your primary agent metric, you'll inadvertently optimize for trajectory mimicry rather than goal achievement. The right design is a **hierarchical evaluation**: ToolCallAccuracy for common/known task types where the optimal path is well-defined, AgentGoalAccuracy as the primary metric for open-ended tasks, and human evaluation for novel task categories.

- **Ragas evaluation costs scale with dataset size and metric selection.** Each faithfulness evaluation makes 2 LLM calls per sample (claim decomposition + claim verification); context_precision makes K LLM calls per sample (one per retrieved chunk). For a dataset of 1,000 samples with top-5 retrieval, a full 4-metric Ragas run can cost $5–50 depending on model choice. Use GPT-4o-mini or a fine-tuned judge for continuous eval, reserve GPT-4 for calibration runs and red-teaming. Cache evaluation results aggressively — re-evaluating the same (query, context, response) triple is pure waste.

---

## F. Code & Architecture Patterns

### Pattern 1: Ragas Evaluation on a RAG Pipeline

```python
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

# Configure judge LLM — explicitly NOT the same model used in generation
# to avoid circular evaluation
judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
judge_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# EvaluationDataset format:
# user_input: str          — the query
# retrieved_contexts: list[str]  — chunks passed to the generator
# response: str            — generated answer
# reference: str           — ground truth answer (required for context_recall)
samples = [
    {
        "user_input": "What is the return policy for electronics?",
        "retrieved_contexts": [
            "Electronics may be returned within 30 days with original packaging.",
            "Software products are non-refundable once opened."
        ],
        "response": "You can return electronics within 30 days if you have the original packaging.",
        "reference": "Electronics can be returned within 30 days with original packaging. Software is non-refundable once opened."
    },
    # ... more samples
]

eval_dataset = EvaluationDataset.from_list(samples)

# Metrics selection: faithfulness + answer_relevancy don't need reference
# context_recall requires reference — only include if you have ground truth
metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,   # requires reference field
    context_precision,
]

results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=judge_llm,
    embeddings=judge_embeddings,
    raise_on_failure=False,  # don't abort on single sample failures
)

# Convert to pandas for analysis and anomaly detection
df = results.to_pandas()

# Validate distribution — sudden all-1.0 or all-NaN is a judge failure signal
for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
    col = df[metric]
    print(f"{metric}: mean={col.mean():.3f}, std={col.std():.3f}, null_rate={col.isna().mean():.2%}")

# Identify worst-performing samples for error analysis
worst = df.nsmallest(10, "faithfulness")[["user_input", "response", "faithfulness"]]
print(worst)
```

---

### Pattern 2: Agent Evaluation with ToolCallAccuracy and AgentGoalAccuracy

```python
from ragas import evaluate, EvaluationDataset
from ragas.metrics import ToolCallAccuracy, AgentGoalAccuracy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))

# Agent evaluation dataset requires:
# user_input: the task description
# response: the agent's final answer/output
# reference: the expected final answer or goal state description
# reference_tool_calls: list of expected tool invocations with args
agent_samples = [
    {
        "user_input": "What is the current weather in San Francisco and should I bring an umbrella?",
        "response": "Current weather in San Francisco is 58°F with 80% chance of rain. Yes, bring an umbrella.",
        "reference": "It is 58°F in San Francisco with high chance of rain; an umbrella is recommended.",
        "reference_tool_calls": [
            {
                "name": "get_weather",
                "args": {"location": "San Francisco, CA", "units": "fahrenheit"}
            }
        ],
        # The actual tool calls the agent made — captured from agent trajectory
        "tool_calls": [
            {
                "name": "get_weather",
                "args": {"location": "San Francisco", "units": "fahrenheit"}
            }
        ]
    },
    {
        "user_input": "Book a meeting with the engineering team for next Monday at 2pm",
        "response": "I've scheduled a meeting with the engineering team for Monday April 7th at 2:00 PM.",
        "reference": "Meeting successfully scheduled with engineering team for next Monday at 2pm.",
        "reference_tool_calls": [
            {"name": "get_calendar_availability", "args": {"team": "engineering", "date": "next_monday"}},
            {"name": "create_calendar_event", "args": {"title": "Engineering Team Meeting", "time": "2pm", "date": "next_monday"}}
        ],
        "tool_calls": [
            {"name": "get_calendar_availability", "args": {"team": "engineering", "date": "next_monday"}},
            {"name": "create_calendar_event", "args": {"title": "Team Meeting", "time": "14:00", "date": "2026-04-07"}}
        ]
    }
]

eval_dataset = EvaluationDataset.from_list(agent_samples)

agent_metrics = [
    ToolCallAccuracy(),      # exact + semantic match on tool name + args
    AgentGoalAccuracy(),     # LLM-as-judge: did agent achieve the stated goal?
]

agent_results = evaluate(
    dataset=eval_dataset,
    metrics=agent_metrics,
    llm=judge_llm,
    raise_on_failure=False,
)

df_agent = agent_results.to_pandas()
print(df_agent[["user_input", "tool_call_accuracy", "agent_goal_accuracy"]])
```

---

### Pattern 3: Cohere Rerank in a LangChain RAG Pipeline

```python
import cohere
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List

class CohereRerankedRetriever(BaseRetriever):
    """
    Two-stage retriever: bi-encoder ANN (top-K) → Cohere cross-encoder rerank (top-k).
    K >> k: cast a wide net, rerank to a precise short list for generation.
    """
    vectorstore: Chroma
    cohere_client: cohere.Client
    initial_k: int = 50      # bi-encoder candidate pool — wide
    final_k: int = 5         # reranked list passed to LLM — precise
    rerank_model: str = "rerank-english-v3.0"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Stage 1: fast ANN retrieval — embedding similarity, no query-doc interaction
        candidates = self.vectorstore.similarity_search(query, k=self.initial_k)

        # Stage 2: cross-encoder rerank — full query-document attention, calibrated scores
        rerank_response = self.cohere_client.rerank(
            model=self.rerank_model,
            query=query,
            documents=[doc.page_content for doc in candidates],
            top_n=self.final_k,
            return_documents=True,
        )

        # Map reranked results back to Document objects (preserving metadata)
        reranked_docs = []
        for result in rerank_response.results:
            doc = candidates[result.index]
            doc.metadata["rerank_score"] = result.relevance_score
            reranked_docs.append(doc)

        return reranked_docs


# Usage
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
co = cohere.Client(api_key="your-cohere-api-key")

retriever = CohereRerankedRetriever(
    vectorstore=vectorstore,
    cohere_client=co,
    initial_k=50,
    final_k=5,
)

# Integrate with a RAG chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(
        f"[Score: {d.metadata.get('rerank_score', 'N/A'):.3f}] {d.page_content}"
        for d in docs
    )

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the provided context.\n\nContext:\n{context}\n\nQuestion: {question}"
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the return policy for electronics?")
```

---

### Pattern 4: LLM-as-Judge with Calibration Check

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from scipy.stats import spearmanr

FAITHFULNESS_JUDGE_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating whether a generated answer is faithful to the provided context.
A faithful answer contains only claims that are directly supported by the context.

Scoring rubric:
1 - Completely unfaithful: major claims contradict or are absent from context
2 - Mostly unfaithful: some claims supported but significant hallucination present
3 - Partially faithful: roughly half the claims are supported
4 - Mostly faithful: minor unsupported details only
5 - Completely faithful: every claim is directly supported by context

Context:
{context}

Answer:
{answer}

First, list each factual claim in the answer and whether it appears in the context.
Then provide your score (1-5) on the last line as: SCORE: <n>
""")

judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
judge_chain = FAITHFULNESS_JUDGE_PROMPT | judge

def run_judge(context: str, answer: str) -> int:
    response = judge_chain.invoke({"context": context, "answer": answer})
    # Parse score from last line
    for line in reversed(response.content.strip().split("\n")):
        if line.startswith("SCORE:"):
            return int(line.split(":")[1].strip())
    raise ValueError(f"Could not parse score from: {response.content}")

# Calibration: compare judge scores to human labels
# Load your human-annotated calibration set
calibration_df = pd.read_csv("calibration_set.csv")  # columns: context, answer, human_score

calibration_df["judge_score"] = calibration_df.apply(
    lambda row: run_judge(row["context"], row["answer"]), axis=1
)

rho, p_value = spearmanr(calibration_df["human_score"], calibration_df["judge_score"])
print(f"Spearman ρ (judge vs human): {rho:.3f}, p={p_value:.4f}")

# Production threshold: ρ > 0.70 is acceptable for continuous monitoring
# ρ < 0.50 means the judge is not reliable for this domain — recalibrate or replace
if rho < 0.70:
    print("WARNING: Judge calibration below threshold. Do not use for production evaluation.")
```

---

*Study guide generated for Session 10: Evaluating RAG & Agents with Ragas*
*Target audience: Senior/Staff AI Engineer interview preparation*
