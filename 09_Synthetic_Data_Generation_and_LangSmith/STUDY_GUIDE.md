# Session 9: Synthetic Data Generation & LangSmith Evaluation
## Interview-Ready Study Guide — Senior/Staff AI Engineer

---

## A. Core Concept Summary

Building a RAG system is easy; **evaluating one honestly is hard**. The cold start problem is the same trap that plagued early recommender system development: you cannot measure retrieval quality or generation faithfulness without labeled query–context–answer triples, yet you have none when you first ship. Synthetic data generation — specifically the RAGAS `TestsetGenerator` pipeline — breaks this dependency by bootstrapping ground-truth from your own corpus, giving you an offline evaluation harness before any human labels exist.

LangSmith closes the loop by providing the infrastructure to **version datasets, run reproducible experiments, and compare pipeline variants** the same way an offline A/B testing framework compares model variants in a recommender stack. The key mental model: treat every RAG experiment as a row in a metrics table where rows are pipeline configurations (chunk size, embedding model, top-k) and columns are evaluator scores (faithfulness, answer relevance, context recall). LangSmith makes that table queryable and persistent. This pairs naturally with the intuition from recommendation systems: offline NDCG/precision doesn't guarantee online CTR lift, and similarly, high RAGAS scores don't guarantee user satisfaction — but they narrow the search space before expensive human eval or A/B testing.

---

## B. Key Terms & Definitions

- **Cold Start Problem (RAG)**: The absence of labeled evaluation data at system inception, making it impossible to measure retrieval or generation quality without first generating or collecting ground-truth QA pairs. Directly analogous to the new-item/new-user cold start in collaborative filtering.

- **RAGAS TestsetGenerator**: A pipeline that ingests source documents and produces `(question, context, reference_answer)` triples by combining LLM-driven question generation, evolution, and answer synthesis. The resulting testset is the offline evaluation corpus.

- **Evol Instruct / Question Evolution**: A technique borrowed from WizardLM where a simple factoid question is iteratively transformed — into multi-hop, reasoning-heavy, conditional, or abstractive variants — to stress-test different retrieval and reasoning capabilities of the pipeline.

- **LangSmith Dataset**: An immutable, versioned collection of `(input, reference_output)` examples against which experiments are run. Acts as the "held-out test set" in the RAG evaluation lifecycle; mutating it requires creating a new version.

- **LangSmith Experiment**: A single execution of `langsmith.evaluate()` against a specific dataset version, producing a row of aggregated evaluator scores and individual run traces. Comparable to a single offline experiment run in an ML training loop.

- **Run Tree**: LangSmith's parent-child trace hierarchy that captures the full execution graph of a chain call — retriever span, LLM call span, output parsing span — enabling per-component latency and token attribution.

- **Annotation Queue**: A LangSmith primitive for routing specific run traces to human reviewers. Used when automated evaluators are insufficient (e.g., subjective tone, domain-specific accuracy) — the equivalent of a labeling pipeline feeding back into your ground-truth corpus.

- **LCEL (LangChain Expression Language)**: A composable, lazy-evaluation DSL for building chains as `runnable | runnable` pipelines. Each `|` operator produces a new `Runnable` that LangSmith can automatically trace at the component level.

- **Faithfulness (RAGAS metric)**: Measures whether every claim in the generated answer is grounded in the retrieved context, scored by an LLM judge. Analogous to precision in retrieval — high faithfulness means no hallucination relative to retrieved content.

- **Context Recall**: Measures whether the retrieved chunks contain all information needed to answer the reference question, scored against the ground-truth answer. Analogous to recall@k in document retrieval.

---

## C. How It Works — Technical Mechanics

### 1. Synthetic Data Generation Pipeline (RAGAS TestsetGenerator)

1. **Document Ingestion & Chunking**: Source documents are split into chunks using configurable `chunk_size` and `chunk_overlap`. These chunks form the retrieval corpus and the candidate context pool. The chunking strategy here is not just an ingestion decision — it defines the granularity of contexts that will appear in your testset, so it must mirror your production chunker.

2. **Knowledge Graph / Relationship Extraction** *(RAGAS v0.2+)*: RAGAS optionally builds a lightweight knowledge graph over chunks, identifying entity co-occurrences across documents. This enables multi-document, multi-hop question generation — questions that require synthesizing facts from two or more chunks.

3. **Scenario Sampling**: A `Scenario` (or `TestsetGenerationConfig`) defines the distribution of question types: simple factoid, multi-hop reasoning, conditional, abstractive summary, etc. Each scenario maps to a `QuerySynthesizer` that has an assigned sampling weight. This is equivalent to stratified sampling over label categories in dataset construction for recommenders.

4. **Question Generation**: For each sampled scenario, an LLM generates a raw question conditioned on one or more retrieved chunks. The prompt instructs the LLM to generate a question answerable *only* from the provided context, reducing distributional leakage.

5. **Evol Instruct Question Evolution**: Simple questions are passed through an evolution prompt chain that applies one of several transformations:
   - **Reasoning evolution**: Adds an inferential step ("Why does X imply Y?")
   - **Multi-hop evolution**: Requires chaining two facts ("Given that A is true, what is the consequence for B?")
   - **Conditional evolution**: Introduces hypotheticals ("If X were false, how would Y change?")
   - **Compression/abstractive evolution**: Requires paraphrasing or summarizing rather than extracting verbatim
   Each evolution type stresses a different failure mode: faithfulness, multi-document retrieval, and reasoning robustness respectively.

6. **Reference Answer Synthesis**: A separate LLM call generates the `reference_answer` conditioned on the question *and* the ground-truth context chunks — not the retriever output. This is a critical distinction: reference answers reflect what *can* be answered from the corpus, not what your RAG pipeline retrieves. Contaminating this step with retriever output introduces label bias identical to training on biased implicit feedback in recommendation.

7. **Filtering**: Low-quality QA pairs are filtered by an LLM critique — checking if the question is answerable, unambiguous, and non-trivial. A final testset of `N` examples is exported as a `pandas.DataFrame` or LangSmith dataset.

---

### 2. LangSmith Experiment Loop

1. **Dataset Registration**: Upload the RAGAS-generated testset to LangSmith as a named dataset. Each row maps `{"question": "..."}` as input and `{"reference_answer": "..."}` as expected output. Pin the dataset version — never mutate the eval set mid-comparison.

2. **Target Function Definition**: Wrap your RAG pipeline in a callable that accepts `{"question": str}` and returns `{"answer": str, "contexts": list[str]}`. This is the function under test.

3. **Evaluator Definition**: Define one or more evaluator functions with signature `(run: Run, example: Example) -> EvaluationResult`. Each evaluator receives the run output and the ground-truth example and returns a named score. Built-in evaluators (e.g., `LangChainStringEvaluator("qa")`) use an LLM judge; custom evaluators can call RAGAS metrics, exact-match scorers, or semantic similarity.

4. **`evaluate()` Invocation**: Call `langsmith.evaluate(target_fn, data=dataset_name, evaluators=[...], experiment_prefix="exp_v1")`. LangSmith fans out each dataset example to `target_fn` (optionally with concurrency), collects outputs, runs evaluators, and stores results under a named experiment.

5. **Run Tree Capture**: Every invocation of the LCEL chain within `target_fn` is automatically traced as a run tree — retriever span (documents fetched, latency), LLM span (prompt tokens, completion tokens, latency), and output parser span. These are queryable in LangSmith's UI and via the SDK.

6. **Experiment Comparison**: Two experiments (e.g., `chunk_256` vs. `chunk_512`) are compared in the LangSmith UI side-by-side. Each row in the results table shows per-example score deltas, and aggregate metrics (mean, std) are shown in column headers. Statistical significance must be assessed separately — LangSmith does not compute p-values; you pull results via `client.get_test_results()` and run your own Wilcoxon or bootstrap CI.

7. **Annotation Queue Routing**: Flag low-scoring examples (faithfulness < 0.5) for human review via annotation queues. Human corrections can be promoted back into the dataset as additional ground-truth labels, tightening the eval loop.

---

## D. Common Interview Questions (with Strong Answers)

---

**Q1: Why can't you just use the questions your users are already asking as your evaluation set?**

**A:** User query logs are tempting — they're real distribution — but they conflate evaluation with production feedback in two dangerous ways. First, they're unlabeled: you have queries but no reference answers, so you can't compute faithfulness or answer correctness without an expensive labeling pass. Second, they're biased by your current system's behavior: if your RAG pipeline systematically fails on multi-hop questions, those queries will be underrepresented because users learn to avoid asking them — the same survivorship bias we see in implicit feedback datasets in recommendation systems (clicks don't tell you about items users never saw). Synthetic testsets intentionally oversample hard cases — multi-hop, conditional, abstractive — to stress-test failure modes your production logs will underrepresent. The right architecture is: synthetic data for initial eval harness → user query logs to calibrate distributional coverage → human annotation to validate edge cases.

> **Staff-Level Extension**: A principal interviewer will ask how you prevent your synthetic testset from becoming stale as your corpus evolves. The answer is treating testset generation as a CI pipeline step: when corpus documents are updated beyond a diff threshold, regenerate or augment the testset, version it, and run regression experiments automatically. This is analogous to retraining evaluation datasets when catalog distribution shifts in a recommender.

---

**Q2: Walk me through how you'd decide between chunk sizes of 256, 512, and 1024 tokens using LangSmith.**

**A:** This is a controlled ablation, and the design matters more than the execution. First, I'd isolate chunk size as the single independent variable — same embedding model, same top-k, same LLM, same prompt. I'd run three experiments against the same pinned dataset version. The primary metric I care about is context recall (does the retrieved set contain the answer?) — chunk size most directly affects this. Smaller chunks increase recall precision but may split a reasoning chain across chunks; larger chunks surface more context per slot but may dilute signal for the embedding model. I'd look at the recall–faithfulness trade-off together: a larger chunk that improves recall but introduces irrelevant content will hurt faithfulness. I'd also look at tail performance (p10, p25 of per-example scores), not just mean — the mean can be flat while the tail gets worse, which matters for production SLAs. Finally, I'd compute a bootstrap 95% CI on the mean score difference across experiments; if the CI includes zero, I default to the operationally simpler configuration (usually smaller chunks, simpler index).

> **Staff-Level Extension**: How do you handle the fact that RAGAS testsets were generated with a specific chunk size, potentially creating alignment bias where the "right" chunk size is whichever one was used during testset generation? Answer: generate the testset at a chunk-agnostic level (document-level contexts for reference answer generation), then vary chunking only in the retrieval index. This decouples testset construction from the hyperparameter under test.

---

**Q3: Your faithfulness scores look good offline, but users are still complaining about hallucinations. What's your diagnosis?**

**A:** This is the offline-online gap, and it mirrors the precision-offline / CTR-online mismatch in recommendation. Three likely culprits: (1) **Evaluator model misalignment** — if your LLM judge for faithfulness is GPT-4 but your generation model is a smaller fine-tuned model, the judge may be too lenient or score on different axes. (2) **Testset distributional gap** — the synthetic testset may oversample factoid questions where hallucination is easy to detect, while user queries skew toward complex, open-ended questions where hallucination is subtle. Check the question evolution distribution: if you have 80% simple factoid and 20% reasoning, your testset doesn't stress the cases where hallucination emerges. (3) **Context boundary hallucination** — the model may be faithful to retrieved contexts but the retrieved contexts themselves are wrong (wrong chunk retrieved). Faithfulness measures grounding in retrieved content, not correctness of retrieved content; you need context precision and context recall to catch this. The fix is a richer evaluation suite: faithfulness, context precision, context recall, and a human annotation pass on a stratified sample of production queries.

> **Staff-Level Extension**: How would you build an online hallucination detection signal without human labels? One approach: use the faithfulness evaluator as an online guardrail — run it asynchronously on production responses and route low-scoring responses to annotation queues. This creates a feedback loop between online detection and offline dataset augmentation.

---

**Q4: How does Evol Instruct question evolution improve the usefulness of a synthetic testset?**

**A:** A testset of only simple factoid questions has the same pathology as a recommendation evaluation dataset built only from high-confidence positive samples — it looks good because the hardest cases are absent. Evolution forces coverage across the difficulty spectrum. Reasoning evolution catches pipelines that retrieve the right chunk but fail to synthesize an inferential answer — this is a generation failure, not a retrieval failure. Multi-hop evolution specifically stresses whether your chunking strategy and top-k retrieval can surface multiple evidence chunks simultaneously; a top-k of 3 may work for factoids but fail for multi-hop. Conditional evolution catches prompt sensitivity and instruction-following brittleness. The practical implication: weight your evolution distribution toward the failure modes you care most about. For a knowledge-intensive enterprise RAG system, I'd oversample multi-hop and conditional; for a simple FAQ bot, factoid is fine. This is identical to the decision of how to weight hard negatives in contrastive learning for retrieval models.

> **Staff-Level Extension**: Evolution adds noise — some evolved questions become unanswerable, ambiguous, or malformed. How do you quality-control evolved questions at scale? Answer: run a separate LLM critique pass (answerability scoring), add human spot-checks on 5-10% of the evolved set, and track `evolution_filter_rate` as a dataset health metric across corpus updates.

---

**Q5: How would you design a LangSmith-based evaluation framework that scales to 10,000 test cases and supports nightly regression runs?**

**A:** At 10K examples, the bottleneck shifts from LLM generation latency to cost and concurrency management. Key design decisions: (1) **Concurrency and rate limiting** — `langsmith.evaluate()` supports `max_concurrency`; set it to match your LLM provider's TPM limits and monitor for 429s. For nightly runs, batch during off-peak hours and use a cheaper evaluator model (GPT-4o-mini for LLM judges rather than GPT-4). (2) **Stratified sampling for fast iteration** — maintain a "fast eval" subset of ~500 stratified examples for PR-level gating (latency < 5 min) and the full 10K for nightly regression. This mirrors the train/val/test split discipline from ML. (3) **Dataset versioning discipline** — pin the dataset version in your CI config; never let nightly runs consume a mutated dataset. Track dataset version as a column in your experiment metadata. (4) **Alerting on metric regression** — define SLA thresholds (e.g., faithfulness must not drop > 2% from baseline) and fail CI automatically. Emit metrics to your observability stack (Datadog, Grafana) alongside LangSmith's internal dashboard for cross-team visibility.

> **Staff-Level Extension**: LLM-as-judge evaluators introduce non-determinism — the same run can score differently across evaluations. How do you handle evaluator variance? Answer: run each evaluator 3x and take the median, track evaluator variance as a metric, and periodically calibrate your evaluator against human labels to detect judge drift.

---

**Q6: How would you adapt a RAGAS-based RAG evaluation pipeline if your corpus is multimodal (text + tables + images)?**

**A:** RAGAS in its standard form is text-only — both the testset generator and evaluators assume text contexts. For multimodal corpora, the evaluation architecture needs extension at three layers. First, **context representation**: tables and images need to be converted to a text-serializable form before entering the RAGAS pipeline — table-to-markdown for structured data, vision model captions or OCR for images. The fidelity of this serialization becomes a first-class quality concern. Second, **question generation**: multimodal content requires multimodal question types — questions that require reading a table column, interpreting a chart, or cross-referencing an image with a caption. Standard RAGAS evolution templates won't generate these; you need custom `QuerySynthesizer` implementations. Third, **faithfulness evaluation**: an LLM judge evaluating faithfulness against a text-serialized table may miss numerical precision errors; you'd want a specialized numerical faithfulness evaluator. The broader lesson is that RAGAS is a framework, not a fixed pipeline — at staff level, you're expected to extend and customize it, not just run the defaults.

---

## E. Gotchas, Trade-offs & Best Practices

- **Testset–Corpus Alignment Drift**: Synthetic testsets go stale as your knowledge base is updated. A testset generated from corpus v1 may have reference answers that are now incorrect after corpus v2 is ingested. Treat testset regeneration as a first-class data engineering task, tied to corpus version control. This is the same staleness problem as maintaining offline evaluation sets in a product catalog that has rapid item churn.

- **LLM Judge Bias and Grade Inflation**: LLM-as-judge evaluators (including RAGAS's built-in metrics) are biased toward verbose, confident-sounding answers regardless of factual accuracy. A model that generates long, fluent non-answers can score high on faithfulness if the judge conflates fluency with grounding. Mitigation: calibrate your judge against human labels on a 100-sample audit set; if judge accuracy < 80% against human, your automated metrics are not trustworthy. Consider using a different (possibly smaller, fine-tuned) judge model.

- **The Chunk-Size–Testset Coupling Trap**: If you generate your testset with one chunking strategy and evaluate retrievers with another, your context recall scores will be artificially low for mismatched chunk sizes because the reference contexts won't align with retrieved chunks. Always generate reference answers at the document level (or with overlapping contexts) and evaluate retrieval independently from generation.

- **`evaluate()` Concurrency and Cost Explosion**: With large testsets and LLM-based evaluators, costs scale as `N_examples × N_evaluators × LLM_cost`. At 10K examples with 4 evaluators using GPT-4, this can exceed $500/run. Establish a tiered evaluation budget: fast/cheap evaluators (embedding similarity, exact match) run on every commit; expensive LLM judges run nightly or on release candidates only.

- **Statistical Significance is Your Responsibility**: LangSmith shows mean metric differences between experiments but does not compute confidence intervals or p-values. A 0.5% mean faithfulness improvement on a 100-sample testset is almost certainly noise. Always pull results via the SDK, compute bootstrap confidence intervals, and treat improvements below the CI threshold as operationally equivalent. This is non-negotiable discipline for anyone with an A/B testing background — apply the same rigor you would to a click-through rate experiment.

---

## F. Code & Architecture Patterns

### Pattern 1: RAGAS Testset Generation from Documents

```python
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

# Load and chunk documents — mirroring production chunking strategy
loader = DirectoryLoader("./data", glob="**/*.md")
docs = loader.load()

# Wrap LLMs — generator LLM for question/answer synthesis,
# critic LLM for filtering (can be same or cheaper model)
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

# Configure synthesizers with sampling weights
# Oversample multi-hop to stress retrieval depth
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=embeddings,
    # Query type distribution mirrors your expected production query mix
)

# Generate testset — reference answers are grounded in source docs,
# NOT in your RAG pipeline's retriever output (critical distinction)
testset = generator.generate_with_langchain_docs(
    docs,
    testset_size=200,
    # Adjust distribution: more multi-hop = harder retrieval stress test
    query_distribution={
        SingleHopSpecificQuerySynthesizer: 0.4,
        MultiHopSpecificQuerySynthesizer: 0.4,
        MultiHopAbstractQuerySynthesizer: 0.2,
    },
)

df = testset.to_pandas()
# Inspect: question, contexts (ground truth), reference_answer, evolution_type
print(df[["question", "reference_answer", "synthesizer_name"]].head())
```

---

### Pattern 2: LangSmith `evaluate()` with Custom Evaluator

```python
import langsmith
from langsmith.evaluation import evaluate, EvaluationResult
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate as ragas_evaluate
from ragas.integrations.langchain import EvaluatorChain
import pandas as pd

client = langsmith.Client()

# --- Upload testset to LangSmith (one-time, then pin the version) ---
dataset_name = "rag-eval-v1"
dataset = client.create_dataset(dataset_name)
for _, row in df.iterrows():
    client.create_example(
        inputs={"question": row["question"]},
        outputs={"reference_answer": row["reference_answer"], "contexts": row["contexts"]},
        dataset_id=dataset.id,
    )

# --- Define RAG pipeline under test (LCEL anatomy) ---
# retriever | prompt | llm | StrOutputParser
# Swapping any component = new experiment

def build_rag_chain(vectorstore, chunk_size_label: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_template(
        "Answer based only on the context below.\nContext: {context}\nQuestion: {question}"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- Target function: must accept dataset input dict, return output dict ---
def rag_target(inputs: dict) -> dict:
    question = inputs["question"]
    answer = rag_chain.invoke(question)
    # Also capture contexts for faithfulness evaluation
    docs = rag_chain.steps[0]["context"].invoke(question)  # retriever step
    return {
        "answer": answer,
        "contexts": [d.page_content for d in docs],
    }

# --- Custom evaluator: RAGAS faithfulness via LLM judge ---
def faithfulness_evaluator(run, example) -> EvaluationResult:
    """
    Evaluator function signature: (Run, Example) -> EvaluationResult
    run.outputs: what the target function returned
    example.outputs: ground truth from dataset
    """
    from datasets import Dataset as HFDataset

    data = {
        "question": [example.inputs["question"]],
        "answer": [run.outputs["answer"]],
        "contexts": [run.outputs["contexts"]],
        "ground_truth": [example.outputs["reference_answer"]],
    }
    hf_dataset = HFDataset.from_dict(data)

    result = ragas_evaluate(
        hf_dataset,
        metrics=[faithfulness],
    )
    score = result["faithfulness"]
    return EvaluationResult(key="faithfulness", score=score)

# --- Run experiment — one call per pipeline variant being compared ---
results = evaluate(
    rag_target,
    data=dataset_name,
    evaluators=[faithfulness_evaluator],
    experiment_prefix="chunk_512_k5",   # change per ablation variant
    max_concurrency=4,
    metadata={"chunk_size": 512, "top_k": 5, "model": "gpt-4o-mini"},
)

# --- Pull results for statistical significance testing ---
# LangSmith does NOT compute CIs — you must do this yourself
results_df = results.to_pandas()
scores = results_df["feedback.faithfulness"].dropna()

import numpy as np
bootstrap_means = [
    np.mean(np.random.choice(scores, size=len(scores), replace=True))
    for _ in range(2000)
]
ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
print(f"Faithfulness: {scores.mean():.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")
```

---

### LCEL Chain Anatomy Reference

```
Input (question)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  retriever          →  List[Document]                   │
│  (swap: k, index,      (affects context recall)         │
│   embedding model)                                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  prompt             →  ChatPromptValue                  │
│  (swap: system         (affects instruction following,  │
│   instructions,         faithfulness)                   │
│   few-shot examples)                                    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  llm                →  AIMessage                        │
│  (swap: model,         (affects reasoning quality,      │
│   temperature,          hallucination rate)             │
│   max_tokens)                                           │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  StrOutputParser    →  str                              │
│  (or custom parser     (affects output structure,       │
│   for structured out)   downstream parseability)        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Output (answer string)
```

**Ablation strategy**: Swap one component per experiment, hold all others constant, and compare on the same pinned dataset version. This is identical to the single-factor ablation discipline in recommender system feature experiments.

---

*Generated for AIE9 Session 9 — Senior/Staff AI Engineering Interview Preparation*
