# AI Engineering Foundations: Interview Study Guide
### Session: Vibe Check & AI Engineering Foundations

---

## A. Core Concept Summary

AI Engineering is the discipline of building reliable, production-grade systems that leverage pre-trained foundation models — the primary leverage is in how you compose, prompt, retrieve, and orchestrate rather than in how you train. Unlike ML Engineering, which centers on the training loop and model ownership, AI Engineering centers on the **inference-time loop**: how well the system behaves given a fixed (or fine-tuned) model. The dominant mental model is **Build → Deploy → Improve**, where evaluation is continuous and often qualitative before it becomes rigorous. The three canonical application patterns — Prompting, RAG, and Agents — are not a progression from simple to advanced; they are distinct architectural choices with different failure modes, latency profiles, and operational costs. Practitioners who come from search and recommendation backgrounds have a material advantage here: retrieval quality, re-ranking, and offline/online eval divergence are already in their vocabulary.

---

## B. Key Terms & Definitions

- **AI Engineering**: The practice of building production systems on top of foundation models (LLMs, multimodal models) via prompting, retrieval, and orchestration — as opposed to training models from scratch. Ownership extends to the full inference pipeline, eval framework, and deployment infrastructure.

- **Vibe Checking**: A fast, informal evaluation heuristic — manually probing model outputs across a diverse set of representative inputs before investing in automated eval harnesses. It is the "does this smell right?" pass that de-risks early architectural decisions.

- **RAG (Retrieval-Augmented Generation)**: An architecture that decouples the knowledge store from the model weights by injecting retrieved context at inference time. Addresses knowledge cutoffs and hallucination at the cost of retrieval quality becoming a new failure surface.

- **Agentic System**: A system in which an LLM emits structured actions (tool calls, API requests, sub-agent invocations) that are executed in a loop with feedback until a terminal condition is reached. Introduces compounding error rates and non-determinism as key risk vectors.

- **Hallucination**: The tendency of LLMs to generate fluent, confident text that is factually incorrect or unsupported by context. Mechanistically, this arises from next-token prediction optimizing for plausibility over correctness.

- **Context Window**: The finite token budget available to a model for a single inference call (input + output). Determines max document size, conversation history depth, and retrieval chunk count — all of which are architectural constraints, not just tunables.

- **Knowledge Cutoff**: The date beyond which a model's training data does not extend. Means the model has no awareness of recent events, API changes, or evolving domain knowledge unless injected at inference time.

- **Non-determinism**: LLM outputs are stochastic at temperature > 0, meaning the same input can produce different outputs across calls. This breaks naïve unit testing and requires distributional evaluation rather than exact-match assertions.

- **Feedback Loop (Build → Deploy → Improve)**: The AI Engineering analogue to the ML training loop. Instead of retraining, the loop tightens through prompt iteration, retrieval tuning, guardrail adjustment, and selective fine-tuning — all driven by production signal.

- **Prompt Engineering**: The practice of crafting, structuring, and iterating on natural language instructions to steer model behavior. At the staff level, this includes few-shot example selection, chain-of-thought elicitation, output schema enforcement, and systematic ablation.

---

## C. How It Works — Technical Mechanics

### Role Divergence: AI Eng vs. ML Eng vs. Data Science

| Dimension | Data Science | ML Engineering | AI Engineering |
|---|---|---|---|
| **Primary artifact** | Insights / models | Trained model + serving infra | LLM-powered application |
| **Core loop** | EDA → model → report | Feature eng → train → serve | Prompt → retrieve → orchestrate |
| **Failure mode ownership** | Model quality | Training pipeline, skew | Hallucination, latency, cost |
| **Eval cadence** | Offline metrics | CI/offline + shadow | Continuous, often human-in-loop |
| **Model relationship** | Trains it | Trains + owns it | Consumes it via API |

Key divergence: ML Engineering owns the training loop and model weights. AI Engineering treats the model as a **black-box dependency** and owns everything around it. This shifts the engineering challenge from optimization to **composition and reliability**.

---

### The Three Core Patterns — Decision Logic

```
Start: What does the system need to do?
│
├─ Can you solve it with a well-crafted prompt + model knowledge alone?
│   └─ YES → PROMPTING (zero/few-shot)
│       Trade-off: fragile to distribution shift, no fresh knowledge
│
├─ Does the system need facts, documents, or data not in model weights?
│   └─ YES → RAG
│       Trade-off: retrieval quality becomes P0; chunk strategy matters
│       Wrong if: the retrieval space is unbounded or queries are deeply
│                 multi-hop (agents may be required)
│
└─ Does the system need to take actions, call tools, or reason iteratively?
    └─ YES → AGENTS
        Trade-off: compounding error, hard to debug, expensive per run
        Wrong if: the task is well-defined and bounded (RAG/prompting suffices)
```

**When NOT to use agents**: If success rate per step is 90%, a 5-step chain has ~59% end-to-end success. Agents are correct when the marginal value of task completion justifies the operational complexity — not by default.

---

### The AI Engineering Feedback Loop

```
[Business Problem]
       │
       ▼
[Prototype: Prompt / RAG / Agent] ──► [Vibe Check: manual probing, 20-50 examples]
       │                                         │
       │                              Looks wrong? iterate here
       │                              Looks promising? proceed
       ▼
[Deploy: shadow or canary]
       │
       ▼
[Collect production signal: thumbs, task completion, implicit feedback]
       │
       ▼
[Improve: prompt update / retrieval tuning / fine-tune / guardrail]
       │
       └──────────────────────────────────────────────────────► loop
```

Key difference from ML training loop: you do not retrain on every cycle. The loop tightens primarily through **prompt and retrieval changes**, with fine-tuning as a later, more expensive lever.

---

### Vibe Checking — Operational Mechanics

1. **Curate a seed set**: 20–50 inputs covering happy path, edge cases, adversarial inputs, and known failure modes from similar systems.
2. **Run blind**: generate outputs without looking at inputs first to avoid confirmation bias.
3. **Score on vibes**: does the output feel correct, on-topic, appropriately hedged? No rubric yet — you're building intuition.
4. **Cluster failures**: are failures systematic (e.g., always fails on negation) or random? Systematic = prompt issue or retrieval miss. Random = model stochasticity or ambiguous input.
5. **Gate decision**: if >80% of outputs pass the vibe check, proceed to automated eval design. If <50%, don't invest in eval infrastructure yet — fix the architecture first.

---

### LLM Limitations — Production Implications

| Limitation | Production Risk | Mitigation |
|---|---|---|
| Knowledge cutoff | Stale facts, wrong API behavior | RAG, tool use, date injection |
| Hallucination | Silent wrong answers | Grounding, citations, consistency checks |
| Context window | Truncation of long docs/history | Chunking strategy, summarization, compression |
| Non-determinism | Eval flakiness, user trust | Temperature 0 for evals, seed-based testing |
| Latency | User experience, timeout risk | Streaming, caching, model tiering |
| Cost | Unit economics at scale | Prompt compression, caching, smaller models |

---

## D. Common Interview Questions (with Strong Answers)

---

**Q: How do you decide between RAG and fine-tuning for a production use case?**

**A:** I treat RAG and fine-tuning as orthogonal levers, not alternatives. RAG addresses *what the model knows* — it's the right choice when the knowledge domain is large, frequently updated, or needs source attribution. Fine-tuning addresses *how the model behaves* — tone, format, domain-specific reasoning patterns. In practice, I reach for RAG first because it's fast to iterate, doesn't require a training pipeline, and the knowledge store is auditable. I consider fine-tuning when I have consistent format or behavior failures that prompt engineering hasn't resolved after 3–4 iterations, and when I have >1K high-quality labeled examples. The failure mode of over-relying on RAG is that retrieval quality becomes load-bearing — a bad chunk strategy silently degrades the whole system. The failure mode of fine-tuning too early is that you overfit to a training distribution that diverges from production.

**Staff-Level Extension**: *What happens when the knowledge base grows to 10M+ documents and RAG latency becomes a bottleneck?* — Answer should address ANN index scaling, two-stage retrieval (embedding + reranker), query decomposition, and potentially hierarchical retrieval or document summarization layers.

---

**Q: What does "vibe checking" actually tell you, and when does it become insufficient?**

**A:** Vibe checking is a fast signal on whether the architecture is in the right ballpark before you invest in eval infrastructure. It catches egregious failures — wrong format, hallucinated entities, completely off-topic responses — that would waste eval budget if not caught early. It's sufficient when you're in the first 1–2 days of a prototype and need a go/no-go signal. It breaks down when you need to measure marginal improvements — the human eye can't reliably distinguish a 72% vs. 78% correct system. It also breaks down when failures are subtle, domain-specific, or correlated with rare inputs that your seed set doesn't cover. The transition to rigorous eval happens when you need to compare two system variants, when you're approaching production, or when you've had a silent failure in prod that vibe checking missed.

**Staff-Level Extension**: *How do you build an eval set that doesn't overfit to your own vibe?* — Answer should address adversarial example mining, user traffic sampling, failure clustering from prod logs, and the risk of evaluator-generator correlation.

---

**Q: How does the AI Engineering build-deploy-improve loop differ from the ML training loop, and what does that mean for team structure?**

**A:** In the ML training loop, the primary cycle time is dominated by data labeling, feature engineering, and training runs — often days to weeks. The improvement is encoded in model weights, which means deployments are infrequent and high-stakes. In the AI Engineering loop, the primary cycle time is dominated by prompt iteration, retrieval tuning, and eval harness design — often hours to days. Improvements are encoded in prompts, retrieval configs, and guardrails, not weights. This means the loop is much tighter but also more fragile to prompt regressions. For team structure, it means AI Engineering teams need embedded eval infrastructure from day one — you can't treat evaluation as a post-hoc step. It also means the line between "model improvement" and "product iteration" blurs, so ML Engineers and product engineers need to work much closer together than in traditional ML orgs.

**Staff-Level Extension**: *How do you prevent prompt regressions at the velocity of software deployments?* — Answer should address prompt versioning, automated regression suites, canary eval on production traffic, and gating deploys on eval score thresholds.

---

**Q: Walk me through how you'd frame an ambiguous business problem into an AI Engineering architecture.**

**A:** I start by decomposing the business problem into a precise task definition: what is the input, what is the desired output, and what does "good" look like — quantitatively if possible. Then I ask: is this a knowledge retrieval problem, a generation problem, a classification problem, or a multi-step reasoning problem? That determines the base pattern (RAG, prompting, agents). Then I assess the failure cost — is a wrong answer silently harmful (medical, financial) or just annoying? High failure cost means more guardrails, human-in-loop, or conservative output strategies. I then prototype the simplest possible version first — usually a single prompt with no retrieval — and vibe check it. This establishes a baseline and reveals whether the model even has the domain knowledge needed before I build a RAG pipeline. The most common mistake I see is jumping to agents before establishing whether a single well-crafted prompt already solves 80% of the task.

**Staff-Level Extension**: *How do you handle the case where the business metric (e.g., revenue, retention) is hard to connect to a model metric (e.g., BLEU, accuracy)?* — Answer should address proxy metrics, A/B testing strategy, implicit feedback loops, and the risk of optimizing a proxy that diverges from the true objective.

---

**Q: How do you handle LLM non-determinism in a production system that requires consistent, auditable outputs?**

**A:** Non-determinism is a spectrum, not a binary. For auditable outputs, I first ask whether I can set temperature to 0 or near-0 — this dramatically reduces variance while not fully eliminating it due to floating-point non-determinism across hardware. For truly consistent outputs (e.g., structured data extraction, classification), I use output schema enforcement (JSON mode, function calling, constrained decoding) to constrain the output space, then post-validate against a schema. For audit trails, I log the full prompt + output + model version + timestamp — prompt versioning is as important as model versioning for reproducibility. Where non-determinism is unavoidable and consequential (e.g., legal or medical contexts), I run multiple samples and take a majority vote or flag disagreement for human review. The subtler issue is that "deterministic" APIs can still change behavior on model version bumps — I treat model version as a deployment dependency and lock it in production.

**Staff-Level Extension**: *How do you test a system where you can't assert exact output equality?* — Answer should address LLM-as-judge evaluation, semantic similarity thresholds, behavioral test suites that check output properties (not exact strings), and canary scoring against held-out golden sets.

---

**Q: What are the hardest parts of moving an AI application from prototype to production, and what would you prioritize?**

**A:** The gap between prototype and production is almost entirely an evaluation and reliability problem, not a model quality problem. In prototype, you've implicitly hand-picked the inputs that work. In production, the input distribution is adversarial, long-tailed, and continuously shifting. I'd prioritize three things: first, an eval harness with a representative golden set before the first production deploy — without this, you're flying blind on regressions. Second, observability: full prompt/response logging, latency percentiles, cost per request, and failure rate by input category. Third, graceful degradation — the system should have a well-defined fallback (a simpler prompt, a retrieval-only answer, or a "I don't know") rather than failing silently with a hallucinated response. The most common mistake is treating the LLM as a reliable function and not building the defensive infrastructure around it until after the first production incident.

**Staff-Level Extension**: *How do you manage prompt versioning across a team of 5+ engineers all iterating simultaneously?* — Answer should address prompt-as-code in version control, experiment tracking for prompt variants, ownership conventions, and the risk of implicit coupling between prompt version and retrieval/tool behavior.

---

## E. Gotchas, Trade-offs & Best Practices

- **Retrieval is the new feature engineering.** In RAG systems, the quality of the retrieved context dominates output quality far more than prompt wording. Engineers from search backgrounds know this instinctively — but engineers new to AI systems underinvest in chunk strategy, embedding model selection, and reranker calibration. A common failure: chunking documents at fixed token sizes without respecting semantic boundaries, then wondering why the model gives incomplete answers.

- **Agents amplify both capability and failure.** Every additional tool or step in an agent loop multiplies failure probability. A 5-step agent with 90% per-step reliability has ~59% end-to-end reliability — unacceptable for most production use cases. Before building an agent, ask whether a well-structured single prompt with tool use in one shot can solve the problem. Reserve true multi-step agents for tasks where the state space is genuinely too large to enumerate upfront.

- **Vibe checking is not a substitute for eval — it's a gate.** The most expensive mistake is building a rigorous eval harness around a fundamentally broken architecture. Vibe checking prevents this by failing fast. But the inverse mistake — shipping on vibes alone — is equally dangerous. The transition criterion should be explicit: a documented seed set, a pass threshold, and a plan for the next eval tier.

- **Model version bumps are silent breaking changes.** LLM providers update models (including "stable" versions) with behavior changes that don't show up as API errors. Without a locked model version and a regression suite, a provider-side change can degrade your production system overnight. Treat model version as a first-class dependency with the same change management as a library upgrade.

- **Cost and latency are architectural constraints, not afterthoughts.** A system that works at 100 QPS with GPT-4o may be economically unviable at 10,000 QPS. Prompt compression, semantic caching (e.g., caching embeddings of common queries and their responses), model tiering (routing simple queries to smaller/cheaper models), and batching are production-critical design decisions that should be in the architecture from the start, not bolted on after cost alarms fire.

---

## F. Code & Architecture Patterns

### Pattern 1: Vibe Check Harness — Fast Prototype Evaluation

```python
from openai import OpenAI
from dataclasses import dataclass
from typing import Callable
import json

client = OpenAI()

@dataclass
class VibeCheckResult:
    input: str
    output: str
    passed: bool
    notes: str

def vibe_check(
    seed_inputs: list[str],
    system_prompt: str,
    judge_fn: Callable[[str, str], tuple[bool, str]],
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> list[VibeCheckResult]:
    """
    Run a fast vibe check over a seed set.
    judge_fn: (input, output) -> (passed: bool, notes: str)
    At prototype stage, judge_fn is often a human or a lightweight LLM-as-judge.
    """
    results = []
    for inp in seed_inputs:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": inp},
            ],
        )
        output = response.choices[0].message.content
        passed, notes = judge_fn(inp, output)
        results.append(VibeCheckResult(inp, output, passed, notes))

    pass_rate = sum(r.passed for r in results) / len(results)
    failures = [r for r in results if not r.passed]

    print(f"Pass rate: {pass_rate:.0%} ({len(results) - len(failures)}/{len(results)})")
    print(f"\nFailure clusters:")
    for f in failures:
        print(f"  INPUT: {f.input[:80]}")
        print(f"  NOTES: {f.notes}\n")

    return results


# --- LLM-as-judge for automated vibe checking ---
def llm_judge(criteria: str) -> Callable[[str, str], tuple[bool, str]]:
    """
    Returns a judge function that uses an LLM to evaluate outputs.
    criteria: natural language description of what 'good' looks like.
    """
    def judge(inp: str, output: str) -> tuple[bool, str]:
        prompt = f"""You are evaluating an AI system output. 
Criteria for a good output: {criteria}

Input: {inp}
Output: {output}

Respond with JSON: {{"passed": true/false, "notes": "brief reason"}}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # cheaper model for judging
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(resp.choices[0].message.content)
        return result["passed"], result["notes"]

    return judge


# --- Usage ---
seed_inputs = [
    "Summarize the Q3 earnings report for a non-finance audience.",
    "What was our revenue growth last quarter?",  # out-of-context — should trigger graceful fallback
    "List the top 3 risks mentioned.",
    "",  # edge case: empty input
]

system_prompt = """You are a financial document assistant. 
Answer only based on the provided document context. 
If context is insufficient, say so explicitly — do not speculate."""

judge = llm_judge("Output is grounded in context, not speculative, and appropriately hedged when uncertain.")

results = vibe_check(seed_inputs, system_prompt, judge)
```

---

### Pattern 2: RAG Pipeline with Annotated Failure Surfaces

```python
from typing import Optional
import numpy as np

class RAGPipeline:
    """
    Failure surface 1: Embedding model quality — domain mismatch degrades recall
    Failure surface 2: Chunk strategy — too small loses context, too large dilutes signal
    Failure surface 3: Reranker — cross-encoder reranking is expensive but critical for P@1
    Failure surface 4: Prompt grounding — model may still hallucinate beyond retrieved context
    """

    def __init__(self, retriever, reranker, generator, max_context_tokens: int = 3000):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.max_context_tokens = max_context_tokens

    def run(self, query: str, top_k: int = 20, rerank_top_n: int = 5) -> dict:
        # Stage 1: Approximate retrieval (fast, high recall, lower precision)
        candidates = self.retriever.retrieve(query, top_k=top_k)

        if not candidates:
            return {"answer": "I don't have relevant information to answer this.",
                    "sources": [], "retrieval_score": 0.0}

        # Stage 2: Reranking (slow, higher precision) — skip if latency-constrained
        reranked = self.reranker.rerank(query, candidates, top_n=rerank_top_n)

        # Stage 3: Context assembly with token budget enforcement
        context = self._assemble_context(reranked, self.max_context_tokens)

        # Stage 4: Grounded generation with explicit citation instruction
        answer = self.generator.generate(
            query=query,
            context=context,
            instruction="Answer based only on the provided context. "
                       "If the context doesn't support a complete answer, say so. "
                       "Cite source IDs where relevant.",
        )

        return {
            "answer": answer,
            "sources": [c["id"] for c in reranked],
            "retrieval_score": np.mean([c["score"] for c in reranked]),
        }

    def _assemble_context(self, chunks: list[dict], max_tokens: int) -> str:
        """
        Token-budget-aware context assembly.
        Ordering matters: most relevant chunks first (lost-in-the-middle effect).
        """
        context_parts = []
        token_count = 0
        for chunk in chunks:
            chunk_tokens = len(chunk["text"].split()) * 1.3  # rough token estimate
            if token_count + chunk_tokens > max_tokens:
                break
            context_parts.append(f"[{chunk['id']}] {chunk['text']}")
            token_count += chunk_tokens
        return "\n\n".join(context_parts)
```

---

### Pattern 3: Business Problem → Architecture Decision Tree

```python
def select_ai_pattern(requirements: dict) -> str:
    """
    requirements keys:
      - needs_external_knowledge: bool  (facts not in model weights)
      - knowledge_changes_frequently: bool
      - needs_multi_step_action: bool
      - latency_budget_ms: int
      - failure_cost: str  # 'low' | 'medium' | 'high'
      - labeled_examples_available: int
    """
    r = requirements

    # Agents: only if genuinely multi-step with dynamic state
    if r["needs_multi_step_action"] and r["latency_budget_ms"] > 5000:
        if r["failure_cost"] == "high":
            return "AGENT with human-in-loop confirmation gates"
        return "AGENT (monitor per-step success rate, set max_iterations)"

    # RAG: knowledge retrieval with freshness requirements
    if r["needs_external_knowledge"] or r["knowledge_changes_frequently"]:
        if r["latency_budget_ms"] < 500:
            return "RAG with pre-computed embeddings + semantic cache"
        return "RAG with two-stage retrieval (ANN + reranker)"

    # Fine-tuning: behavioral adaptation with sufficient data
    if r["labeled_examples_available"] > 1000 and not r["needs_external_knowledge"]:
        return "FINE-TUNE on behavior pattern, then prompt for task"

    # Default: prompting is always the baseline
    return "PROMPTING (zero/few-shot) — establish baseline before adding complexity"
```

---

> **Key mental model to carry into any interview**: AI Engineering is fundamentally about managing the gap between what a model *can* do in a notebook and what a system *reliably does* in production. Every architectural decision — RAG vs. fine-tune, prompting vs. agents, vibe check vs. rigorous eval — is a bet on where that gap lives and how to close it with the least operational complexity. Senior practitioners earn their keep by choosing the simplest architecture that meets the reliability bar, not the most sophisticated one.
