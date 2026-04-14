# Session 7: Deep Agents — Planning, Subagents, Skills
### Interview-Ready Study Guide | Senior / Staff AI Engineer Level

---

## A. Core Concept Summary

A **deep agent** extends the basic ReAct (Reason + Act) loop by adding explicit long-horizon planning, persistent workspace state, and a hierarchical execution model — separating orchestration concerns from tool execution concerns. Where a standard ReAct agent reacts step-by-step to immediate observations, a deep agent maintains a structured plan that persists across steps, survives partial failures, and can delegate subtasks to specialized child agents (subagents). This decomposition is critical because real-world tasks (code generation pipelines, research workflows, multi-file refactors) exceed what fits in a single context window with an acceptable latency budget.

The key mental model: treat the agent's **filesystem workspace** as its working memory and **the plan (TODO list) as its control flow**. The agent reads its own plan, executes against it, writes results back to disk, marks tasks complete, and replans on failure — all as a recoverable, inspectable, resumable process. This is fundamentally different from prompt chaining; the intermediate state is durable and human-auditable.

---

## B. Key Terms & Definitions

- **Deep Agent**: An LLM-orchestrated agent designed for long-horizon, multi-step tasks with explicit planning, persistent state, and tolerance for high latency. Distinguished from ReAct agents by structured task decomposition and durable context management.

- **Orchestrator/Worker Pattern**: An architectural split where a parent (orchestrator) agent maintains the plan and delegates subtasks to child (worker) agents that operate with narrower, specialized context. The orchestrator aggregates results and manages overall task state.

- **Subagent**: A child agent spawned by an orchestrator to execute a discrete, bounded subtask. Receives a focused context slice from the parent; results are returned to the orchestrator for integration into the plan.

- **SKILL.md**: A markdown file encoding a reusable capability definition — what the skill does, when to invoke it, required inputs/outputs, and step-by-step instructions. The agent reads this at runtime and follows it as structured guidance, enabling capability injection without model fine-tuning.

- **Filesystem-as-Context**: A pattern where agents read and write markdown/JSON files to disk as durable working memory. Enables state to persist across agent restarts, survive context window limits, and be audited or corrected by humans mid-run.

- **Checkpointing**: The practice of persisting intermediate agent state (completed tasks, partial results, tool outputs) so a long-running agent can resume from a known-good point after failure rather than restarting from scratch.

- **Replanning**: The agent's ability to revise its TODO list in response to new information, unexpected tool failures, or changed task scope. Distinguished from simply retrying — replanning changes the *structure* of work remaining.

- **Task Decomposition**: Breaking a high-level goal into a sequence of discrete, individually executable subtasks with clear inputs, outputs, and completion criteria. Quality of decomposition is the primary determinant of deep agent reliability.

- **Tool Design for Deep Agents**: The set of file I/O primitives (`read_file`, `write_file`, `list_directory`), web search (e.g., Tavily), and code execution tools that form the agent's actuation surface. Design decisions around atomicity, idempotency, and error messaging directly affect agent reliability.

- **deepagents-cli**: A command-line harness for running deep agents that handles environment setup, workspace initialization, streaming output, and process lifecycle — abstracting the concerns of running a LangGraph graph as a long-lived background process.

---

## C. How It Works — Technical Mechanics

### The Deep Agent Execution Loop

```
GOAL
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  PLAN PHASE                                          │
│  • LLM decomposes goal into ordered TODO list        │
│  • Plan written to workspace (e.g., plan.md or       │
│    structured state field)                           │
│  • Each task has: id, description, status, deps      │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  EXECUTE PHASE  (loop until plan complete)           │
│  • Select next pending task with satisfied deps      │
│  • Dispatch: inline tool call OR subagent spawn      │
│  • Write raw output to workspace file                │
│  • On success: mark task DONE in plan                │
│  • On failure: mark task FAILED, trigger replan      │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  CHECKPOINT PHASE                                    │
│  • Persist full plan state to disk                   │
│  • Log step summary (what ran, what it produced)     │
│  • Detect anomalies: loop detection, cost budget     │
└──────────────────────────┬───────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │ Any FAILED tasks?       │
             YES                       NO
              │                        │
              ▼                        ▼
┌─────────────────────┐    ┌──────────────────────────┐
│  REPLAN PHASE       │    │  AGGREGATE & RESPOND     │
│  • LLM diagnoses    │    │  • Collect outputs from  │
│    failure cause    │    │    workspace files        │
│  • Revises TODO     │    │  • Synthesize final       │
│    list (may add,   │    │    answer / artifact      │
│    remove, reorder) │    │  • Write final output     │
│  • Re-enters        │    │    to workspace           │
│    Execute phase    │    └──────────────────────────┘
└─────────────────────┘
```

### Subagent Invocation Mechanics

The orchestrator calls a `spawn_subagent(task, context_slice)` tool that:
1. Serializes the subtask description + relevant workspace context into a prompt
2. Instantiates a fresh agent graph with a narrower system prompt
3. Runs the child agent to completion (or timeout)
4. Returns structured output (result + status + any written artifacts)

Context passing is the primary design challenge — you must pass *enough* context for correctness while *avoiding* full workspace injection that would exceed context limits or dilute focus.

### SKILL.md — How It Works

The agent runtime checks a `skills/` directory at startup (or on demand). When the LLM determines a task maps to a skill category, it issues a `read_skill("skill_name")` tool call, receives the SKILL.md content, and then follows it as structured instructions for that subtask. This is essentially **few-shot prompting with deferred loading** — skills aren't injected into every prompt, only retrieved when relevant, keeping the base context lean.

### Filesystem-as-Context — Write Pattern

```
workspace/
├── plan.md           # live TODO list with statuses
├── research_raw.md   # raw web search output
├── analysis.md       # LLM synthesis of raw research
├── draft.md          # current artifact draft
└── final_output.md   # terminal artifact
```

Each file represents a checkpoint of an intermediate artifact. The agent reads prior files as context input for the next step, enabling a "rolling context" approach that works around finite context windows without summarization loss.

---

## D. Common Interview Questions (with Strong Answers)

---

**Q: How does a deep agent differ architecturally from a standard ReAct agent, and when do you choose one over the other?**

A: A ReAct agent is fundamentally reactive — each step is a single (thought, action, observation) cycle with no explicit global plan. This works well for tasks under ~10 steps where all required context fits comfortably in-window. A deep agent adds a durable, structured plan as a first-class data structure; the agent operates *against* that plan rather than free-forming each next action. The plan also functions as the coordination layer when you introduce subagents — children execute plan leaves while the orchestrator manages the DAG. The decision rule I use: if the task has more than ~8 discrete decisions, requires multi-file or multi-session state, or has steps with independent parallelism opportunities, reach for a deep agent. The overhead is real — plan initialization, checkpoint I/O, subagent latency — so for simple tool-use tasks, ReAct is leaner.

> **Staff-Level Extension**: A principal interviewer will push on *plan representation*. A flat TODO list breaks down when tasks have complex dependency graphs (e.g., step C requires both A and B, but B can start before A finishes). Production systems need a DAG-structured plan with explicit dependency edges and parallel execution support — which is a much harder problem for the LLM to reason about and maintain correctly.

---

**Q: How do you handle context passing to subagents without blowing up token budgets?**

A: The naive approach — passing the full workspace to every subagent — is expensive and often counterproductive because the child agent's attention dilutes across irrelevant content. The right pattern is explicit **context slicing**: the orchestrator decides *what* the subagent needs (task description, relevant prior outputs, applicable constraints) and constructs a focused prompt containing only that. I implement this as a `build_subagent_context(task_id, workspace)` function that looks up the task's declared dependencies in the plan, reads only those workspace files, and truncates with a budget. Results come back as structured objects (status, output_path, summary), and the orchestrator writes the child's artifacts to the workspace under a namespaced path (`subagents/task_03/output.md`). The failure mode to avoid is *implicit* context sharing through global mutable workspace state — if two parallel subagents both write to `analysis.md`, you have a race condition.

> **Staff-Level Extension**: At scale, this maps directly to the "RAG for agent memory" problem — you need retrieval over workspace artifacts, not just directory listing. For sufficiently large workspaces, the orchestrator should embed workspace summaries and retrieve contextually relevant ones rather than linearly scanning.

---

**Q: Walk me through how you'd implement reliable checkpointing for a deep agent that runs for 30+ minutes.**

A: Checkpointing has two dimensions: *what* to persist and *when*. For *what*: the full plan state (task statuses, outputs, any accumulated error context), all intermediate artifacts already written to the workspace, and the agent's step counter with a timestamp. I store this as a JSON checkpoint file (`workspace/.checkpoint.json`) updated after every task completion. For *when*: after every successful task mark-complete — not mid-task, because partial task state is harder to resume cleanly. On startup, the agent checks for an existing checkpoint and offers to resume from it (or start fresh if the checkpoint is stale by some TTL). The critical implementation detail is making tool calls **idempotent**: if a task is re-executed on resume because the checkpoint was written before the task was marked complete, the tool should detect prior output and skip re-execution. Without idempotency, resumed agents can produce duplicate side effects (multiple API calls, duplicate file writes).

> **Staff-Level Extension**: For true production reliability, you want to version-control the checkpoint alongside the workspace (or push to object storage with versioning). This lets you "time travel" to any past agent state for debugging — essential when an agent corrupts its own workspace after dozens of steps.

---

**Q: What are the failure modes specific to filesystem-as-context, and how do you mitigate them?**

A: Three primary failure modes. **Dirty state**: the agent writes a partial artifact (e.g., half-completed research.md) then fails — on the next run, it reads this corrupted file as authoritative input and produces garbage downstream. Mitigation: write to a temp file (`research.md.tmp`), then atomically rename on completion. **Stale context**: the agent reads an old version of a file because it doesn't know the file was updated by a subagent; you need explicit file invalidation or version tracking in the plan. **Semantic drift**: after many read-write cycles, intermediate files accumulate inconsistencies because each step only transforms a slice of prior state — the document as a whole drifts from the original goal. Mitigation: periodic "coherence check" tasks in the plan where the agent re-reads all artifacts against the original goal and flags contradictions. The deeper trade-off: filesystem state is human-auditable and resumable but lacks the transactional guarantees you'd get from a proper key-value store or database — for high-stakes workflows, consider wrapping filesystem ops with a thin transaction layer.

> **Staff-Level Extension**: This is structurally identical to the "eventually consistent distributed system" problem — agent steps are loosely coupled writers with no global coordinator enforcing consistency. Production solutions borrow ideas from distributed systems: write-ahead logs, tombstones for deleted content, vector clocks for concurrent subagent writes.

---

**Q: How do you design tools specifically for deep agents vs. standard tool use?**

A: Deep agent tools have different requirements than point-tool-use: they need to be **idempotent** (safe to re-run on resume), **atomic** (no partial writes observable by other tools), **descriptive in failure** (rich error messages the LLM can reason about, not just stack traces), and **context-efficient** (return summaries + artifact paths rather than dumping full content inline). For `write_file`, I enforce atomic write-via-temp-rename and return a content hash so the agent can verify write integrity. For `read_file`, I add a `max_tokens` budget parameter — the tool truncates and flags truncation explicitly so the agent knows it has partial context. For web search (Tavily), I cache results to a workspace file immediately so the agent doesn't re-query on resume. The anti-pattern I see most: tools that return large raw content inline — this blows up the context window and trains the agent to process in-context rather than write to disk, undermining the filesystem-as-context architecture.

---

**Q: What does loop detection look like in a deep agent, and where does it break down?**

A: Simple loop detection tracks `(task_id, attempt_count)` pairs — if the same task is attempted more than N times without a status change, it's stuck. But this misses *semantic* loops where the agent replans into a structurally different task that has the same effective action (e.g., "search for X" → fails → replan → "research X using web" → same Tavily call). A better heuristic fingerprints the *tool call signature* (tool name + key arguments hashed) and counts distinct-signature execution attempts per planning cycle. If the agent issues the same effective tool call three times with no new information, flag for human-in-the-loop escalation rather than replanning again. The breakdown: highly creative replanning can generate novel-looking task descriptions that translate to the same tool call — defeating signature-based detection. For production systems, combine signature detection with a global step budget and cost budget (token + API spend) as hard circuit breakers, and instrument every loop detection event for post-hoc analysis.

---

## E. Gotchas, Trade-offs & Best Practices

- **Plan quality is the dominant reliability lever.** Poor initial decomposition — tasks that are too coarse (can't be executed atomically), too fine (excessive coordination overhead), or have unmodeled dependencies — is responsible for the majority of deep agent failures in production. Invest in plan validation: after initial decomposition, have the LLM review the plan against the original goal *before* execution begins, checking for missing steps, circular dependencies, and ambiguous task boundaries.

- **Subagent failure isolation requires explicit error contracts.** If a subagent fails, the orchestrator must decide: retry the same subagent, spawn a differently-scoped subagent, mark the task failed and replan, or escalate to human. Without an explicit error taxonomy (transient infrastructure error vs. impossible task vs. missing context), the orchestrator will retry blindly, burning tokens on unrecoverable failures. Define error codes in your subagent return type and route them explicitly in orchestrator logic.

- **SKILL.md files are a double-edged sword.** They enable clean capability modularity and make behavior auditable and editable without redeployment — a huge operational advantage. But they introduce a runtime dependency on file system state: if the skills directory is corrupted, out of date, or missing, the agent silently loses capabilities. Version-control skills alongside agent code, validate skill schemas at startup, and log every skill invocation with the version hash of the skill file used.

- **Latency tolerance is a user expectation problem, not just a technical one.** Deep agents can run for minutes to hours — shipping one without a progress streaming mechanism (structured events: task started, task completed, replanning triggered) will erode user trust even if the final output is correct. Build streaming into the agent harness from day one; retrofitting it is painful. This also means every task in the plan needs a human-readable description, not just an ID.

- **Context window management is the hidden scaling limit.** As the workspace grows (more files, longer intermediate outputs), orchestrator prompts that include workspace summaries grow proportionally. Set hard limits on workspace artifact sizes, enforce summarization tasks in the plan for any artifact expected to exceed N tokens, and profile context window utilization as a first-class metric in production — treat context budget like memory budget in a traditional service.

---

## F. Code / Architecture Pattern

### Minimal Deep Agent: Structured Plan + Task Completion + Subagent Invocation

```python
import json
from pathlib import Path
from typing import Annotated, Literal
from dataclasses import dataclass, field, asdict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: str
    description: str
    status: Literal["pending", "in_progress", "done", "failed"] = "pending"
    result_path: str | None = None
    attempts: int = 0
    error: str | None = None


@dataclass
class AgentState:
    goal: str
    plan: list[Task] = field(default_factory=list)
    workspace: str = "./workspace"
    step_count: int = 0
    messages: Annotated[list, add_messages] = field(default_factory=list)

    def next_task(self) -> Task | None:
        return next((t for t in self.plan if t.status == "pending"), None)

    def is_complete(self) -> bool:
        return all(t.status in ("done", "failed") for t in self.plan)


# ── Workspace I/O tools ───────────────────────────────────────────────────────

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a workspace file atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.rename(p)  # atomic on POSIX
    return f"Written: {path} ({len(content)} chars)"


@tool
def read_file(path: str, max_chars: int = 8000) -> str:
    """Read a workspace file, truncating to max_chars."""
    p = Path(path)
    if not p.exists():
        return f"ERROR: {path} not found"
    content = p.read_text()
    if len(content) > max_chars:
        return content[:max_chars] + f"\n...[TRUNCATED at {max_chars} chars]"
    return content


@tool
def mark_task_done(task_id: str, result_path: str, state_path: str) -> str:
    """Mark a task complete and checkpoint plan state."""
    checkpoint = Path(state_path)
    plan_data = json.loads(checkpoint.read_text())
    for task in plan_data["plan"]:
        if task["id"] == task_id:
            task["status"] = "done"
            task["result_path"] = result_path
    checkpoint.write_text(json.dumps(plan_data, indent=2))
    return f"Task {task_id} marked done. Checkpoint updated."


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(state: AgentState) -> None:
    p = Path(state.workspace) / ".checkpoint.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "goal": state.goal,
        "step_count": state.step_count,
        "plan": [asdict(t) for t in state.plan],
    }, indent=2))


def load_checkpoint(workspace: str) -> dict | None:
    p = Path(workspace) / ".checkpoint.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


# ── Subagent invocation ───────────────────────────────────────────────────────

def spawn_subagent(task: Task, context_files: list[str], llm: ChatOpenAI) -> dict:
    """
    Run a child agent for a discrete subtask.
    Returns structured result: {status, output_path, summary, error}.
    Context is explicitly sliced — only files relevant to this task are passed.
    """
    context_content = ""
    for cf in context_files:
        p = Path(cf)
        if p.exists():
            context_content += f"\n\n### Context: {cf}\n{p.read_text()[:3000]}"

    subagent_prompt = f"""You are a focused subagent executing a single task.

TASK: {task.description}

AVAILABLE CONTEXT:
{context_content}

Execute the task. Return a JSON object with:
  - "status": "success" or "failure"
  - "output": the result content (will be written to disk by orchestrator)
  - "summary": 1-2 sentence summary for the orchestrator
  - "error": null or error description
"""
    response = llm.invoke(subagent_prompt)

    # In production: parse structured output with retry on JSON parse failure
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "status": "failure",
            "output": "",
            "summary": "Subagent returned non-JSON output",
            "error": response.content[:500],
        }
    return result


# ── Orchestrator nodes ────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def plan_node(state: AgentState) -> AgentState:
    """Decompose goal into structured task list."""
    existing = load_checkpoint(state.workspace)
    if existing and existing.get("plan"):
        # Resume from checkpoint — do not replan from scratch
        state.plan = [Task(**t) for t in existing["plan"]]
        state.step_count = existing.get("step_count", 0)
        return state

    prompt = f"""Break this goal into an ordered task list. Return JSON array.
Each task: {{"id": "t1", "description": "...", "status": "pending"}}

GOAL: {state.goal}"""

    response = llm.invoke(prompt)
    tasks_raw = json.loads(response.content)
    state.plan = [Task(**t) for t in tasks_raw]
    save_checkpoint(state)
    return state


def execute_node(state: AgentState) -> AgentState:
    """Execute next pending task, dispatching to subagent."""
    task = state.next_task()
    if not task:
        return state

    task.status = "in_progress"
    task.attempts += 1
    state.step_count += 1

    # Loop detection: bail if same task attempted too many times
    if task.attempts > 3:
        task.status = "failed"
        task.error = "Max attempts exceeded — escalating"
        save_checkpoint(state)
        return state

    # Context slicing: only pass files relevant to this task
    # In production: derive from task dependency metadata
    workspace_files = list(Path(state.workspace).glob("*.md"))
    context_files = [str(f) for f in workspace_files[-3:]]  # last 3 artifacts

    result = spawn_subagent(task, context_files, llm)

    if result["status"] == "success":
        output_path = f"{state.workspace}/{task.id}_output.md"
        Path(output_path).write_text(result["output"])
        task.status = "done"
        task.result_path = output_path
    else:
        task.status = "failed"
        task.error = result.get("error", "Unknown subagent error")

    save_checkpoint(state)
    return state


def replan_node(state: AgentState) -> AgentState:
    """Revise plan on failures — may add recovery tasks or mark unrecoverable."""
    failed = [t for t in state.plan if t.status == "failed"]
    if not failed:
        return state

    completed_summaries = "\n".join(
        f"- [{t.id}] {t.description}: DONE → {t.result_path}"
        for t in state.plan if t.status == "done"
    )
    failed_summaries = "\n".join(
        f"- [{t.id}] {t.description}: FAILED — {t.error}"
        for t in failed
    )

    prompt = f"""Agent plan encountered failures. Revise the remaining plan.

COMPLETED:
{completed_summaries}

FAILED:
{failed_summaries}

REMAINING (pending):
{[t.description for t in state.plan if t.status == "pending"]}

Return revised JSON task list for remaining work only.
If a failure is unrecoverable, omit dependent tasks."""

    response = llm.invoke(prompt)
    try:
        revised_raw = json.loads(response.content)
        # Replace pending + failed tasks with revised plan
        done_tasks = [t for t in state.plan if t.status == "done"]
        state.plan = done_tasks + [Task(**t) for t in revised_raw]
        save_checkpoint(state)
    except json.JSONDecodeError:
        # Replan failed — let execution complete with remaining tasks
        for t in failed:
            t.status = "pending"  # retry once more

    return state


def should_continue(state: AgentState) -> Literal["execute", "replan", "end"]:
    if state.is_complete():
        return "end"
    if any(t.status == "failed" for t in state.plan):
        return "replan"
    return "execute"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_deep_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("replan", replan_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges("execute", should_continue, {
        "execute": "execute",
        "replan": "replan",
        "end": END,
    })
    graph.add_edge("replan", "execute")

    return graph.compile()


# ── Usage ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = build_deep_agent_graph()
    final_state = agent.invoke(AgentState(
        goal="Research the impact of transformer architecture on recommendation systems "
             "and produce a structured report with key findings and open problems.",
        workspace="./workspace",
    ))
    print(f"Completed {final_state.step_count} steps")
    print(f"Plan summary: {[(t.id, t.status) for t in final_state.plan]}")
```

### SKILL.md Pattern — Example

```markdown
# SKILL: web_research

## When to Use
Invoke this skill when a task requires gathering current information
from the web that is not available in the workspace.

## Inputs Required
- `query`: The search query (be specific, include domain context)
- `output_path`: Workspace path to write raw results

## Steps
1. Call `tavily_search(query, max_results=5)`
2. For each result: extract title, URL, key excerpt (≤200 chars)
3. Write formatted results to `output_path` using `write_file`
4. Return summary: N results found, top source URLs

## Output Format
Markdown file with sections per result:
### [Title](URL)
> Key excerpt

## Failure Handling
- If Tavily returns 0 results: broaden query, retry once
- If Tavily errors: mark task failed with error code "SEARCH_UNAVAILABLE"
```

```python
# Agent reads skills at task dispatch time — not injected into every prompt

@tool
def read_skill(skill_name: str, skills_dir: str = "./skills") -> str:
    """Load a SKILL.md definition for the agent to follow."""
    p = Path(skills_dir) / skill_name / "SKILL.md"
    if not p.exists():
        return f"ERROR: Skill '{skill_name}' not found in {skills_dir}"
    return p.read_text()

# In orchestrator dispatch logic:
# if task requires web research → agent calls read_skill("web_research")
# then follows the SKILL.md instructions for that subtask
```

---

*Generated for AIE9 Session 7 — Deep Agents | April 2026*
