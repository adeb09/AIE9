# AI Engineering Study Guide — Per-Session Prompts

This file contains 19 self-contained prompts (one per bootcamp session + one synthesis prompt).
Each prompt is designed to be pasted directly into an LLM (Claude, GPT-4o, etc.).

**Usage:**
- Copy the prompt for the session you want to study
- Paste into your LLM of choice
- Save the output as `STUDY_GUIDE.md` inside the corresponding session folder
- Run the final **Synthesis Prompt** last and save its output as `AI_ENGINEERING_INTERVIEW_PREP.md` at the repo root

---

## TABLE OF CONTENTS

- [Session 1 — Vibe Check & AI Engineering Foundations](#session-1)
- [Session 2 — Dense Vector Retrieval & Embeddings](#session-2)
- [Session 3 — The Agent Loop (LangChain / LCEL)](#session-3)
- [Session 4 — Agentic RAG from Scratch (LangGraph + ReAct)](#session-4)
- [Session 5 — Multi-Agent Systems with LangGraph](#session-5)
- [Session 6 — Agent Memory (CoALA Framework)](#session-6)
- [Session 7 — Deep Agents (Planning, Subagents, Skills)](#session-7)
- [Session 8 — Open Deep Research](#session-8)
- [Session 9 — Synthetic Data Generation & LangSmith](#session-9)
- [Session 10 — Evaluating RAG & Agents with Ragas](#session-10)
- [Session 11 — Advanced Retrieval Strategies](#session-11)
- [Session 12 — Industry Use Cases & Capstone Design](#session-12)
- [Session 14 — MCP Connectors](#session-14)
- [Session 15 — LangGraph Deployments](#session-15)
- [Session 16 — LLM Servers & Dedicated Endpoints](#session-16)
- [Session 17 — MCP Servers & Agent-to-Agent (A2A)](#session-17)
- [Session 18 — Production RAG, Guardrails & Caching](#session-18)
- [Synthesis Prompt — Cheat Sheet & 10 Hardest Questions](#synthesis-prompt)

---

---

## Session 1

### PROMPT — Session 1: Vibe Check & AI Engineering Foundations

> Save output as: `01_Vibe_Check/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Vibe Check & AI Engineering Foundations**

Cover all of the following concepts in depth:
- Definition of AI Engineering vs. ML Engineering vs. Data Science — where the roles overlap and where they diverge, especially in terms of system ownership and production responsibilities
- Three core AI application patterns: Prompting, RAG, Agents — when each is the right tool and when each is the wrong one
- What "vibe checking" means as a fast-iteration evaluation heuristic before committing to rigorous eval infrastructure
- Limitations of LLMs: knowledge cutoffs, hallucination, context window constraints, tool/real-time limits, and non-determinism
- The AI Engineering feedback loop: Build → Deploy → Improve — and how it differs from the ML training loop
- How to frame an AI Engineering project from business problem to technical approach

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Use numbered steps, decision trees, or structured comparisons where appropriate.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative code snippet or architecture diagram (described in text or Python) illustrating the key concept from this session.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Understands embeddings, retrieval, and offline evaluation deeply. Now applying those foundations in LLM-powered production systems.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 2

### PROMPT — Session 2: Dense Vector Retrieval & Embeddings

> Save output as: `02_Dense_Vector_Retrieval/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Dense Vector Retrieval & Embeddings**

Cover all of the following concepts in depth:
- What embeddings are and how they encode semantic meaning — the geometry of embedding space and what proximity means
- Similarity metrics: cosine similarity vs. dot product vs. Euclidean distance — mathematical properties, when each is appropriate, and practical differences when using normalized vs. unnormalized vectors
- Vector databases: in-memory (e.g., custom `VectorDatabase` class) vs. managed solutions (Qdrant, Pinecone, Weaviate) — indexing strategies (flat, HNSW, IVF), trade-offs at scale
- Building a vector store from scratch in Python: text chunking strategies, embedding via OpenAI API, indexing and querying
- In-context learning vs. dense retrieval as two distinct augmentation strategies — when RAG adds value vs. when a well-prompted base model is sufficient
- OpenAI Embeddings API: `text-embedding-3-small` vs. `text-embedding-3-large` — dimensionality, cost, latency, quality trade-offs
- Chunking strategies: fixed-size, sentence-based, paragraph-based — how chunk size affects retrieval quality and generation quality independently
- The `aimakerspace` utility pattern: `TextFileLoader`, `CharacterTextSplitter`, custom `VectorDatabase`

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Use numbered steps, decision trees, or structured comparisons where appropriate.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative code snippet or architecture diagram illustrating the key concept — building a simple vector store and querying it.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Understands embeddings, retrieval, and offline evaluation deeply. Now applying those foundations in LLM-powered production systems. Note: draw explicit contrasts between recommendation-system embeddings (collaborative filtering, user/item towers) and RAG-style text embeddings where useful — this person will find those comparisons illuminating rather than confusing.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer (e.g., "How would you scale this to 100M documents?").

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 3

### PROMPT — Session 3: The Agent Loop (LangChain / LCEL)

> Save output as: `03_The_Agent_Loop/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: The Agent Loop (LangChain / LCEL)**

Cover all of the following concepts in depth:
- LangChain Expression Language (LCEL): the `Runnable` interface, pipe (`|`) composition, `RunnablePassthrough`, `RunnableLambda`, `RunnableParallel` — the design philosophy and when LCEL adds value vs. adds friction
- `create_react_agent()` — what it abstracts, what the underlying prompt template looks like, and what you lose by using the abstraction vs. building from scratch
- The ReAct loop in detail: Thought → Action → Observation → (loop) → Final Answer — how the model reasons across steps, how tool results are injected back into context
- Tools: `@tool` decorator, `StructuredTool`, tool schemas (JSON Schema / Pydantic), tool binding to chat models via `bind_tools()`
- Qdrant as a production vector store: collection management, distance metrics configuration, sparse vs. dense payload filtering
- Middleware patterns in the agent loop: logging, rate limiting, retry, input/output transformation
- Human-in-the-loop (HITL): interrupt strategies, `input()` patterns in notebooks vs. production HITL via async queues
- Error handling in agent loops: max iterations, tool execution errors, malformed tool calls, graceful degradation

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Use numbered steps, decision trees, or structured comparisons where appropriate.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative code snippet showing a LangChain LCEL RAG chain and a basic ReAct agent with a custom tool.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Understands system reliability and API design patterns deeply.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 4

### PROMPT — Session 4: Agentic RAG from Scratch (LangGraph + ReAct)

> Save output as: `04_Agentic_RAG_From_Scratch/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Agentic RAG from Scratch (LangGraph + ReAct)**

Cover all of the following concepts in depth:
- LangGraph core abstractions: `StateGraph`, nodes (Python functions), edges (deterministic), conditional edges (dynamic routing), `END` node — how this compares to a state machine and where the analogy breaks down
- State management: `TypedDict` as state schema, `Annotated` fields with reducer functions (e.g., `operator.add` for message accumulation), how state flows through the graph
- Building ReAct from scratch with LangGraph: no `create_react_agent` wrapper — implementing the reasoning node, tool execution node, and routing logic manually
- Conditional routing patterns: how to inspect state to decide whether to continue looping or terminate
- RAG as a LangGraph tool: wiring a retriever as a callable tool inside the graph vs. as a standalone node
- `interrupt` for HITL in LangGraph: how interrupts pause graph execution, how to resume, use cases for human approval gates
- Ollama for local inference: model management (`ollama pull`), OpenAI-compatible API surface, when local models make sense vs. API models (latency, cost, data privacy, quality)
- Debugging LangGraph graphs: `graph.get_graph().draw_mermaid()`, inspecting state at each step, common graph construction errors

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Use numbered steps, decision trees, or structured comparisons where appropriate. Include a text diagram of the ReAct graph topology.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative LangGraph `StateGraph` implementation showing the ReAct loop: state definition, reasoning node, tool node, conditional routing, and graph compilation.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Familiar with state machines and DAG-based workflow systems.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 5

### PROMPT — Session 5: Multi-Agent Systems with LangGraph

> Save output as: `05_Multi_Agent_with_LangGraph/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Multi-Agent Systems with LangGraph**

Cover all of the following concepts in depth:
- Supervisor pattern: a central orchestrator LLM that routes tasks to specialized sub-agents — how routing is implemented (structured output, tool call to `transfer_to_X`), failure modes when the supervisor hallucinates a route
- Handoff pattern: agents transfer control laterally via explicit tool calls — how `Command(goto=...)` works in LangGraph, differences from supervisor in terms of coupling and observability
- When to use multi-agent vs. single-agent: complexity thresholds, specialization benefits, parallelism opportunities, and the overhead cost of coordination
- Shared state vs. private state in multi-agent graphs: what each sub-agent can see, how to prevent cross-contamination of context
- Tavily search tool: API integration, result structure, how to ground agent responses in real-time web data
- LangSmith tracing in multi-agent systems: run tree structure, how to find the root trace vs. child traces, using traces to debug routing failures
- Common failure modes: infinite supervisor loops, context window exhaustion across agent hops, tool call storms, agents disagreeing on facts

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a text diagram comparing Supervisor topology vs. Handoff topology.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative LangGraph multi-agent graph showing a supervisor node routing to two specialized sub-agents, with LangSmith tracing enabled.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Familiar with distributed systems, microservices, and service orchestration patterns.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 6

### PROMPT — Session 6: Agent Memory (CoALA Framework)

> Save output as: `06_Agent_Memory/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Agent Memory (CoALA Framework)**

Cover all of the following concepts in depth:
- CoALA (Cognitive Architectures for Language Agents) memory taxonomy: the academic framing and how it maps to practical LangGraph implementation
- Four memory types in detail:
  - **Working memory**: the current context window — what's in it, how to manage it, trimming vs. summarization strategies
  - **Episodic memory**: records of past interactions — how to store, retrieve, and surface relevant episodes at query time
  - **Semantic memory**: factual knowledge about users, entities, preferences — how to structure it in a persistent store
  - **Procedural memory**: skills and instructions the agent has learned — how system prompt injection implements this
- Short-term memory in LangGraph: `messages` state field, `trim_messages()`, `summarize_messages()` — trade-offs between trimming (information loss) and summarization (latency, hallucination risk)
- Long-term memory in LangGraph: `InMemoryStore` vs. persistent backends (Postgres, Redis), namespaces (e.g., `("user", user_id, "preferences")`), `store.put()` / `store.search()`
- LangGraph Studio (`langgraph dev`): visual state inspection, thread management, replay — how to use it for debugging memory issues
- Memory retrieval strategies: recency-weighted, relevance-weighted (semantic search over memories), hybrid
- Production memory design: privacy considerations, memory TTL, memory conflicts (contradictory beliefs), memory poisoning attacks

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a structured comparison table of the four memory types: storage location, retrieval mechanism, LangGraph implementation, typical use case.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative LangGraph agent snippet showing both short-term message trimming and long-term memory read/write using LangGraph `Store`.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Familiar with caching systems, user-state management, and personalization infrastructure.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 7

### PROMPT — Session 7: Deep Agents (Planning, Subagents, Skills)

> Save output as: `07_Deep_Agents/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Deep Agents (Planning, Subagents, Skills)**

Cover all of the following concepts in depth:
- What distinguishes a "deep agent" from a standard ReAct agent: long-horizon planning, persistent workspace context, multi-step task decomposition, tolerance for latency
- Planning and TODO tracking inside an agent: how to implement a structured plan (e.g., a task list in agent state or on-disk), how the agent marks tasks complete, when to replan
- Subagents: spawning child agents from a parent — orchestrator/worker patterns, how context is passed to subagents, how results are aggregated, failure isolation
- Skills: reusable capability definitions encoded as `SKILL.md` files — how an agent reads skill definitions, when to use skills vs. inline tool logic
- Filesystem-as-context: agents reading/writing markdown files to disk as persistent working memory — why this is powerful for long-horizon tasks and where it creates risks (dirty state, partial writes)
- Tool design for deep agents: file I/O tools (`read_file`, `write_file`, `list_directory`), web search (Tavily), code execution
- Reliability patterns for long-running agents: checkpointing progress, detecting loops, self-correction on error, graceful timeout and resumption
- The `deepagents-cli` pattern: what it provides, how it differs from running a LangGraph graph directly

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a text description of the deep agent execution loop: plan → execute → checkpoint → replan.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative code pattern showing a deep agent with: a structured plan in state, a tool for marking tasks done, and a subagent invocation pattern.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Familiar with workflow orchestration and distributed task execution.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 8

### PROMPT — Session 8: Open Deep Research (Multi-Step Research Pipelines)

> Save output as: `08_Open_DeepResearch/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Open Deep Research (Multi-Step Research Pipelines)**

Cover all of the following concepts in depth:
- The Deep Research pattern: query → plan research subtopics → parallel web/document search → synthesize section-by-section → produce final structured report — how this differs architecturally from a standard RAG pipeline
- LangGraph orchestration of branching research workflows: `Send` API for dynamic parallelism (fan-out), collecting parallel results back into shared state (fan-in), ordering parallel results deterministically
- State design for long research tasks: what goes in state (plan, section drafts, sources, metadata), how to checkpoint so a crashed run can resume rather than restart
- Multi-provider LLM routing: using different models for different subtasks (e.g., fast/cheap model for search queries, powerful model for synthesis) — routing logic, fallback strategies
- External data source integration: web search (Tavily, Exa, DuckDuckGo), academic (Arxiv), databases (Supabase), cloud search (Azure AI Search) — normalizing results across sources
- Structured report generation: how to enforce section schema (Pydantic models, structured output), avoiding hallucinated citations
- Quality vs. cost vs. latency triangle in deep research: how to tune the number of search queries, sources per query, and synthesis passes
- Where Deep Research breaks: source quality issues, conflicting information across sources, topic drift in long runs

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a text diagram of the full Deep Research graph: planner → parallel searchers → section writers → final synthesizer.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative LangGraph snippet showing the `Send` API for fan-out parallel research, and how results are collected in the parent state.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Understands distributed computation, batch inference pipelines, and data quality challenges.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 9

### PROMPT — Session 9: Synthetic Data Generation & LangSmith

> Save output as: `09_Synthetic_Data_Generation_and_LangSmith/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Synthetic Data Generation & LangSmith Evaluation**

Cover all of the following concepts in depth:
- The cold start problem in RAG evaluation: why you can't evaluate a system you just built without labeled data, and why synthetic data is the pragmatic solution
- RAGAS `TestsetGenerator`: how it processes source documents to produce question-context-answer triples, the underlying synthesis pipeline (chunking → question generation → answer generation → filtering)
- Evol Instruct-style question evolution: transforming simple questions into reasoning-heavy, multi-hop, or conditional variants — what each evolution type tests and when it matters
- LangSmith concepts: datasets (versioned, immutable ground truth), experiments (a run of `evaluate()` against a dataset), run trees (parent-child trace hierarchy), annotation queues
- The `evaluate()` API: `langsmith.evaluate()` signature, custom evaluator functions, built-in evaluators (correctness, faithfulness, etc.), how results are stored and compared
- Iterating on a RAG pipeline with LangSmith: comparing experiment A vs. B (e.g., chunk size 256 vs. 512), reading the results table, statistical significance considerations
- The LCEL RAG chain anatomy: `retriever | prompt | llm | StrOutputParser` — what each component does, how to swap components for ablation
- RAG hyperparameters: chunk size, chunk overlap, embedding model, top-k retrieval — how each affects retrieval recall and generation faithfulness independently

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include numbered steps for: (1) synthetic data generation pipeline and (2) LangSmith experiment loop.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative code snippet showing: (1) generating a RAGAS testset from documents and (2) running a LangSmith `evaluate()` call with a custom evaluator.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Highly familiar with offline evaluation frameworks, A/B test design, and dataset construction. Draw explicit parallels to recommendation system evaluation (offline metrics → online metrics, dataset bias, label quality).
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 10

### PROMPT — Session 10: Evaluating RAG & Agents with Ragas

> Save output as: `10_Evaluating_RAG_With_Ragas/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Evaluating RAG & Agents with Ragas**

Cover all of the following concepts in depth:
- Ragas core RAG metrics in depth — how each is computed, what it measures, and what a low score tells you about the pipeline:
  - `faithfulness`: are claims in the answer supported by the retrieved context? (LLM-as-judge, NLI-based)
  - `answer_relevancy`: does the answer address the question? (embedding-based reverse question generation)
  - `context_recall`: did retrieval surface the chunks needed to answer? (requires ground truth)
  - `context_precision`: are the retrieved chunks relevant (not just sufficient)?
- The Ragas `EvaluationDataset` format: required fields (`user_input`, `retrieved_contexts`, `response`, optional `reference`)
- `ragas.evaluate()` pipeline: metrics selection, LLM judge configuration, result `Dataset` object, converting to pandas for analysis
- Agent evaluation with Ragas: `ToolCallAccuracy`, `AgentGoalAccuracy`, `TopicAdherence` — what each measures and how to construct evaluation datasets for agents (reference tool call sequences)
- Cohere Rerank API: cross-encoder mechanics (why it's more accurate but slower than bi-encoder retrieval), `co.rerank()` call, integrating into a LangChain retriever pipeline
- LLM-as-judge design: when it's appropriate, prompt design for judges, calibration against human labels, cost/latency implications
- When NOT to use LLM-as-judge: circular evaluation (using GPT-4 to judge GPT-4 outputs), high-stakes decisions, latency-sensitive pipelines

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a table mapping each Ragas metric to: what it measures, computation method, what a low score implies, and how to fix it.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative code snippet showing: a Ragas evaluation run on a RAG pipeline, and an agent evaluation with `ToolCallAccuracy`.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Deeply familiar with evaluation metrics (NDCG, MRR, precision@k), offline vs. online eval, and the dangers of metric gaming. Draw explicit parallels to search/recommendation evaluation where useful.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 11

### PROMPT — Session 11: Advanced Retrieval Strategies

> Save output as: `11_Advanced_Retrieval/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Advanced Retrieval Strategies**

Cover all of the following concepts in depth:
- Why naive top-k dense retrieval breaks at scale: vocabulary mismatch, embedding model domain gaps, long-tail queries, exact-match needs
- BM25 (sparse retrieval): TF-IDF mechanics, the BM25 saturation and length normalization parameters (k1, b), strengths for keyword/exact-match queries, `rank-bm25` Python library
- Hybrid search: combining BM25 + dense vector retrieval — score normalization problem, fusion strategies, why ensemble consistently outperforms either retriever alone
- Reciprocal Rank Fusion (RRF): the formula (`1 / (k + rank)`), why it's robust to score scale differences, how to implement it as a custom LangChain retriever
- Multi-query retrieval: using an LLM to generate N query variants from the original question, running all N retrievals, deduplicating results — when this helps (ambiguous queries) and when it hurts (cost, latency)
- Parent Document Retriever: child chunks (small, precise) for indexing/scoring, parent chunks (large, contextual) for generation — how to implement the two-tier store in LangChain
- Semantic chunking: embedding-based sentence boundary detection, `SemanticChunker` in LangChain — trade-offs vs. fixed-size chunking
- RAG-Fusion: multi-query + RRF combined — the full pipeline and when the extra complexity pays off
- Reranking as a second-stage retriever: cross-encoder mechanics (Cohere, BGE), latency/quality trade-offs, where to place reranking in the retrieval pipeline
- Evaluating retrieval independently of generation: context precision@k, context recall, how to isolate retrieval quality from LLM quality in Ragas

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a decision tree: given query characteristics (keyword-heavy, semantic, ambiguous, long-tail), which retrieval strategy to choose.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative code snippet showing: a hybrid BM25 + dense ensemble retriever with RRF in LangChain, and a Parent Document Retriever setup.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Deeply familiar with information retrieval, two-tower models, ANN search, and retrieval evaluation. Draw explicit parallels to production search system design — this person has likely built or worked with Elasticsearch/Solr, FAISS/ScaNN, and knows recall vs. precision trade-offs intimately.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 12

### PROMPT — Session 12: Industry Use Cases & Capstone Design

> Save output as: `12_Certification_Challenge/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Industry Use Cases & Capstone System Design**

Cover all of the following concepts in depth:
- "What not to build": a rigorous decision framework for when RAG/agents are the wrong solution — keyword search suffices, DB query is better, the task is too deterministic, hallucination risk is unacceptable
- The AI Engineering project lifecycle: problem scoping → prototype (vibe check) → structured eval → retrieval upgrade → production hardening → monitoring
- Capstone architecture for an agentic RAG system: async data ingestion → preprocessing/chunking → hybrid vector store → LangGraph agent → evaluation loop — key design decisions at each layer
- Async Python for AI engineering: `asyncio`, `aiohttp` for concurrent API calls, semaphores for rate-limit compliance, `stamina` for retry-with-backoff on transient failures
- GitHub API integration patterns: `PyGithub` (sync), `gidgethub` (async) — fetching files, markdown, code at scale; pagination, rate limits, caching strategies
- End-to-end RAG system design decisions: embedding model selection, vector DB selection, chunk strategy selection, retrieval strategy selection, eval framework — how to justify each choice
- How to communicate AI system design to non-technical stakeholders: framing uncertainty, demonstrating eval results, setting reliability expectations

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a full system architecture walkthrough for a production-grade agentic RAG system (ingestion → retrieval → agent → eval loop).

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative async Python snippet showing: concurrent document ingestion with `aiohttp` + semaphore rate limiting + `stamina` retry, feeding into a vector store.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Experienced with production system design, reliability engineering, and stakeholder communication.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 14

### PROMPT — Session 14: Model Context Protocol (MCP) Connectors

> Save output as: `14_MCP_Connectors/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Model Context Protocol (MCP) Connectors**

Cover all of the following concepts in depth:
- What MCP is: Anthropic's open standard for connecting LLMs to external tools and data — the protocol specification (JSON-RPC over stdio/SSE), the three primitive types (tools, resources, prompts)
- Why MCP exists: the N×M problem (N models × M tools = N×M integrations without a standard), and how MCP reduces this to N+M
- MCP vs. LangChain tools: protocol-level vs. library-level tool integration — when you want the portability of MCP and when LangChain tools are simpler
- `langchain-mcp-adapters`: the `MultiServerMCPClient` — how it connects to MCP servers (stdio vs. SSE transport), converts MCP tool schemas to LangChain `BaseTool` objects
- GitHub-as-MCP: reading PRs, issues, file contents via `github-mcp-server` — practical patterns for using GitHub as a knowledge source in an agent
- X/Twitter API integration as LangChain tools: OAuth 2.0 flow, API v2 rate limits, tool schema design for tweet search and posting
- Social listening pipeline design: tweet ingestion → LLM classification → structured issue creation in GitHub — end-to-end pattern and failure modes
- MCP security model: credential scoping, tool permission boundaries, risks of prompt injection via MCP tool results

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a text diagram of the MCP client-server communication flow.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative code snippet showing: connecting to a GitHub MCP server via `MultiServerMCPClient`, loading tools, and using them inside a LangGraph agent.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background with deep systems experience. Familiar with API standards, RPC protocols, and OAuth flows.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 15

### PROMPT — Session 15: LangGraph Deployments (Serving Agents)

> Save output as: `15_LangGraph_Deployments/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: LangGraph Deployments (Serving Agents)**

Cover all of the following concepts in depth:
- `langgraph dev`: what it starts (a local LangGraph API server), the default port (2024), the `/runs`, `/threads`, `/assistants` REST API endpoints, hot reload behavior
- `langgraph.json` anatomy: `graphs` key (mapping assistant name → Python import path), `env` (env file path), `dependencies` (packages to install) — what each field controls
- Assistants in LangGraph: the concept of a named, versioned, configurable instance of a graph — how `config` is used to parameterize behavior at runtime without code changes
- Testing a deployed LangGraph agent: the `RemoteGraph` client pattern, streaming vs. blocking invocation, how to construct a proper `HumanMessage` input
- Graph design for production: stateless (no checkpointer) vs. stateful (Postgres/Redis checkpointer) — thread IDs, resumability, cost of persistent state
- Checkpointers: `MemorySaver` (dev only), `AsyncPostgresSaver`, `AsyncRedisSaver` — when each is appropriate
- FastMCP integration: consuming an external MCP server's tools from inside a deployed LangGraph graph — tool loading at startup vs. at request time
- Production deployment considerations: environment variable injection, dependency management with `uv`, Docker image building, horizontal scaling of stateless vs. stateful graphs

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a walkthrough of the full deployment lifecycle: `langgraph.json` → `langgraph dev` → REST API → client invocation.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative set of snippets showing: (1) a `langgraph.json` config, (2) a simple agent graph with checkpointer, and (3) a test client using `RemoteGraph`.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background with deep backend and systems experience. Familiar with REST APIs, service deployment, Docker, and distributed state management.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 16

### PROMPT — Session 16: LLM Servers & Dedicated Endpoints (Fireworks AI)

> Save output as: `16_LLM_Servers/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: LLM Servers & Dedicated Endpoints (Fireworks AI)**

Cover all of the following concepts in depth:
- The LLM serving landscape: serverless inference (pay-per-token, cold starts, shared infrastructure) vs. dedicated endpoints (reserved GPU, predictable latency, higher baseline cost) — decision framework for choosing
- Fireworks AI platform: model catalog (Llama, Mistral, Mixtral, Qwen, etc.), dedicated endpoint provisioning, the OpenAI-compatible API surface
- Key LLM serving metrics: TTFT (Time to First Token), TPOT (Time Per Output Token), throughput (tokens/sec), P50/P99 latency — how to measure each and what drives them
- "Endpoint slamming" / load testing: concurrent request patterns, how to measure throughput ceiling, identifying the knee of the performance curve
- Embeddings via open-source model endpoints: using a Fireworks embedding endpoint as a drop-in replacement for OpenAI embeddings — dimension compatibility, quality comparison, cost comparison
- Building a full RAG pipeline on open-source endpoints: wiring Fireworks LLM + Fireworks embeddings + Qdrant in a LangGraph agent
- When to use open-source models vs. frontier APIs: capability gap analysis, total cost of ownership, data privacy/compliance requirements, customization (fine-tuning), latency SLAs
- Responsible endpoint management: cost monitoring, automatic shutdown, reserved capacity pricing models

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a comparison table: serverless vs. dedicated endpoint across latency, cost, throughput, cold start, and data privacy dimensions.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative code snippet showing: initializing a Fireworks LLM and embedding model via LangChain, building a simple RAG chain, and running a basic throughput test.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background with infrastructure and systems experience. Familiar with cloud cost optimization, SLAs, and capacity planning.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 17

### PROMPT — Session 17: MCP Servers & Agent-to-Agent (A2A) Communication

> Save output as: `17_MCP_A2A/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: MCP Servers & Agent-to-Agent (A2A) Communication**

Cover all of the following concepts in depth:
- Building an MCP server with `mcp[cli]`: the `@mcp.tool()` decorator, `@mcp.resource()`, server startup (`mcp.run()`), stdio vs. SSE transport — what each transport is for
- OAuth in MCP servers: why an MCP server needs OAuth (acting on behalf of a user to call external APIs), the OAuth 2.0 PKCE flow for CLI/local servers, token storage and refresh
- `ngrok` for local MCP server exposure: tunnel mechanics, how a remote LLM client discovers the local server URL, security considerations for development tunnels
- Agent-to-Agent (A2A) protocol: Google's open standard — the `AgentCard` (capability advertisement), `Task` lifecycle (submitted → running → completed/failed), streaming responses via SSE
- `a2a-sdk` patterns: `AgentExecutor` (server-side: receives a `Task`, executes agent logic, yields updates), `A2AClient` (client-side: submits tasks, polls/streams results)
- MCP vs. A2A: fundamental design differences — MCP is tool/resource access (synchronous RPC semantics), A2A is agent delegation (async task semantics) — when to use which
- Composing agent networks: an orchestrator agent using MCP for tool access AND A2A for delegating to specialized sub-agents — the combined architecture
- Security and trust in agent networks: authentication between agents, capability scoping, preventing agent impersonation

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a side-by-side comparison of MCP and A2A: communication model, transport, use case, Python SDK, and typical topology.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative pair of snippets: (1) a simple MCP server exposing a tool with `@mcp.tool()`, and (2) an A2A `AgentExecutor` skeleton that receives a task and streams back results.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background with systems and API design experience. Familiar with RPC frameworks, async task queues, and distributed service communication patterns.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Session 18

### PROMPT — Session 18: Production RAG, Guardrails & Caching

> Save output as: `18_Production_RAG_and_Guardrails/STUDY_GUIDE.md`

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has completed a 10-week AI Engineering bootcamp. I am preparing for senior and staff-level AI Engineering interviews.

Generate a focused, interview-ready study guide for the following session topic:

**SESSION: Production RAG, Guardrails & Caching**

Cover all of the following concepts in depth:
- The prototype-to-production gap in AI systems: what changes beyond the happy path — adversarial inputs, schema drift, latency SLAs, cost at scale, compliance requirements
- Guardrails AI framework: the hub guard model (pre-built validators), `Guard` object, `@guard.validate` pattern, input guards vs. output guards — how to wrap a LangGraph agent
- Key guard types and their mechanics:
  - Topic restriction: embedding-based similarity to forbidden topics
  - Jailbreak detection: pattern-based + LLM-based detection
  - Competitor mention: named entity detection
  - RAG faithfulness: cross-checking generated answer against retrieved context
  - Profanity filtering: regex + model-based
- Caching strategies for LLM applications:
  - Exact-match cache: hash of (prompt, model, params) → cached response — highest precision, lowest recall
  - Semantic cache: embed the query, find nearest cached query above similarity threshold — higher recall, false positive risk
  - SQLite / Redis / vector DB as cache backends — trade-offs at different scale points
- Cache invalidation in LLM apps: when cached responses go stale (source document updates, model version changes), TTL strategies
- Docker for AI app deployment: `Dockerfile` for a LangGraph app, multi-stage builds, environment variable injection, image size optimization
- Observability in production: LangSmith traces for every request, custom metadata tagging, guard violation dashboards, latency P99 monitoring, cost per request tracking
- Common production failure modes: prompt injection through user input, guardrail bypass via encoding tricks, cache poisoning, context window overflows under load

---

For this topic, produce a study guide with the following six sections:

### A. Core Concept Summary
3–5 sentences explaining the topic, why it matters in AI Engineering, and the key mental model a practitioner must hold.

### B. Key Terms & Definitions
Bulleted glossary of 5–10 terms specific to this topic. 1–2 sentence definition for each.

### C. How It Works — Technical Mechanics
A precise technical explanation of the mechanisms at play. Include a layered architecture diagram (described in text) of a production RAG system: input guardrails → retrieval → generation → output guardrails → cache write.

### D. Common Interview Questions (with strong answers)
4–6 interview questions a staff/principal engineer or hiring manager would ask. For each, provide a model answer (3–6 sentences) labeled **Q:** and **A:** that reflects senior/staff-level thinking: trade-offs, failure modes, quantified decisions.

### E. Gotchas, Trade-offs & Best Practices
3–5 bullet points covering what trips up practitioners, common architectural trade-offs, and production best practices.

### F. Code or Architecture Pattern
A minimal but representative code snippet showing: wrapping a LangChain chain with Guardrails AI input + output guards, and adding a semantic cache layer.

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background with production reliability and systems experience. Familiar with caching infrastructure, observability stacks, and security considerations in production services.
- Do NOT explain basics. Go deep on trade-offs, failure modes, and system design decisions.
- Every answer should sound like a senior practitioner who has shipped production AI systems.
- Where relevant, include a **Staff-Level Extension** — a follow-up angle a principal interviewer would push on after a correct base answer.

Use Markdown with clear `##` and `###` headers. Code in Python fenced blocks. Begin the study guide now.

---

---

## Synthesis Prompt

### PROMPT — Synthesis: LLM Stack Cheat Sheet & 10 Hardest Interview Questions

> Save output as: `AI_ENGINEERING_INTERVIEW_PREP.md` at the repo root

---

You are an expert AI Engineering educator and technical interviewer. I am a Senior/Staff-level software engineer (5–10+ years of industry experience) who has just completed a 10-week AI Engineering bootcamp covering the following 18 sessions:

1. Vibe Check & AI Engineering Foundations
2. Dense Vector Retrieval & Embeddings
3. The Agent Loop (LangChain / LCEL)
4. Agentic RAG from Scratch (LangGraph + ReAct)
5. Multi-Agent Systems with LangGraph
6. Agent Memory (CoALA Framework)
7. Deep Agents (Planning, Subagents, Skills)
8. Open Deep Research (Multi-Step Research Pipelines)
9. Synthetic Data Generation & LangSmith Evaluation
10. Evaluating RAG & Agents with Ragas
11. Advanced Retrieval Strategies
12. Industry Use Cases & Capstone Design
13. Full-Stack Agent Apps
14. MCP Connectors (Model Context Protocol)
15. LangGraph Deployments
16. LLM Servers & Dedicated Endpoints (Fireworks AI)
17. MCP Servers & Agent-to-Agent (A2A) Communication
18. Production RAG, Guardrails & Caching

The stack used throughout includes: LangChain, LangGraph, LangSmith, Ragas, Qdrant, OpenAI, Anthropic, Ollama, Fireworks AI, Cohere, Tavily, Guardrails AI, MCP, A2A SDK, `uv`, Docker.

Generate two synthesis artifacts:

---

## ARTIFACT 1: LLM Stack Cheat Sheet

A comprehensive reference table with the following columns:

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---------|---------------------|-----------------------|----------------------|

Include one row for every major concept across all 18 sessions. Cover at minimum:
- All major LangChain/LCEL building blocks
- All major LangGraph building blocks (StateGraph, Send, interrupt, checkpointers, Store)
- All Ragas metrics and eval constructs
- All retrieval strategies (BM25, hybrid, RRF, multi-query, parent doc, semantic chunking, rerank)
- Memory types and LangGraph memory primitives
- MCP primitives (tool, resource, prompt, transport types)
- A2A primitives (AgentCard, Task, AgentExecutor)
- Guardrails AI constructs
- Caching patterns
- LangSmith constructs (dataset, experiment, evaluator, run tree)
- Serving constructs (langgraph.json, assistants, RemoteGraph)
- LLM serving metrics (TTFT, TPOT, throughput)

---

## ARTIFACT 2: 10 Hardest AI Engineering Interview Questions

These should be cross-cutting, synthesis-level questions that span multiple sessions — the kind a principal engineer or engineering director asks to assess whether a candidate truly understands the full stack, not just individual tools.

For each question:
- Label it **Q[N]:** with the question
- Provide a thorough **A[N]:** model answer (8–12 sentences) that demonstrates staff-level breadth and depth
- End each answer with a **Staff-Level Follow-Up:** — the next question a principal interviewer would ask to dig deeper

The 10 questions must cover:
1. RAG vs. fine-tuning vs. prompting — when to use which and why, with cost/latency/quality trade-offs
2. How you would design and evaluate a production RAG system from scratch for a corpus of 10M documents with sub-100ms P99 latency requirement
3. How you think about agent reliability — what makes agents fail, how you detect it, how you design against it
4. The trade-offs between the Supervisor pattern and the Handoff pattern in multi-agent systems — when each breaks down
5. How you would build a complete eval strategy for an agentic system with no ground truth labels
6. How memory architecture choices affect agent behavior at scale — context window limits, cost, consistency
7. The difference between MCP and A2A — when to use which, and how they compose in a real system
8. How you would decide whether to use an open-source LLM endpoint vs. a frontier API (OpenAI/Anthropic) for a production system
9. What "production-ready" means for an LLM application — the full checklist beyond just making it work
10. How your background in recommendation systems and search directly applies to — and where it can mislead you in — AI Engineering with LLMs

---

**TONE & AUDIENCE:**
- Audience: Senior to Staff-level engineer interviewing for AI Engineer, Staff AI Engineer, or Principal AI Engineer roles.
- Background: Strong ML/data background — specifically recommendation systems, search, and personalization. Understands embeddings, retrieval, and offline evaluation deeply. Now applying those foundations in LLM-powered production systems.
- Every answer should sound like a principal AI engineer who has designed and shipped multiple production LLM systems.
- Answers to question 10 should be especially specific to the recommendation systems / search / personalization background — drawing real parallels and real divergences.

Use Markdown with clear `##` and `###` headers. Begin both artifacts now.

---
