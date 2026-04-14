# AI Engineering Interview Prep
### Senior / Staff-Level Reference — 18-Week Bootcamp Synthesis

---

## ARTIFACT 1: LLM Stack Cheat Sheet

> A comprehensive reference across all 18 sessions. One row per major concept.

---

### LangChain / LCEL Core

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| Runnable interface | LangChain / LCEL | `Runnable` | Universal composable unit with `.invoke()`, `.stream()`, `.batch()` |
| Pipe composition | LangChain / LCEL | `\|` operator / `RunnableSequence` | Chains runnables left-to-right into a pipeline |
| Parallel execution | LangChain / LCEL | `RunnableParallel` | Fans out to multiple runnables simultaneously, merging results |
| Dynamic routing | LangChain / LCEL | `RunnableBranch` | Conditionally routes to different chains based on input |
| Passthrough | LangChain / LCEL | `RunnablePassthrough` | Passes input unchanged, useful for injecting context |
| Lambda step | LangChain / LCEL | `RunnableLambda` | Wraps any Python callable as a runnable |
| Retry / fallback | LangChain / LCEL | `.with_retry()` / `.with_fallbacks()` | Adds resilience and fallback chains to any runnable |
| Chat model | LangChain | `ChatOpenAI`, `ChatAnthropic` | Stateless LLM wrapper returning `AIMessage` |
| Prompt template | LangChain | `ChatPromptTemplate`, `PromptTemplate` | Typed, composable prompt construction |
| Output parser | LangChain | `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser` | Converts raw LLM output to structured Python types |
| Document | LangChain | `Document` | Core data class: `page_content` + `metadata` dict |
| Text splitter | LangChain | `RecursiveCharacterTextSplitter`, `SemanticChunker` | Splits docs into chunks with configurable overlap |
| Semantic chunker | LangChain / LangChain Experimental | `SemanticChunker` | Splits text on embedding-based sentence similarity boundaries |
| Embedding model | LangChain | `OpenAIEmbeddings`, `CohereEmbeddings` | Converts text to dense float vectors |
| Vector store | LangChain / Qdrant | `QdrantVectorStore`, `FAISS` | Stores and retrieves embeddings via ANN search |
| Retriever | LangChain | `VectorStoreRetriever`, `BM25Retriever` | Abstracts over any source returning `List[Document]` |
| Conversational memory | LangChain | `ConversationBufferMemory`, `ConversationSummaryMemory` | Stateful message history for chain invocations |
| Tool | LangChain | `@tool` decorator / `StructuredTool` | Typed function exposed to an LLM for calling |
| Tool calling | LangChain | `bind_tools()` | Attaches tool schemas to a chat model via function calling |
| Agent executor | LangChain | `AgentExecutor` | Legacy loop: plan → act → observe until stop |

---

### LangGraph Core

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| State graph | LangGraph | `StateGraph` | Directed graph where nodes transform a shared typed state dict |
| Typed state | LangGraph | `TypedDict` / `Annotated` fields | Defines the schema of the agent's shared memory |
| Reducer | LangGraph | `operator.add` / custom reducer | Merges concurrent node outputs into shared state |
| Node | LangGraph | `graph.add_node()` | Any callable that receives and returns state |
| Edge | LangGraph | `graph.add_edge()` | Unconditional transition between nodes |
| Conditional edge | LangGraph | `graph.add_conditional_edges()` | Routes to different nodes based on state or a routing function |
| Entry / finish point | LangGraph | `set_entry_point()` / `END` | Defines graph start and terminal conditions |
| Compile | LangGraph | `graph.compile()` | Freezes graph structure and returns an invocable `CompiledGraph` |
| Checkpointer | LangGraph | `MemorySaver`, `SqliteSaver`, `PostgresSaver` | Persists graph state for resumability and human-in-the-loop |
| Thread | LangGraph | `config={"configurable": {"thread_id": ...}}` | Isolates a conversation or run within shared checkpointer storage |
| Interrupt | LangGraph | `interrupt()` / `NodeInterrupt` | Pauses graph execution mid-node for human approval or input |
| `Send` | LangGraph | `Send(node, state)` | Dynamically spawns parallel sub-invocations of a node (map-reduce) |
| Subgraph | LangGraph | Compiled graph as node | Nests one graph inside another for modularity |
| Store | LangGraph | `InMemoryStore`, `AsyncPostgresStore` | Cross-thread persistent key-value memory (long-term agent memory) |
| ReAct agent | LangGraph | `create_react_agent()` | Prebuilt reason-then-act loop node with tool calling |
| Supervisor | LangGraph | Custom routing node | Orchestrates handoffs between specialized subagent nodes |
| Handoff | LangGraph | `Command(goto=..., update=...)` | Transfers control and state updates to another node or subgraph |
| Command | LangGraph | `Command` | Return type from a node that explicitly sets next destination + state |
| RemoteGraph | LangGraph / LangGraph Platform | `RemoteGraph` | Calls a deployed LangGraph app over HTTP as if it were local |
| Studio | LangGraph Platform | LangGraph Studio UI | Visual debugger for graph structure, state, and trace replay |
| `langgraph.json` | LangGraph Platform | Configuration file | Declares graphs, dependencies, and env for LangGraph Cloud deployment |
| Assistants API | LangGraph Platform | `/assistants` endpoint | Named, configurable instances of a deployed graph |

---

### Retrieval Strategies

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| Dense retrieval | Qdrant / LangChain | `QdrantVectorStore.similarity_search()` | ANN search over embedding vectors using cosine/dot-product |
| Sparse retrieval (BM25) | `rank_bm25` / LangChain | `BM25Retriever` | Keyword-based TF-IDF-style retrieval, no embeddings required |
| Hybrid retrieval | Qdrant / LangChain | `QdrantHybridRetriever` | Combines dense and sparse scores before re-ranking |
| Reciprocal Rank Fusion | LangChain / Custom | `RRF(lists, k=60)` | Merges ranked lists without score normalization using rank-based formula |
| Multi-query retrieval | LangChain | `MultiQueryRetriever` | LLM generates N query variants; union of all result sets |
| Parent document retrieval | LangChain | `ParentDocumentRetriever` | Retrieves small child chunks, returns their full parent documents |
| Contextual compression | LangChain | `ContextualCompressionRetriever` | Post-retrieval LLM or embedding filter to remove irrelevant passages |
| Reranking | Cohere / CrossEncoder | `CohereRerank`, `CrossEncoderReranker` | Scores each (query, doc) pair to re-order retrieval results |
| Hypothetical Document Embeddings | Custom / LangChain | HyDE pattern | Embeds LLM-generated hypothetical answers, not the query |
| Semantic chunking | LangChain Experimental | `SemanticChunker` | Chunk boundary at semantic drift rather than character count |
| Sliding window chunking | LangChain | `RecursiveCharacterTextSplitter(overlap=...)` | Fixed-size chunks with overlapping context |
| RAPTOR / hierarchical indexing | Custom | Tree-structure summarization | Recursive summary nodes indexed at multiple granularities |
| Self-query retrieval | LangChain | `SelfQueryRetriever` | LLM converts natural-language filters into metadata filter predicates |
| Ensemble retriever | LangChain | `EnsembleRetriever` | Combines multiple retrievers with weighted score fusion |

---

### Agentic RAG & Agent Patterns

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| ReAct loop | LangGraph / LangChain | `create_react_agent()` | Interleaves Reasoning and Acting steps until terminal condition |
| Tool node | LangGraph | `ToolNode` | Executes tool calls emitted by an LLM and appends results to state |
| Routing agent | LangGraph | Conditional edge + router LLM | Decides which specialist node handles the current query |
| Corrective RAG | LangGraph | Custom grading + re-retrieve node | Grades retrieved docs; if irrelevant, re-queries or web-searches |
| Self-RAG | LangGraph | Custom reflection nodes | Iteratively critiques and regenerates its own retrieved context |
| STORM / deep research | Custom / LangGraph | Multi-step outline + parallel retrieval | Simulates expert researcher: outline → parallel sub-queries → synthesis |
| Subagent pattern | LangGraph | `Send` + subgraph | Spawns task-specific agents and aggregates their outputs |
| Skill library | Custom | Callable registry | Named, versioned callable skills dispatched by a planning agent |
| Planning node | LangGraph | Custom LLM planner | Decomposes a goal into an ordered task list before execution |
| Human-in-the-loop | LangGraph | `interrupt()` + `graph.update_state()` | Pauses for approval, edit, or additional context from a human |

---

### Multi-Agent Patterns

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| Supervisor pattern | LangGraph | Supervisor node + conditional routing | Central orchestrator dispatches tasks to worker agents |
| Swarm / handoff pattern | LangGraph | `Command(goto=...)` | Agents pass control peer-to-peer without a central coordinator |
| Network topology | LangGraph | `StateGraph` with multiple subgraphs | Fully connected mesh where any agent can route to any other |
| Shared message state | LangGraph | `MessagesState` | All agents share a single append-only message list |
| Hierarchical agents | LangGraph | Nested `StateGraph` | Orchestrator manages sub-orchestrators, each with their own workers |
| A2A task lifecycle | A2A SDK | `Task`, `TaskState` | Standardized task object flowing between agents: submitted → working → completed |
| AgentCard | A2A SDK | `AgentCard` | Agent's self-description: capabilities, skills, input/output schema |
| AgentExecutor | A2A SDK | `AgentExecutor` | Server-side class that wraps a LangGraph/custom agent for A2A protocol |
| A2A transport | A2A SDK | HTTP + SSE | Protocol layer for inter-agent communication and streaming |
| A2A vs MCP | Architecture | — | A2A = agent-to-agent orchestration; MCP = agent-to-tool/resource access |

---

### Memory (CoALA Framework)

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| In-context (working) memory | LangGraph / LangChain | `MessagesState`, `state["messages"]` | Current conversation window; bounded by context length |
| External (episodic) memory | LangGraph Store | `store.put() / store.search()` | Long-term facts or conversation summaries outside the context window |
| Semantic memory | Vector store + Store | `QdrantVectorStore`, indexed `Store` | Stores generalized knowledge retrievable by embedding similarity |
| Procedural memory | System prompt / skills | Prompt templates, skill registry | Encodes "how to act" — stable, rarely updated |
| Memory write strategies | Custom | summarize, reflect, extract | LLM-powered transforms before writing to long-term store |
| Conversation summarization | LangChain | `ConversationSummaryMemory` | Compresses growing history into a rolling summary |
| Cross-thread memory | LangGraph Store | `InMemoryStore`, `AsyncPostgresStore` | Persists facts across separate conversation threads for a user |
| Memory namespace | LangGraph Store | `(user_id, "memories")` tuple | Scopes stored memories to a user, agent, or session |

---

### Evaluation (LangSmith + Ragas)

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| Dataset | LangSmith | `client.create_dataset()` | Named collection of input/output examples for evaluation |
| Example | LangSmith | `client.create_examples()` | Single input + optional reference output in a dataset |
| Evaluator | LangSmith | `evaluate()` + `LangChainStringEvaluator` | Scores a run against reference or LLM-as-judge criteria |
| Run tree | LangSmith | `RunTree` | Hierarchical trace object capturing inputs, outputs, and child runs |
| Experiment | LangSmith | `evaluate(target_fn, data=dataset)` | Runs a function over a dataset and records all traces + scores |
| LLM-as-judge | LangSmith / Custom | `CriteriaEvalChain` | Uses an LLM to score outputs on rubric criteria (correctness, helpfulness) |
| Tracing | LangSmith | `@traceable` / `LangChainTracer` | Auto-instruments LangChain/LangGraph calls to LangSmith |
| Annotation queue | LangSmith | Annotation UI | Human review queue for labeling or correcting traces |
| Faithfulness | Ragas | `faithfulness` | Fraction of answer claims entailed by retrieved context |
| Answer relevancy | Ragas | `answer_relevancy` | Embedding similarity between answer and the original question |
| Context precision | Ragas | `context_precision` | Proportion of retrieved chunks that are actually relevant |
| Context recall | Ragas | `context_recall` | Fraction of ground-truth answer covered by retrieved context |
| Answer correctness | Ragas | `answer_correctness` | Semantic + factual overlap between generated and reference answer |
| RAGAS evaluation pipeline | Ragas | `evaluate(dataset, metrics=[...])` | Batch-scores a RAG dataset across multiple Ragas metrics |
| Synthetic data generation | Ragas / LangSmith | `TestsetGenerator` | Auto-generates Q&A pairs from a corpus for eval dataset bootstrapping |
| Agent trajectory eval | LangSmith / Custom | Step-level trace scoring | Evaluates correctness of tool calls, routing decisions, and intermediate steps |

---

### MCP (Model Context Protocol)

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| MCP tool | MCP SDK | `@mcp.tool()` | Callable function exposed to an LLM client with typed schema |
| MCP resource | MCP SDK | `@mcp.resource()` | Read-only data source (file, DB row, API response) the LLM can access |
| MCP prompt | MCP SDK | `@mcp.prompt()` | Reusable, parameterized prompt template served over MCP |
| MCP server | MCP SDK | `FastMCP` / `mcp.run()` | Process that hosts tools, resources, and prompts |
| MCP client | MCP SDK / LangChain | `MultiServerMCPClient` | Connects to one or more MCP servers and exposes their tools to an LLM |
| stdio transport | MCP SDK | `transport="stdio"` | In-process communication for local/embedded MCP servers |
| SSE transport | MCP SDK | `transport="sse"` | HTTP Server-Sent Events for remote, long-lived MCP server connections |
| Tool discovery | MCP SDK | `list_tools()` | Dynamic enumeration of available tools from a running MCP server |
| MCP vs function calling | Architecture | — | MCP standardizes the server protocol; function calling is the LLM-side invocation |

---

### Guardrails & Production Safety

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| Guard | Guardrails AI | `Guard.from_pydantic()` | Wraps an LLM call with input/output validators |
| Validator | Guardrails AI | `@register_validator` | Custom or built-in check applied to LLM output fields |
| Rail spec | Guardrails AI | `Rail.from_string()` | RAIL XML or Pydantic schema describing valid output structure |
| On-fail action | Guardrails AI | `OnFail.REASK` / `OnFail.FIX` / `OnFail.EXCEPTION` | Specifies behavior when a validator fails |
| Input guardrail | Guardrails AI / Custom | Prompt injection check, PII detection | Validates or sanitizes user input before reaching the LLM |
| Output guardrail | Guardrails AI | Toxicity, hallucination, schema checks | Validates LLM response before returning to the user |
| Semantic cache | GPTCache / LangChain | `set_llm_cache(GPTCache(...))` | Returns cached response for semantically similar prior queries |
| Exact cache | LangChain | `InMemoryCache`, `RedisCache` | Hash-based lookup for identical prompt strings |
| Prompt injection defense | Custom / LangChain | Input sanitization + instruction hierarchy | Detects and neutralizes attempts to override system instructions |
| Rate limiting | API Gateway / Custom | Token bucket, per-user quotas | Prevents abuse and controls cost at the application layer |
| PII redaction | Custom / Presidio | Named-entity masking before LLM call | Strips personal data from prompts and responses |

---

### LLM Serving & Inference

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| TTFT | Serving metrics | Time-to-First-Token | Latency from request submission to first generated token |
| TPOT | Serving metrics | Time-per-Output-Token | Per-token decode latency after first token; drives streaming UX |
| Throughput | Serving metrics | Tokens/sec (server-side) | Total tokens generated per second across all concurrent requests |
| Prefill vs decode | LLM serving | — | Prefill processes the prompt (parallelizable); decode is autoregressive and memory-bound |
| Continuous batching | vLLM / TGI | Iteration-level scheduling | Adds new requests mid-batch to maximize GPU utilization |
| KV cache | vLLM / TGI | PagedAttention | Reuses attention key-value tensors across steps and requests |
| Speculative decoding | vLLM / Custom | Draft model + verification | Small model drafts tokens; large model verifies in parallel for lower TPOT |
| Quantization | llama.cpp / AWQ / GPTQ | INT4/INT8 weights | Reduces model memory footprint and increases inference throughput |
| Dedicated endpoint | Fireworks AI / Together AI | `fireworks.ai/inference` | Provisioned GPU capacity for a specific model with SLA guarantees |
| Serverless endpoint | Fireworks AI | Pay-per-token API | Shared infrastructure, no reserved capacity, variable latency |
| Ollama | Ollama | `ollama run <model>` | Local model server for development and privacy-sensitive workloads |
| OpenAI-compatible API | vLLM / Fireworks | `/v1/chat/completions` | Drop-in replacement endpoint for OpenAI SDK compatibility |

---

### Full-Stack Agent Apps & Deployment

| Concept | Primary Tool/Library | Key Class or Function | One-Line Description |
|---|---|---|---|
| LangGraph Cloud | LangGraph Platform | Hosted graph runtime | Managed deployment with persistence, assistants, and streaming |
| `langgraph.json` | LangGraph Platform | Deployment manifest | Declares graph entry points, Python deps, and env vars for Cloud |
| Assistants | LangGraph Platform | `POST /assistants` | Named, configurable instances of a deployed graph with isolated state |
| Thread (platform) | LangGraph Platform | `POST /threads` | Persistent conversation container tied to a checkpointer |
| Streaming modes | LangGraph Platform | `stream_mode="values"/"updates"/"events"` | Controls granularity of SSE output from deployed graphs |
| FastAPI integration | FastAPI | `app.include_router()` | Exposes agent as REST endpoint for frontend or service consumption |
| Docker deployment | Docker | `Dockerfile` + `docker-compose.yml` | Containerizes agent app with all dependencies for reproducible deployment |
| `uv` | Astral uv | `uv sync`, `uv run` | Ultra-fast Python package and environment manager |
| Environment isolation | uv / venv | `.venv`, `pyproject.toml` | Per-project dependency isolation to prevent version conflicts |
| Structured output | LangChain / OpenAI | `.with_structured_output(Schema)` | Forces LLM to emit JSON conforming to a Pydantic/JSON schema |

---

## ARTIFACT 2: 10 Hardest AI Engineering Interview Questions

> Cross-cutting, synthesis-level questions for Staff / Principal AI Engineer roles.

---

### Q1: RAG vs. Fine-Tuning vs. Prompting — When Do You Use Each?

**Q1:** Walk me through your decision framework for choosing between RAG, fine-tuning, and pure prompting — including the cost, latency, and quality trade-offs at each tier.

**A1:** The decision starts with diagnosing the actual failure mode. Pure prompting is the right first bet for the vast majority of tasks: a frontier model already knows how to reason, summarize, and follow instructions, so adding examples or structured instructions via few-shot prompting is zero-infrastructure and zero-training-cost. I push prompting until it's clearly not enough — usually because the model lacks factual knowledge (private or recent data), not because it lacks capability. That's the signal to reach for RAG. RAG externalizes the knowledge problem: rather than baking facts into weights, you retrieve them at inference time from a live corpus, which means the knowledge is updateable, attributable, and auditable without a retraining cycle. The cost of RAG is added latency (retrieval + re-ranking adds 50–300ms) and retrieval failure modes — irrelevant chunks degrade answers in ways that are silent and hard to detect without eval. Fine-tuning becomes the right answer when the problem is *behavioral* rather than *factual*: you need the model to consistently follow a specific format, adopt a particular tone or domain vocabulary, or perform a task type that isn't well-represented in the base model's training mix. Fine-tuning on proprietary instruction formats (e.g., a structured extraction schema over legal documents) can dramatically cut prompt length and improve reliability. However, fine-tuned models are expensive to maintain — every time the task evolves or the base model is updated, you re-train. Critically, RAG and fine-tuning are not mutually exclusive: a fine-tuned model with RAG is the production pattern for high-stakes knowledge-intensive tasks (think: medical coding, financial analysis). The hierarchy I use in practice is: prompt first → add RAG when knowledge is the bottleneck → fine-tune when format/style/task-type consistency is the bottleneck → fine-tune with RAG when both are bottlenecks.

**Staff-Level Follow-Up:** How would you design a continuous fine-tuning pipeline that uses LangSmith traces and human corrections to iteratively improve a fine-tuned RAG model in production without accumulating training data drift?

---

### Q2: Designing a Production RAG System for 10M Documents at Sub-100ms P99 Latency

**Q2:** Design a production RAG system for a 10-million-document enterprise corpus with a P99 latency requirement of under 100ms. Walk me through every architectural decision and where the latency budget goes.

**A2:** 100ms P99 over 10M documents is genuinely hard because retrieval, re-ranking, and generation each want a slice of that budget. My first move is to be precise about what "100ms" covers — if it includes LLM generation, we're almost certainly using streaming and measuring time-to-first-token, not end-to-end completion; if it's retrieval + re-rank only, it's more tractable. Assuming TTFT is the SLA: the retrieval layer must complete in under 30ms. With 10M documents, Qdrant with HNSW indexing (m=16, ef_construction=100) on a dedicated node returns top-k=50 results in ~5–15ms if vectors are in-memory; I'd partition by domain/tenant to keep index size manageable and parallelize across partitions. Chunking strategy matters enormously here: I'd use parent document retrieval with small 256-token child chunks for the index (maximizing recall precision) and return the 512-token parents to the LLM (maximizing context quality) — this is a quality win with no latency cost. For re-ranking, a full cross-encoder (Cohere Rerank or a local cross-encoder) adds 40–80ms over 50 candidates, which blows the budget; I'd use a lightweight embedding-based re-scorer or Cohere's compressed rerank API with top-10 candidates to stay under 20ms. Semantic caching (GPTCache or a custom Qdrant-backed cache) is essential at this scale — a 30–40% cache hit rate on common queries eliminates retrieval entirely for those requests. For the LLM call, I'd use a dedicated endpoint (Fireworks AI or Azure OpenAI PTU) with a fast model like GPT-4o-mini or Llama-3-8B fine-tuned for this domain, which delivers TTFT under 300ms with streaming. The eval strategy is equally important to the architecture: I'd run Ragas context precision/recall and faithfulness against a synthetic golden set generated from the corpus, and instrument every retrieval call in LangSmith to track p50/p95/p99 latency per component so I can pinpoint regression sources in CI.

**Staff-Level Follow-Up:** Your context precision metric is dropping over time as the corpus grows. How do you diagnose whether the degradation is in the embedding model, the chunking strategy, the HNSW index parameters, or the query distribution — without reindexing everything?

---

### Q3: Agent Reliability — What Makes Agents Fail and How Do You Design Against It?

**Q3:** What are the root causes of agent failure in production, and how do you architect an agentic system to be reliable?

**A3:** Agents fail in structurally different ways than deterministic software, and conflating them leads to the wrong mitigations. The five categories I track are: **compounding errors** (early wrong tool call poisons all downstream reasoning), **infinite loops** (the agent oscillates between two states with no escape condition), **tool contract violations** (tool returns unexpected schema or throws, and the agent hallucinates a recovery), **context window saturation** (long agentic traces degrade reasoning quality as the window fills), and **goal drift** (the agent reinterprets the original task mid-trajectory after tool outputs redirect its attention). The architectural responses are layered. For compounding errors, I use a reflection or grading node after each major retrieval or action step — a cheap LLM call that scores whether the step output is plausible before proceeding. For loops, I enforce a hard max-step budget in the graph state and add a loop-detection check that fingerprints the (last N state hashes) and breaks if a repeat is detected. For tool contract violations, every tool node wraps its call in a try/except that returns a structured error document rather than raising — the LLM handles errors as data. For context window saturation, I either summarize the trajectory above a token threshold or use LangGraph's Store to offload intermediate results out-of-band. For goal drift, I periodically inject the original task back into the prompt as a grounding anchor. Beyond these, the single most important production practice is **observability**: every LangGraph run is traced in LangSmith with node-level timing and state snapshots, so when an agent fails I have full replay capability. I also maintain a golden trajectory dataset in LangSmith and run step-level evaluators in CI — any deployment that changes agent routing failure rates by more than a threshold blocks promotion.

**Staff-Level Follow-Up:** How would you design an automatic self-healing mechanism where a failed agent trajectory is automatically diagnosed, a corrective prompt patch is proposed, and the patch is A/B tested against the original — without requiring engineer involvement for each incident?

---

### Q4: Supervisor vs. Handoff Pattern in Multi-Agent Systems

**Q4:** What are the trade-offs between the Supervisor pattern and the Handoff (swarm) pattern in multi-agent systems, and when does each break down?

**A4:** The Supervisor pattern centralizes routing intelligence: a single orchestrator node sees the full task state and decides which worker to call next. This gives you a single point of reasoning about global state, makes routing logic auditable and testable, and is easy to reason about in LangSmith traces because the orchestrator's decisions are explicit nodes. It breaks down when the number of workers exceeds the orchestrator's context capacity (you can't fit 20 worker descriptions in the routing prompt without degrading decision quality), when workers need to negotiate or collaborate without the orchestrator's involvement (requiring a round-trip through the supervisor per exchange, exploding latency), and when the orchestrator becomes a bottleneck under parallel load. The Handoff (swarm) pattern distributes routing: each agent knows which peers it can transfer to and makes local transfer decisions. This enables lower-latency peer-to-peer coordination and scales better when workers have clearly delineated domains. It breaks down when global task state is needed for a decision no individual agent can make locally, when circular handoffs emerge because no agent is responsible for the terminal condition, and when debugging becomes extremely hard — a trace through a swarm is a DAG of peer transfers that is much harder to follow than a tree rooted at a supervisor. In practice I use a hybrid: a lightweight supervisor for initial routing and escalation, with peer-to-peer handoffs permitted within a domain cluster. The critical design invariant for either pattern is that task state travels with the handoff — an agent receiving control must have enough context to act without querying a shared store, which means the `Command(goto=..., update=...)` payload in LangGraph must be carefully designed.

**Staff-Level Follow-Up:** In a swarm architecture, how would you implement distributed deadlock detection — where two agents are each waiting for the other to act — and what's the recovery mechanism?

---

### Q5: Building a Complete Eval Strategy for an Agentic System With No Ground Truth Labels

**Q5:** You're deploying an agentic system and have no ground truth labels. How do you build a complete evaluation strategy from scratch?

**A5:** No ground truth is the normal state in early production, and waiting for labels before evaluating is a trap. My strategy has four layers that don't require reference answers. **Layer 1: Synthetic data generation.** I use Ragas `TestsetGenerator` or a custom LLM pipeline to generate (question, expected answer, supporting context) triples from the actual corpus. This bootstraps reference-free eval with plausible ground truth and is far cheaper than manual annotation. The generated test set's coverage is itself a risk — I periodically sample production queries and cluster them to ensure my synthetic set covers the real query distribution. **Layer 2: Reference-free LLM-as-judge metrics.** Faithfulness (are claims in the answer supported by retrieved context?), answer relevancy (does the answer address the question?), and response coherence can all be scored without a reference answer. These are noisy but directionally reliable, especially for catching regressions. **Layer 3: Behavioral / trajectory evaluation.** For agentic tasks I define expected *behaviors* rather than expected outputs: the agent should call the retrieval tool before answering; the agent should not call external APIs for factual questions answerable from the corpus; the agent should terminate within N steps. These are deterministic checks on the trace that require no labels. **Layer 4: Human annotation pipeline.** I sample 1–5% of production traces into a LangSmith annotation queue, define a rubric (correctness, helpfulness, tool use appropriateness), and have domain experts label them. This creates a growing golden dataset over time. The feedback loop closes when I treat LangSmith annotation queue items as training signal for fine-tuning evaluators — so the human-label cost decreases as the LLM judge improves.

**Staff-Level Follow-Up:** Your LLM-as-judge evaluator has a known bias toward longer, more verbose answers. How do you debias it, and how do you validate that your debiasing actually works without access to a ground-truth preference dataset?

---

### Q6: How Memory Architecture Choices Affect Agent Behavior at Scale

**Q6:** How do memory architecture choices — in-context, external, episodic, semantic — affect agent behavior and cost at scale? Where do systems break?

**A6:** Memory architecture is the most underspecified part of most agent designs I've reviewed, and it's where production systems quietly fail. In-context (working) memory is effectively free at invocation time but has two hard limits: the context window ceiling and quadratic attention cost. At GPT-4o's 128k-token window, you can fit a lot — until you're running a long agentic session with tool call history, retrieved documents, and a multi-turn conversation simultaneously. The failure mode is subtle: reasoning quality degrades as the window fills with irrelevant prior turns, and the model can begin "forgetting" instructions from the beginning of the context (lost-in-the-middle problem). My response is aggressive summarization: at a token threshold I trigger a summarization node that condenses prior tool call history into a compact state record and truncates the raw history. External memory via LangGraph Store solves the persistence problem — facts, user preferences, and prior session summaries survive thread boundaries — but introduces consistency and staleness risks. If two concurrent threads write conflicting facts for the same user, the last-write-wins semantics of most key-value stores produce silent inconsistencies. For episodic memory I apply a write strategy (LLM extracts salient facts from a conversation before writing) rather than writing raw conversation turns — this keeps the store from becoming a noisy append-only log. Semantic memory (vector-indexed) is excellent for fuzzy recall ("what did we discuss about the user's dietary restrictions?") but adds retrieval latency and requires periodic re-embedding if the embedding model changes. The cost dimension is often ignored in memory design: cross-thread Store reads are cheap, but triggering a vector search on every agent turn adds both latency and embedding API cost at scale. I instrument memory access frequency and hit rates in LangSmith to decide which memory tier each fact type belongs in.

**Staff-Level Follow-Up:** You have a user-facing agent with 10 million users, each with growing episodic memory in a vector store. How do you handle memory compaction, staleness decay, and privacy-compliant deletion at that scale?

---

### Q7: MCP vs. A2A — When to Use Which, and How They Compose

**Q7:** Explain the architectural difference between MCP and A2A, when you'd use each, and how they compose in a real production system.

**A7:** MCP and A2A operate at different levels of abstraction and solve different problems, which is why conflating them leads to over-engineered systems. **MCP (Model Context Protocol)** is a standardization layer for *tool, resource, and prompt access* — it answers the question "how does an LLM agent call a capability?" in a way that is transport-agnostic, schema-typed, and discoverable. An MCP server is essentially a capability host: it exposes functions (tools), data sources (resources), and reusable instruction templates (prompts) over stdio or SSE. The client (your agent) discovers available tools at runtime via `list_tools()` and invokes them. MCP is the right choice whenever you want to decouple capability development from agent logic — a database team ships an MCP server for their schema, a calendar team ships one for scheduling, and your agent consumes both without knowing their internals. **A2A (Agent-to-Agent protocol)** is a standardization layer for *agent orchestration* — it answers "how does one agent delegate a task to another agent?" An A2A agent advertises its capabilities via an `AgentCard`, accepts `Task` objects, executes them, and streams results back. A2A is the right choice when the unit of delegation is a task with its own multi-step reasoning, not a single function call. In production these compose naturally: a top-level orchestrator uses A2A to delegate sub-tasks to specialist agents; each specialist agent uses MCP to access tools and resources. The boundary is: if the delegated unit is a function call (deterministic, fast, stateless), use MCP. If the delegated unit is a goal (may require multi-step reasoning, state, and tool use of its own), use A2A. The failure mode to avoid is wrapping a simple MCP tool call in A2A overhead — you pay the task lifecycle cost (create/submit/poll) for what should be a 50ms tool invocation.

**Staff-Level Follow-Up:** How would you implement capability-based authorization in a multi-tenant system where different A2A agents have different MCP tool access permissions, and the permission model needs to be enforced without trusting the agent to self-report its identity?

---

### Q8: Open-Source LLM Endpoint vs. Frontier API — How Do You Decide?

**Q8:** How do you decide whether to use an open-source LLM (self-hosted or via Fireworks/Together) vs. a frontier API like OpenAI or Anthropic for a production system?

**A8:** This decision has five dimensions and the right answer changes dramatically based on which ones dominate for a given workload. **Data sensitivity** is often the override: regulated data (HIPAA, GDPR, financial PII) frequently cannot leave a controlled perimeter, which forces open-source self-hosted or VPC-deployed endpoints regardless of capability delta. If data can leave, the other dimensions apply. **Capability gap**: for open-ended reasoning, long-horizon planning, and complex code generation, GPT-4o and Claude 3.5 Sonnet have a meaningful capability lead over open-source 70B models — that gap narrows rapidly but still exists for the hardest tasks. For structured extraction, classification, summarization, and RAG grounding, a fine-tuned Llama-3-70B or Mixtral on a dedicated Fireworks endpoint matches or beats frontier models at 5–10x lower cost per token. **Cost at scale**: at 100M+ tokens/day, frontier API costs ($15–30/M output tokens for GPT-4o) become a dominant operating cost; dedicated open-source endpoints on Fireworks or self-hosted vLLM drop this to $1–3/M. **Latency control**: shared frontier APIs have variable tail latency and no SLA at the token level; dedicated endpoints give you a stable TPOT baseline. Fireworks dedicated deployments are my go-to for latency-sensitive production workloads where a capable open-source model suffices. **Iteration velocity**: during prototyping and eval, frontier APIs are strictly better — no infra, no model management, immediate access to the latest models. I use them for development and shift to open-source endpoints for production after eval shows acceptable quality. The practical pattern: prototype on GPT-4o, evaluate quality on your task-specific benchmark, fine-tune Llama-3-70B on distilled data from GPT-4o outputs, deploy on Fireworks dedicated, run Ragas-based regression tests to confirm quality parity.

**Staff-Level Follow-Up:** You've fine-tuned Llama-3-70B on GPT-4o distillation data and it passes your quality benchmark. Three months later, OpenAI releases GPT-4o-mini which outperforms your fine-tune at 1/3 the cost. What's your decision framework for migrating back to the frontier API, and how do you manage the transition without user-facing regression?

---

### Q9: What Does "Production-Ready" Mean for an LLM Application?

**Q9:** Walk me through what "production-ready" means for an LLM application — the complete checklist beyond "it works in the demo."

**A9:** Production-readiness for LLM apps has a longer tail than for conventional software because the failure modes are probabilistic, the outputs are unstructured, and the system has an adversarial surface that traditional software doesn't. My checklist has six layers. **Reliability**: every LLM call has retry logic with exponential backoff, fallback models, and timeout budgets. Every tool call has a structured error contract. Every agent has a max-step budget and loop detection. The system degrades gracefully — a retrieval failure returns a "I couldn't find relevant context" response, not a 500. **Observability**: every production trace goes to LangSmith with node-level timing, token counts, tool call payloads, and a run ID that can be correlated with application logs. Latency, error rate, and cost are tracked per model and per workflow in a real-time dashboard. Anomaly detection alerts on p99 latency spikes and error rate increases. **Evaluation and regression testing**: there is a golden dataset in LangSmith, a CI job that runs evaluators on every deployment candidate, and a pass/fail gate on key metrics (faithfulness, task completion rate, latency p95). No deployment happens without passing eval. **Safety and guardrails**: input validation strips prompt injection vectors and PII before the LLM sees them. Output validation checks for policy violations, hallucinated citations, and schema conformance. Sensitive operations require human-in-the-loop approval. **Cost controls**: per-request token budgets, semantic caching for repeat queries, model tier routing (cheap model for simple intents, expensive model for complex ones). Cost per query is tracked and alerted on. **Operational hygiene**: secrets are in a vault, not `.env` files. Models are versioned and pinned. The `langgraph.json` specifies exact dependency versions. Rollback is a one-command operation. GDPR/CCPA deletion hooks are implemented for any user data stored in memory or vector indices. Most demos have layer 1 partially covered. Production requires all six.

**Staff-Level Follow-Up:** Your LLM application has just gone viral — traffic is 50x the expected load in 30 minutes. Walk me through the exact sequence of actions you take, in order, to keep the system running and minimize user-facing degradation.

---

### Q10: How Your Recommendation Systems & Search Background Applies — and Where It Misleads — in AI Engineering

**Q10:** You have deep experience in recommendation systems, search, and personalization. Where does that background directly accelerate your work in AI Engineering with LLMs, and where does it actively mislead you?

**A10:** This is the question I find most generative to think about because the parallels are real and the divergences are sharp — and confusing the two categories is how experienced ML engineers make avoidable mistakes in LLM systems. Let me take each direction seriously.

**Where the background directly accelerates:**

The most transferable asset is **retrieval intuition**. I've spent years thinking about precision-recall trade-offs, ANN index tuning, and query understanding — that translates directly to RAG system design. Understanding why BM25 outperforms dense retrieval on tail queries (vocabulary mismatch vs. semantic generalization), why hybrid retrieval with RRF works better than either alone, and why re-ranking is a separate problem from retrieval are all things I arrived at intuitively because I've solved analogous problems in product search. The chunking strategy question ("what's the right unit of retrieval?") is exactly the same as the passage retrieval problem I've solved in web search — parent document retrieval is just passage-level retrieval with document-level return, which is textbook.

**Offline evaluation methodology** is the second direct transfer. Building golden datasets, defining precision/recall/NDCG-style metrics, running A/B experiments with holdout sets, and thinking about counterfactual evaluation — these are core recsys skills that most LLM practitioners are still learning. I brought a rigorous eval-first discipline to LLM work from day one, which meant I didn't fall into the "vibes-based" eval trap that slows many teams.

**Embedding spaces** are deeply familiar. I've worked with user/item embeddings in collaborative filtering, understood cosine similarity as a retrieval signal, and debugged embedding drift when the item catalog changes. This makes Qdrant HNSW index tuning, embedding model selection, and the dimensionality-vs-recall trade-off immediately legible.

**Personalization patterns** transfer to agent memory design. The problem of "how do I represent what this user cares about, efficiently, in a way that retrieves the right context at inference time?" is structurally identical to user interest modeling in recsys. CoALA's semantic memory tier is essentially a user-interest vector store.

**Where the background actively misleads:**

The most dangerous transfer failure is **applying offline-metric thinking to LLM output quality**. In recsys, optimizing NDCG@10 on a logged dataset has a strong (if imperfect) correlation with online business metrics. In LLM systems, a model that scores higher on Ragas faithfulness can simultaneously produce worse user experience because faithfulness doesn't capture fluency, tone, or the specific failure modes users care about. The evaluation surface is higher-dimensional and the metrics are noisier. I've had to consciously override the instinct to declare success when a metric improves.

**The stationarity assumption** is another trap. Recsys models are trained on logged user behavior and assume the item catalog and user preferences evolve slowly. LLM applications often have task distributions that shift rapidly as users discover new capabilities — what users ask your agent today is not what they'll ask in three months. This means my intuition about "collect a big dataset and train" is wrong: the dataset is stale by the time the model ships, and continuous evaluation against a live query sample matters more than a large static benchmark.

**Latency tolerance is different**. In recsys, 200ms for a recommendation response is often fine; the user is scrolling and doesn't notice. In an LLM chat interface, 200ms to first token feels slow because the user is actively waiting for a response. My calibration for "acceptable latency" was wrong initially, and TTFT / streaming UX required a reframe.

**The failure mode taxonomy is inverted**. Recommender systems fail by being irrelevant — you surface a bad item. The cost is a missed click. LLM systems can fail by being confidently wrong — hallucinating a fact with authoritative tone. The cost can be a real-world harm. This asymmetry means that the risk appetite I calibrated in recsys (move fast, ship, iterate on metrics) is wrong for high-stakes LLM applications, where I need more conservative guardrails and a higher bar for production readiness before initial launch.

**Staff-Level Follow-Up:** You're designing a personalized RAG system where retrieved context should be adapted to individual user preferences and expertise level. How do you represent user state, how do you incorporate it into retrieval and generation, and how do you run a valid online experiment to measure the personalization lift without confounding effects from the base RAG quality?

---

*Generated for AI Engineering Interview Prep — Staff / Principal AI Engineer roles.*
*Bootcamp: AIE9 — 18-session LLM Engineering curriculum.*
*Stack: LangChain · LangGraph · LangSmith · Ragas · Qdrant · OpenAI · Anthropic · Fireworks AI · Ollama · MCP · A2A · Guardrails AI · uv · Docker*
