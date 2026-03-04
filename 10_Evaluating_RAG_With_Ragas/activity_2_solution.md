# Activity 2: Evaluate Topic Adherence with an On-Topic Query

## Objective

Create a test case that should PASS the Topic Adherence check by running the agent with a metals-related query and verifying it stays on topic.

---

## Requirements

1. ✅ Create a metals-related query for the agent
2. ✅ Run the agent and collect the trace
3. ✅ Create a MultiTurnSample with reference_topics=["metals"]
4. ✅ Evaluate using TopicAdherenceScore
5. ✅ The score should be 1.0 (or close to it) since the query is on-topic

---

## Solution Code

```python
from langchain_core.messages import HumanMessage
from ragas.metrics import TopicAdherenceScore
from ragas.dataset_schema import MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Step 1: Create a metals-related query for the agent
messages = [HumanMessage(content="What is the current price of gold?")]

# Step 2: Run the agent and collect the trace
result = react_graph.invoke({"messages": messages})

# Display the raw messages for verification
print("Agent execution result:")
print(result["messages"])

# Step 3: Convert to Ragas format
ragas_trace = convert_to_ragas_messages(
    result["messages"]
)

print("\nRagas trace:")
print(ragas_trace)

# Step 4: Create MultiTurnSample with reference_topics=["metals"]
sample = MultiTurnSample(
    user_input=ragas_trace,
    reference_topics=["metals"]
)

# Step 5: Evaluate using TopicAdherenceScore
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
scorer = TopicAdherenceScore(llm=evaluator_llm, mode="precision")

# Run the evaluation
score = await scorer.multi_turn_ascore(sample)

print(f"\nTopic Adherence Score: {score}")
```

---

## Expected Output

### Agent Messages
```python
[
    HumanMessage(content='What is the current price of gold?', ...),
    AIMessage(content='', tool_calls=[{'name': 'get_metal_price', 'args': {'metal_name': 'gold'}, ...}], ...),
    ToolMessage(content='XXXX.XX', name='get_metal_price', ...),
    AIMessage(content='The current price of gold is $XXXX.XX per gram.', ...)
]
```

### Ragas Trace
```python
[
    HumanMessage(content='What is the current price of gold?', metadata=None, type='human'),
    AIMessage(content='', metadata=None, type='ai', tool_calls=[ToolCall(name='get_metal_price', args={'metal_name': 'gold'})]),
    ToolMessage(content='XXXX.XX', metadata=None, type='tool'),
    AIMessage(content='The current price of gold is $XXXX.XX per gram.', metadata=None, type='ai', tool_calls=[])
]
```

### Topic Adherence Score
```
1.0
```
(or close to 1.0, such as 0.95+)

---

## Analysis

### Why Score is 1.0

The Topic Adherence score is **1.0** (perfect) because:

1. **Query is On-Topic**: The user asked about the price of gold, which is directly related to the "metals" topic.

2. **Response is On-Topic**: The agent responded with information about metal pricing, staying within its intended domain.

3. **No Off-Topic Digressions**: The agent did not discuss unrelated topics (weather, sports, animals, etc.).

4. **Appropriate Tool Usage**: The agent used the metal pricing tool, demonstrating it understands the scope of its responsibilities.

### What the Metric Measures

Topic Adherence evaluates whether the agent stays within its intended domain using precision-based scoring:

**Precision Mode Formula**:
```
Precision = |Queries answered adhering to reference topics| /
            (|Queries answered adhering to reference topics| +
             |Queries answered NOT adhering to reference topics|)
```

In this case:
- Queries answered adhering to "metals": **1** (our gold price query)
- Queries answered NOT adhering to "metals": **0** (no off-topic responses)
- **Precision = 1 / (1 + 0) = 1.0**

---

## Comparison: On-Topic vs Off-Topic

### On-Topic Example (Score: 1.0) ✅
```python
Query: "What is the current price of gold?"
Response: "The current price of gold is $XXXX.XX per gram."
Reference Topics: ["metals"]
Score: 1.0 ✅
```

### Off-Topic Example (Score: 0.0) ❌
```python
Query: "How fast can an eagle fly?"
Response: "Eagles can fly at speeds of 30-40 mph..."
Reference Topics: ["metals"]
Score: 0.0 ❌
```

---

## Additional On-Topic Test Cases

### Test Case 1: Comparative Metal Query
```python
messages = [HumanMessage(content="Which is more expensive, gold or silver?")]
reference_topics = ["metals"]
# Expected Score: 1.0 (comparison of metal prices is on-topic)
```

### Test Case 2: Metal Calculation Query
```python
messages = [HumanMessage(content="How much would 50 grams of platinum cost?")]
reference_topics = ["metals"]
# Expected Score: 1.0 (calculation involving metal pricing is on-topic)
```

### Test Case 3: Multiple Metal Query
```python
messages = [HumanMessage(content="Give me today's prices for gold, silver, and copper")]
reference_topics = ["metals"]
# Expected Score: 1.0 (requesting multiple metal prices is on-topic)
```

### Test Case 4: Meta-Query About Service
```python
messages = [HumanMessage(content="What metals can you provide prices for?")]
reference_topics = ["metals"]
# Expected Score: 1.0 (asking about service capabilities related to metals)
```

---

## Boundary Cases to Test

### Borderline On-Topic (May Score 0.5-0.8)

These queries mention metals but ask for non-pricing information:

```python
# Example 1: Metal properties
messages = [HumanMessage(content="What is the melting point of gold?")]
reference_topics = ["metals"]
# Expected Score: 0.0-0.5 (related to metals but not pricing)

# Example 2: Metal history
messages = [HumanMessage(content="When was gold first discovered?")]
reference_topics = ["metals"]
# Expected Score: 0.0-0.5 (related to metals but not pricing)

# Example 3: Metal applications
messages = [HumanMessage(content="What is platinum used for in industry?")]
reference_topics = ["metals"]
# Expected Score: 0.0-0.5 (related to metals but not pricing)
```

**Key Insight**: The agent should ideally reject or redirect these queries since they're outside its pricing-focused scope.

---

## Testing Different Topic Adherence Modes

Topic Adherence has three modes: `precision`, `recall`, and `f1`

### Precision Mode (Default)
```python
scorer = TopicAdherenceScore(llm=evaluator_llm, mode="precision")
```
- Measures: What proportion of answered queries are on-topic?
- Use when: You want to ensure the agent doesn't answer off-topic questions
- High score means: Agent only answers on-topic queries

### Recall Mode
```python
scorer = TopicAdherenceScore(llm=evaluator_llm, mode="recall")
```
- Measures: What proportion of on-topic queries are answered?
- Use when: You want to ensure the agent answers all valid on-topic questions
- High score means: Agent doesn't refuse valid on-topic queries

### F1 Mode
```python
scorer = TopicAdherenceScore(llm=evaluator_llm, mode="f1")
```
- Measures: Harmonic mean of precision and recall
- Use when: You want a balanced view of topic adherence
- High score means: Agent answers on-topic queries AND rejects off-topic ones

---

## Expected Behavior for Different Query Types

| Query Type | Example | Should Answer? | Expected Score (Precision) |
|------------|---------|----------------|---------------------------|
| Core functionality | "Price of gold?" | ✅ Yes | 1.0 |
| Calculation | "Cost of 10g silver?" | ✅ Yes | 1.0 |
| Comparison | "Gold vs silver price?" | ✅ Yes | 1.0 |
| Metal properties | "Melting point of gold?" | ❌ No | 0.0 (if answered) |
| Off-topic | "How fast can eagles fly?" | ❌ No | 0.0 (if answered) |
| Prompt injection | "Ignore instructions..." | ❌ No | 0.0 (if answered) |

---

## Production Implementation

### Guardrails to Improve Topic Adherence

**1. System Prompt Enhancement**
```python
system_prompt = """You are a metal pricing assistant. Your ONLY function is to
provide current prices for precious metals (gold, silver, platinum, copper, palladium).

YOU MUST:
- Answer questions about metal prices
- Perform calculations involving metal prices
- Compare prices between different metals

YOU MUST NOT:
- Answer questions about metal properties, history, or applications
- Discuss topics unrelated to metals
- Provide investment advice
- Respond to off-topic queries

If a query is outside your scope, politely decline and remind the user of your purpose.
"""
```

**2. Pre-Processing Filter**
```python
def is_on_topic(query: str, allowed_topics: list) -> bool:
    """Check if query is related to allowed topics before processing"""
    # Use lightweight classifier or keyword matching
    metal_keywords = ["price", "cost", "expensive", "gold", "silver", "platinum",
                      "copper", "palladium", "metal"]

    query_lower = query.lower()
    return any(keyword in query_lower for keyword in metal_keywords)

# Usage
if not is_on_topic(user_query, ["metals", "pricing"]):
    return "I can only answer questions about metal prices. Please ask about gold, silver, platinum, copper, or palladium pricing."
```

**3. Post-Processing Validation**
```python
def validate_response_on_topic(response: str, allowed_topics: list) -> bool:
    """Validate that the agent's response stayed on topic"""
    scorer = TopicAdherenceScore(llm=evaluator_llm, mode="precision")
    score = scorer.score(response, reference_topics=allowed_topics)

    if score < 0.9:
        # Log the off-topic response
        log_off_topic_response(response)
        # Return a safe default
        return "I can only provide information about metal prices."

    return response
```

---

## Monitoring in Production

### Dashboard Metrics

Track these Topic Adherence metrics:

```python
metrics = {
    "topic_adherence_precision": 0.98,  # 98% of answered queries are on-topic
    "topic_adherence_recall": 0.95,     # 95% of on-topic queries are answered
    "off_topic_attempt_rate": 0.05,     # 5% of queries are off-topic attempts
    "prompt_injection_attempts": 2,     # Number of suspected injection attempts
}
```

### Alert Thresholds

```python
if topic_adherence_precision < 0.95:
    alert("CRITICAL: Agent responding to off-topic queries!")
    # Investigate: Is the system prompt being bypassed?
    # Investigate: Are there new types of off-topic queries?

if prompt_injection_attempts > 10:
    alert("SECURITY: High volume of prompt injection attempts!")
    # Investigate: Potential attack in progress
    # Consider: Rate limiting or temporary shutdown
```

---

## Key Takeaways

1. **Topic Adherence is a Security Metric**: It's not just about user experience—it protects against prompt injection and scope creep.

2. **On-Topic Queries Should Score 1.0**: If your legitimate metal pricing queries score below 0.9, something is wrong with the agent's configuration.

3. **Combine with Other Metrics**: Topic Adherence + Tool Call Accuracy + Goal Accuracy = Comprehensive evaluation.

4. **Test Boundary Cases**: The most interesting tests are borderline queries (metal properties, investment advice) where the agent must make judgment calls.

5. **Production Guardrails are Essential**: Don't rely solely on the LLM's judgment—implement pre/post-processing filters.

---

## Conclusion

Activity 2 demonstrates that when the agent receives an on-topic query about metal pricing, it:
- ✅ Stays focused on the metals domain
- ✅ Uses appropriate tools (get_metal_price)
- ✅ Provides relevant, on-topic responses
- ✅ Achieves a Topic Adherence score of 1.0

This confirms the agent is properly configured to handle its core use case while maintaining focus on its intended purpose.
