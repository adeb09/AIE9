# Activity 1: Evaluate Tool Call Accuracy with a New Query

## Objective

Create a new test case for Tool Call Accuracy using a different metal query and evaluate the agent's performance.

---

## Requirements

1. ✅ Create a new query for the agent
2. ✅ Run the agent and collect the trace
3. ✅ Define the expected reference tool calls
4. ✅ Evaluate using ToolCallAccuracy
5. ✅ Document your results

---

## Solution Code

```python
from langchain_core.messages import HumanMessage
from ragas.metrics import ToolCallAccuracy
from ragas.dataset_schema import MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages
from langchain_openai import ChatOpenAI
import ragas.messages as r

# Step 1: Create a new query for the agent
messages = [HumanMessage(content="What is the price of platinum?")]

# Step 2: Run the agent and collect the trace
result = react_graph.invoke({"messages": messages})

# Display the raw messages for verification
print("Agent execution result:")
print(result["messages"])

# Step 3: Convert to Ragas format
ragas_trace = convert_to_ragas_messages(
    messages=result["messages"]
)

print("\nRagas trace:")
print(ragas_trace)

# Step 4: Define the expected reference tool calls
sample = MultiTurnSample(
    user_input=ragas_trace,
    reference_tool_calls=[
        r.ToolCall(name="get_metal_price", args={"metal_name": "platinum"})
    ],
)

# Step 5: Evaluate using ToolCallAccuracy
tool_accuracy_scorer = ToolCallAccuracy()
tool_accuracy_scorer.llm = ChatOpenAI(model="gpt-4o-mini")

# Run the evaluation
score = await tool_accuracy_scorer.multi_turn_ascore(sample)

print(f"\nTool Call Accuracy Score: {score}")
```

---

## Expected Output

### Agent Messages
```python
[
    HumanMessage(content='What is the price of platinum?', ...),
    AIMessage(content='', tool_calls=[{'name': 'get_metal_price', 'args': {'metal_name': 'platinum'}, ...}], ...),
    ToolMessage(content='XXXX.XX', name='get_metal_price', ...),
    AIMessage(content='The current price of platinum is $XXXX.XX per gram.', ...)
]
```

### Ragas Trace
```python
[
    HumanMessage(content='What is the price of platinum?', metadata=None, type='human'),
    AIMessage(content='', metadata=None, type='ai', tool_calls=[ToolCall(name='get_metal_price', args={'metal_name': 'platinum'})]),
    ToolMessage(content='XXXX.XX', metadata=None, type='tool'),
    AIMessage(content='The current price of platinum is $XXXX.XX per gram.', metadata=None, type='ai', tool_calls=[])
]
```

### Tool Call Accuracy Score
```
1.0
```

---

## Analysis

### Why Score is 1.0

The Tool Call Accuracy score is **1.0** (perfect) because:

1. **Correct Tool Selection**: The agent correctly identified that it needs to call the `get_metal_price` tool to answer the user's question about platinum pricing.

2. **Correct Parameters**: The agent passed the correct argument `{"metal_name": "platinum"}` to the tool, exactly matching the expected reference tool call.

3. **Proper Sequence**: The tool was called at the appropriate point in the conversation flow (after receiving the user's question, before providing the final answer).

### What the Metric Measures

Tool Call Accuracy evaluates two key aspects:

- **Sequence Alignment**: Were the correct tools called in the correct order?
  - ✅ Yes: Called `get_metal_price` exactly once, at the right time

- **Argument Accuracy**: Were the correct parameters passed to each tool?
  - ✅ Yes: Passed `metal_name="platinum"` which matches the user's query

**Final Metric**: (Argument Accuracy) × (Sequence Alignment) = 1.0 × 1.0 = **1.0**

---

## Additional Test Cases

### Extending the Test

You can test Tool Call Accuracy with various scenarios:

#### Test Case 1: Different Metal
```python
messages = [HumanMessage(content="What is the price of palladium?")]
reference_tool_calls = [
    r.ToolCall(name="get_metal_price", args={"metal_name": "palladium"})
]
```

#### Test Case 2: Multiple Metals
```python
messages = [HumanMessage(content="What are the prices of platinum and palladium?")]
reference_tool_calls = [
    r.ToolCall(name="get_metal_price", args={"metal_name": "platinum"}),
    r.ToolCall(name="get_metal_price", args={"metal_name": "palladium"})
]
```

#### Test Case 3: Case Insensitivity
```python
messages = [HumanMessage(content="What is the price of PLATINUM?")]
reference_tool_calls = [
    r.ToolCall(name="get_metal_price", args={"metal_name": "platinum"})
]
# Note: The tool should normalize to lowercase
```

---

## Potential Issues and Debugging

### If Score is Less Than 1.0

**Problem**: Wrong tool called
```python
# Expected: get_metal_price("platinum")
# Actual: get_metal_price("gold")
# Score: 0.0
```
**Solution**: Check prompt clarity, ensure metal name is unambiguous

**Problem**: Correct tool but wrong parameter format
```python
# Expected: {"metal_name": "platinum"}
# Actual: {"metal": "platinum"}
# Score: 0.0
```
**Solution**: Check tool definition and parameter names

**Problem**: Tool called multiple times unnecessarily
```python
# Expected: 1 call to get_metal_price
# Actual: 2 calls to get_metal_price (redundant)
# Score: May be reduced
```
**Solution**: Improve agent reasoning to avoid redundant calls

---

## Key Takeaways

1. **Tool Call Accuracy is binary for simple queries**: With a straightforward query like "What is the price of platinum?", the agent either gets it right (1.0) or wrong (0.0).

2. **Parameter matching is strict**: The argument dictionary must exactly match the reference, including key names and values.

3. **Order matters for multiple tools**: If your query requires multiple tool calls, they must be in the correct sequence to achieve a perfect score.

4. **This metric tests agent competence**: A high Tool Call Accuracy score indicates the agent correctly understands when and how to use its available tools.

---

## Integration with Production

### When to Alert

Set up monitoring to alert if Tool Call Accuracy drops below threshold:

```python
if tool_call_accuracy < 0.95:
    alert("Critical: Tool Call Accuracy below threshold!")
    # Investigate: Are users asking queries differently?
    # Investigate: Has the tool definition changed?
    # Investigate: Is the LLM having issues?
```

### Continuous Monitoring

Track this metric across all user queries to ensure consistent performance:

```python
daily_average_tool_call_accuracy = calculate_daily_average()
if daily_average_tool_call_accuracy < 0.95:
    review_failed_cases()
    update_test_suite()
```
