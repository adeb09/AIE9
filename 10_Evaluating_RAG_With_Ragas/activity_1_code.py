"""
Activity 1: Evaluate Tool Call Accuracy with a New Query

This code demonstrates how to test Tool Call Accuracy with a platinum price query.
"""

from langchain_core.messages import HumanMessage
from ragas.metrics import ToolCallAccuracy
from ragas.dataset_schema import MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages
from langchain_openai import ChatOpenAI
import ragas.messages as r

# Step 1: Create a new query for the agent
print("=" * 60)
print("ACTIVITY 1: Tool Call Accuracy Evaluation")
print("=" * 60)

messages = [HumanMessage(content="What is the price of platinum?")]

# Step 2: Run the agent and collect the trace
print("\n[Step 1] Running agent with query: 'What is the price of platinum?'")
result = react_graph.invoke({"messages": messages})

# Display the raw messages for verification
print("\n[Step 2] Agent execution result:")
for i, msg in enumerate(result["messages"]):
    print(f"  Message {i}: {type(msg).__name__}")
    if hasattr(msg, 'content'):
        print(f"    Content: {msg.content[:100]}...")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"    Tool Calls: {msg.tool_calls}")

# Step 3: Convert to Ragas format
print("\n[Step 3] Converting to Ragas format...")
ragas_trace = convert_to_ragas_messages(messages=result["messages"])

print("\nRagas trace:")
for i, msg in enumerate(ragas_trace):
    print(f"  {i}. {msg}")

# Step 4: Define the expected reference tool calls
print("\n[Step 4] Defining expected reference tool calls...")
sample = MultiTurnSample(
    user_input=ragas_trace,
    reference_tool_calls=[
        r.ToolCall(name="get_metal_price", args={"metal_name": "platinum"})
    ],
)

print(f"  Expected tool call: get_metal_price(metal_name='platinum')")

# Step 5: Evaluate using ToolCallAccuracy
print("\n[Step 5] Evaluating Tool Call Accuracy...")
tool_accuracy_scorer = ToolCallAccuracy()
tool_accuracy_scorer.llm = ChatOpenAI(model="gpt-4o-mini")

# Run the evaluation
score = await tool_accuracy_scorer.multi_turn_ascore(sample)

print("\n" + "=" * 60)
print(f"RESULT: Tool Call Accuracy Score = {score}")
print("=" * 60)

# Interpretation
if score == 1.0:
    print("\n✅ PERFECT SCORE!")
    print("   - Agent called the correct tool (get_metal_price)")
    print("   - Agent used the correct parameter (metal_name='platinum')")
    print("   - Tool was called at the appropriate time in the conversation")
elif score >= 0.8:
    print("\n⚠️  GOOD SCORE (but not perfect)")
    print("   - Check if parameters were slightly different")
    print("   - Check if tool was called in wrong order")
elif score >= 0.5:
    print("\n⚠️  MODERATE SCORE")
    print("   - Significant issues with tool calls")
    print("   - Review the trace to identify problems")
else:
    print("\n❌ LOW SCORE")
    print("   - Agent likely called wrong tool or used wrong parameters")
    print("   - Review the trace carefully")

print("\n" + "=" * 60)
