"""
Activity 2: Evaluate Topic Adherence with an On-Topic Query

This code demonstrates how to test Topic Adherence with a metals-related query.
"""

from langchain_core.messages import HumanMessage
from ragas.metrics import TopicAdherenceScore
from ragas.dataset_schema import MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Step 1: Create a metals-related query for the agent
print("=" * 60)
print("ACTIVITY 2: Topic Adherence Evaluation")
print("=" * 60)

messages = [HumanMessage(content="What is the current price of gold?")]

# Step 2: Run the agent and collect the trace
print("\n[Step 1] Running agent with query: 'What is the current price of gold?'")
result = react_graph.invoke({"messages": messages})

# Display the raw messages for verification
print("\n[Step 2] Agent execution result:")
for i, msg in enumerate(result["messages"]):
    print(f"  Message {i}: {type(msg).__name__}")
    if hasattr(msg, 'content'):
        content_preview = msg.content[:100] if msg.content else "(empty)"
        print(f"    Content: {content_preview}...")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"    Tool Calls: {msg.tool_calls}")

# Step 3: Convert to Ragas format
print("\n[Step 3] Converting to Ragas format...")
ragas_trace = convert_to_ragas_messages(result["messages"])

print("\nRagas trace:")
for i, msg in enumerate(ragas_trace):
    print(f"  {i}. {msg.type}: {msg.content[:80] if msg.content else '(tool call)'}...")

# Step 4: Create MultiTurnSample with reference_topics=["metals"]
print("\n[Step 4] Creating sample with reference topics...")
sample = MultiTurnSample(
    user_input=ragas_trace,
    reference_topics=["metals"]
)

print(f"  Reference topics: {sample.reference_topics}")

# Step 5: Evaluate using TopicAdherenceScore
print("\n[Step 5] Evaluating Topic Adherence...")
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
scorer = TopicAdherenceScore(llm=evaluator_llm, mode="precision")

# Run the evaluation
score = await scorer.multi_turn_ascore(sample)

print("\n" + "=" * 60)
print(f"RESULT: Topic Adherence Score = {score}")
print("=" * 60)

# Interpretation
if score >= 0.95:
    print("\n✅ EXCELLENT SCORE!")
    print("   - Agent stayed completely on topic (metals)")
    print("   - Response directly addressed metal pricing")
    print("   - No off-topic digressions detected")
    print("\n   This is the expected behavior for on-topic queries.")
elif score >= 0.8:
    print("\n⚠️  GOOD SCORE (but not perfect)")
    print("   - Agent mostly stayed on topic")
    print("   - May have included some tangential information")
    print("   - Review response for any off-topic content")
elif score >= 0.5:
    print("\n⚠️  MODERATE SCORE")
    print("   - Agent partially deviated from topic")
    print("   - Significant off-topic content detected")
    print("   - Review the trace to identify problems")
else:
    print("\n❌ LOW SCORE")
    print("   - Agent went significantly off-topic")
    print("   - This should NOT happen for a metal pricing query")
    print("   - URGENT: Review system prompt and guardrails")

print("\n" + "=" * 60)

# Comparison with off-topic query
print("\n📊 BONUS: Comparison with Off-Topic Query")
print("=" * 60)

# Run an off-topic query for comparison
print("\nRunning off-topic query: 'How fast can an eagle fly?'")
off_topic_messages = [HumanMessage(content="How fast can an eagle fly?")]
off_topic_result = react_graph.invoke({"messages": off_topic_messages})

off_topic_trace = convert_to_ragas_messages(off_topic_result["messages"])
off_topic_sample = MultiTurnSample(
    user_input=off_topic_trace,
    reference_topics=["metals"]
)

off_topic_score = await scorer.multi_turn_ascore(off_topic_sample)

print(f"\nOff-Topic Query Score: {off_topic_score}")

print("\n📈 Comparison:")
print(f"  On-Topic Query  ('What is the price of gold?'): {score}")
print(f"  Off-Topic Query ('How fast can an eagle fly?'): {off_topic_score}")
print(f"  Difference: {abs(score - off_topic_score):.2f}")

if score > 0.8 and off_topic_score < 0.2:
    print("\n✅ Agent correctly distinguishes on-topic from off-topic queries!")
elif score > 0.8 and off_topic_score >= 0.2:
    print("\n⚠️  WARNING: Agent answered off-topic query!")
    print("   This indicates weak topic adherence guardrails.")
    print("   Consider strengthening system prompt or adding filters.")
else:
    print("\n❌ PROBLEM: Unexpected scoring pattern")
    print("   Review agent configuration and evaluation setup.")

print("\n" + "=" * 60)
