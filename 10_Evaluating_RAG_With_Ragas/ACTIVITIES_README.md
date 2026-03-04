# Activities 1 & 2 - Complete Solutions

This directory contains complete solutions and documentation for Activities 1 and 2 from the Evaluating_Agents_Assignment.ipynb notebook.

---

## 📁 Files Created

### Documentation Files (Markdown)

1. **`activity_1_solution.md`** - Complete documentation for Activity 1
   - Objective and requirements
   - Full solution code
   - Expected output
   - Analysis of results
   - Additional test cases
   - Debugging tips
   - Production implementation guidance

2. **`activity_2_solution.md`** - Complete documentation for Activity 2
   - Objective and requirements
   - Full solution code
   - Expected output
   - Analysis of results
   - Comparison of on-topic vs off-topic queries
   - Boundary cases
   - Production implementation guidance

### Code Files (Python)

3. **`activity_1_code.py`** - Executable code for Activity 1
   - Tool Call Accuracy evaluation with platinum query
   - Step-by-step execution with detailed output
   - Automatic score interpretation

4. **`activity_2_code.py`** - Executable code for Activity 2
   - Topic Adherence evaluation with on-topic gold price query
   - Bonus comparison with off-topic eagle query
   - Automatic score interpretation and recommendations

---

## 🚀 How to Use

### Option 1: Copy Code into Notebook Cells

1. Open `Evaluating_Agents_Assignment.ipynb`
2. Navigate to **Activity 1** (cell 51)
3. Copy the code from `activity_1_code.py` into the cell
4. Run the cell
5. Navigate to **Activity 2** (cell 53)
6. Copy the code from `activity_2_code.py` into the cell
7. Run the cell

### Option 2: Run as Standalone Scripts

**Prerequisites:**
- Complete all previous cells in the notebook (cells 1-50)
- Ensure `react_graph` is defined
- Ensure all imports are available

**Run Activity 1:**
```bash
# In Jupyter notebook cell:
%run activity_1_code.py
```

**Run Activity 2:**
```bash
# In Jupyter notebook cell:
%run activity_2_code.py
```

### Option 3: Study the Documentation

If you want to understand the concepts deeply before implementing:

1. Read `activity_1_solution.md` to understand Tool Call Accuracy
2. Read `activity_2_solution.md` to understand Topic Adherence
3. Use the code examples as reference while writing your own solution

---

## 📊 What Each Activity Demonstrates

### Activity 1: Tool Call Accuracy

**Query:** "What is the price of platinum?"

**What It Tests:**
- Does the agent call the right tool? (`get_metal_price`)
- Does the agent use the correct parameter? (`metal_name="platinum"`)
- Is the tool called at the right time in the conversation?

**Expected Score:** 1.0 (perfect)

**Key Learnings:**
- Tool Call Accuracy is binary for simple queries (1.0 or 0.0)
- Parameter matching must be exact
- This metric validates the agent's technical competence

### Activity 2: Topic Adherence

**Query:** "What is the current price of gold?"

**What It Tests:**
- Does the agent stay focused on the "metals" topic?
- Does the response directly address metal pricing?
- Does the agent avoid off-topic digressions?

**Expected Score:** 1.0 (perfect)

**Key Learnings:**
- On-topic queries should score ≥ 0.95
- Topic Adherence is a security metric (prevents prompt injection)
- Precision mode measures: "Of all queries answered, what % are on-topic?"

**Bonus Comparison:**
The Activity 2 code also runs an off-topic query ("How fast can an eagle fly?") to demonstrate the contrast. This should score 0.0.

---

## 🎯 Expected Results

### Activity 1 Output

```
============================================================
ACTIVITY 1: Tool Call Accuracy Evaluation
============================================================

[Step 1] Running agent with query: 'What is the price of platinum?'

[Step 2] Agent execution result:
  Message 0: HumanMessage
    Content: What is the price of platinum?...
  Message 1: AIMessage
    Content: ...
    Tool Calls: [{'name': 'get_metal_price', 'args': {'metal_name': 'platinum'}, ...}]
  Message 2: ToolMessage
    Content: XXXX.XX...
  Message 3: AIMessage
    Content: The current price of platinum is $XXXX.XX per gram....

[Step 3] Converting to Ragas format...

[Step 4] Defining expected reference tool calls...
  Expected tool call: get_metal_price(metal_name='platinum')

[Step 5] Evaluating Tool Call Accuracy...

============================================================
RESULT: Tool Call Accuracy Score = 1.0
============================================================

✅ PERFECT SCORE!
   - Agent called the correct tool (get_metal_price)
   - Agent used the correct parameter (metal_name='platinum')
   - Tool was called at the appropriate time in the conversation
```

### Activity 2 Output

```
============================================================
ACTIVITY 2: Topic Adherence Evaluation
============================================================

[Step 1] Running agent with query: 'What is the current price of gold?'

[Step 2] Agent execution result:
  Message 0: HumanMessage
    Content: What is the current price of gold?...
  Message 1: AIMessage
    Content: (tool call)...
    Tool Calls: [{'name': 'get_metal_price', 'args': {'metal_name': 'gold'}, ...}]
  Message 2: ToolMessage
    Content: YYYY.YY...
  Message 3: AIMessage
    Content: The current price of gold is $YYYY.YY per gram....

[Step 3] Converting to Ragas format...

[Step 4] Creating sample with reference topics...
  Reference topics: ['metals']

[Step 5] Evaluating Topic Adherence...

============================================================
RESULT: Topic Adherence Score = 1.0
============================================================

✅ EXCELLENT SCORE!
   - Agent stayed completely on topic (metals)
   - Response directly addressed metal pricing
   - No off-topic digressions detected

   This is the expected behavior for on-topic queries.

============================================================

📊 BONUS: Comparison with Off-Topic Query
============================================================

Running off-topic query: 'How fast can an eagle fly?'

Off-Topic Query Score: 0.0

📈 Comparison:
  On-Topic Query  ('What is the price of gold?'): 1.0
  Off-Topic Query ('How fast can an eagle fly?'): 0.0
  Difference: 1.00

✅ Agent correctly distinguishes on-topic from off-topic queries!
```

---

## 🔍 Understanding the Metrics

### Tool Call Accuracy Formula

```
Tool Call Accuracy = (Sequence Alignment) × (Argument Accuracy)
```

- **Sequence Alignment**: Were the right tools called in the right order?
- **Argument Accuracy**: Were the right parameters passed to each tool?

For Activity 1:
- Sequence Alignment: 1.0 (called get_metal_price exactly once)
- Argument Accuracy: 1.0 (used metal_name="platinum" correctly)
- **Final Score: 1.0 × 1.0 = 1.0**

### Topic Adherence Formula (Precision Mode)

```
Precision = |Queries answered AND on-topic| /
            (|Queries answered AND on-topic| + |Queries answered AND off-topic|)
```

For Activity 2:
- Queries answered and on-topic: 1 (gold price query)
- Queries answered and off-topic: 0 (none)
- **Final Score: 1 / (1 + 0) = 1.0**

---

## 🛠️ Troubleshooting

### "NameError: name 'react_graph' is not defined"

**Solution:** Make sure you've run all previous cells in the notebook, especially:
- Cell 20: Builds and compiles the `react_graph`

### "NameError: name 'ChatOpenAI' is not defined"

**Solution:** Make sure you've run the import cells, especially:
- Cell 8: `from langchain_openai import ChatOpenAI`

### "ModuleNotFoundError: No module named 'ragas'"

**Solution:** Install required dependencies:
```bash
uv sync
```

### Score is not 1.0 for Activity 1

**Possible causes:**
1. Agent called wrong tool (check tool_calls in output)
2. Agent used wrong parameter (check args in output)
3. Metal name was different (check case sensitivity)

**Debugging:**
- Look at the "Agent execution result" section
- Compare actual tool calls with expected reference tool calls

### Score is not 1.0 for Activity 2

**Possible causes:**
1. Agent went off-topic in response
2. Agent refused to answer (recall issue, not precision issue)
3. Response included non-metal content

**Debugging:**
- Read the final AIMessage content
- Check if response is truly about metal pricing
- Try mode="f1" instead of mode="precision" to see recall score

---

## 📚 Additional Resources

### Related Documentation Files

- **`understanding_traces.md`** - Explains what a "trace" is in agent evaluation
- **`question_3_analysis.md`** - Implications of metrics for user trust and safety
- **`question_4_test_suite_design.md`** - Comprehensive test suite design

### External Resources

- [Ragas Documentation - Tool Call Accuracy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#tool-call-accuracy)
- [Ragas Documentation - Topic Adherence](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#topic-adherence)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)

---

## ✅ Checklist

Before submitting your assignment, verify:

- [ ] Activity 1 code runs without errors
- [ ] Activity 1 achieves Tool Call Accuracy score ≥ 0.95
- [ ] Activity 2 code runs without errors
- [ ] Activity 2 achieves Topic Adherence score ≥ 0.95
- [ ] You understand why each metric scored the way it did
- [ ] You've documented your results in the notebook

---

## 🎓 Key Takeaways

1. **Tool Call Accuracy validates technical competence**
   - The agent must call the right tools with the right parameters
   - Critical for production systems where wrong data = financial harm

2. **Topic Adherence is a security metric**
   - Prevents prompt injection and scope creep
   - Ensures agent stays within its authorized domain
   - Critical for regulated industries (finance, healthcare)

3. **Evaluation is not just about perfect scores**
   - Understanding WHY a score is high or low is more important
   - Edge cases reveal system limitations
   - Real learning comes from analyzing failures

4. **Production systems need continuous monitoring**
   - Don't just evaluate once - track metrics over time
   - User feedback helps identify new failure modes
   - Test suite should grow with real-world usage

---

## 💡 Next Steps

After completing these activities:

1. **Experiment with variations:**
   - Try different metals (palladium, rhodium)
   - Try different phrasings ("How much is...", "Price of...", "Cost of...")
   - Try edge cases (invalid metals, typos, mixed case)

2. **Test failure modes:**
   - What happens if you ask for aluminum? (not in API)
   - What happens if you ask about metal properties? (borderline topic)
   - What happens with prompt injection attempts?

3. **Design your own test cases:**
   - Use the test suite design from `question_4_test_suite_design.md`
   - Create a mini test suite with 5-10 test cases
   - Automate the evaluation

4. **Think about production:**
   - How would you monitor these metrics in production?
   - What thresholds would you set for alerting?
   - How would you handle metric degradation?

---

Good luck with your assignment! 🚀
