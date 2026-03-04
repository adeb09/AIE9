# LangGraph: Multiple Input Parameters and `.invoke()`

This note explains whether a node function (e.g. `retrieve`) in a LangGraph can have more than one input parameter besides `state`, and how that affects the `.invoke()` method.

---

## Can the retrieve function have more than one parameter (other than state)?

**Yes.** LangGraph allows node functions to take a second parameter: **`config`** (the runtime `RunnableConfig`). So you can use:

- `(state)` — single parameter (what you have now)
- `(state, config)` — state plus config

So you *can* have “more than one input” in the sense of **state + config**. Custom “extra” inputs are not added as extra positional parameters; they’re passed either via **state** or via **config**.

---

## Two ways to pass “extra” inputs

### 1. Put them in **state**

Add fields to your `State` TypedDict (e.g. `k`, `top_n`, `retriever_mode`). The node reads them from `state["k"]`, etc. You supply them when calling the graph:

- **Invoke:** You pass a single input (the initial state), e.g.  
  `graph.invoke({"question": "...", "k": 10, "top_n": 5})`.  
  So **yes** — those extra parameters are supplied there, in the same dict as `"question"`.

### 2. Use **config** (second parameter)

Define the node with two parameters and read from `config`:

- **Signature:** e.g. `def retrieve(state, config):`
- **Inside the node:** get values from `config["configurable"]`, e.g.  
  `k = config["configurable"].get("k", 3)`.
- **Invoke:** You pass the initial state as the first argument and the config as the second (or as `config=...`), e.g.  
  `graph.invoke({"question": "..."}, config={"configurable": {"k": 10, "top_n": 5}})`.

So **yes** — if you use config for those parameters, they are supplied at `invoke` time in the **second** argument (the config), not in the state dict.

---

## Summary

| Approach   | Node signature           | Where you supply the extra params                                                |
|-----------|---------------------------|-----------------------------------------------------------------------------------|
| State     | `def retrieve(state)`     | First argument to `invoke`: `invoke({"question": "...", "k": 10})`               |
| Config    | `def retrieve(state, config)` | Second argument: `invoke(initial_state, config={"configurable": {"k": 10}})` |

So: you can have “more than one input” by using **state + config**. Extra parameters are supplied at `invoke` time either in the **state** dict or in the **config** dict, depending on which approach you use.
