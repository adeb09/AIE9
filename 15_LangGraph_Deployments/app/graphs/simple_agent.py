"""A minimal tool-using agent graph.

The graph:
- Calls a chat model bound to the tool belt.
- If the last message requested tool calls, routes to a ToolNode.
- Otherwise, terminates.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.models import get_chat_model
from app.state import MessagesState
from app.tools import get_tool_belt


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def build_graph():
    """Build an agent graph that interleaves model and tool execution."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "action", END: END})
    graph.add_edge("action", "agent")
    return graph


# Export compiled graph for LangGraph
graph = build_graph().compile()
