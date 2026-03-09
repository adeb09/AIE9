"""An agent graph with a post-response tool-call summarizer node.

After the agent finishes its agentic loop (no more tool calls), a summarizer
node collects every ToolMessage from the conversation history, builds a
structured summary of all tool calls made, and prepends it to the final
agent response before returning it to the user.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.state import MessagesState
from app.models import get_chat_model
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


def route_to_action_or_summarizer(state: MessagesState) -> str:
    """Decide whether to execute tools or run the summarizer."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "summarizer"


def summarizer_node(state: MessagesState) -> dict:
    """Prepend a structured summary of all tool calls to the final agent response.

    Scans the full message history for ToolMessage instances, extracts the tool
    name and a truncated snippet of each result, then prepends the summary block
    to the last AIMessage content so the user can see exactly what was retrieved.
    """
    messages = state["messages"]

    tool_entries: list[tuple[str, str]] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "unknown_tool")
            content = str(getattr(msg, "content", ""))
            snippet = content[:300] + ("..." if len(content) > 300 else "")
            tool_entries.append((tool_name, snippet))

    if tool_entries:
        lines = ["=== Tool Calls Summary ==="]
        for i, (name, snippet) in enumerate(tool_entries, start=1):
            lines.append(f"{i}. {name} — {snippet}")
        lines.append("==========================")
        summary_block = "\n".join(lines)
    else:
        summary_block = "=== Tool Calls Summary ===\n(no tool calls were made)\n=========================="

    last_ai_message = messages[-1]
    original_content = getattr(last_ai_message, "content", "")
    updated_content = f"{summary_block}\n\n{original_content}"

    return {"messages": [AIMessage(content=updated_content)]}


def build_graph():
    """Build an agent graph with a tool-call summarizer post-processing node."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())

    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("summarizer", summarizer_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_summarizer,
        {"action": "action", "summarizer": "summarizer"},
    )
    graph.add_edge("action", "agent")
    graph.add_edge("summarizer", END)

    return graph


graph = build_graph().compile()
