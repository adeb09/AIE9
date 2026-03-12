"""A minimal tool-using agent graph.

Uses ``langchain.agents.create_agent`` — the modern LangChain 1.0 API —
to build a ReAct agent with the configured tool belt.
"""
from __future__ import annotations

from langchain.agents import create_agent

from app.models import get_chat_model
from app.tools import get_tool_belt


# Export compiled graph for LangGraph
graph = create_agent(
    model=get_chat_model(),
    tools=get_tool_belt(),
)
