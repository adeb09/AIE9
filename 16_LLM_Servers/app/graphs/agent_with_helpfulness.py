"""An agent graph with a post-response helpfulness check loop.

After the agent responds, a secondary node evaluates helpfulness.
If helpful, end; otherwise, continue the loop or terminate after a safe limit.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

from app.state import MessagesState
from app.models import get_chat_model, fix_tool_calls
from app.tools import get_tool_belt


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    response = fix_tool_calls(model.invoke(messages))
    return {"messages": [response]}


def route_to_action_or_helpfulness(state: MessagesState):
    """Decide whether to execute tools or run the helpfulness evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "helpfulness"


_helpfulness_prompt = ChatPromptTemplate.from_template(
    "Given an initial query and a final response, determine if the final response "
    "is extremely helpful or not. Respond with only Y or N.\n\n"
    "Initial Query:\n{initial_query}\n\n"
    "Final Response:\n{final_response}"
)


def helpfulness_node(state: MessagesState) -> dict:
    """Evaluate helpfulness of the latest response relative to the initial query."""
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="HELPFULNESS:END")]}

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    chain = _helpfulness_prompt | get_chat_model() | StrOutputParser()
    result = chain.invoke(
        {
            "initial_query": initial_query.content,
            "final_response": final_response.content,
        }
    )

    decision = "Y" if result.strip().upper().startswith("Y") else "N"
    return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}


def helpfulness_decision(state: MessagesState):
    """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
    if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
        return END

    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if "HELPFULNESS:Y" in text:
        return "end"
    return "continue"


def build_graph():
    """Build an agent graph with an auxiliary helpfulness evaluation subgraph."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()
