"""
File Manager Agent – scaffolding for use in LangGraph workflows.

Provides file system tools for wellness plans and a reusable agent/node
that can be wired into supervisor or other graphs.

Usage:
    from file_manager_agent import (
        get_file_manager_tools,
        create_file_manager_agent,
        create_file_manager_node,
    )
    tools = get_file_manager_tools(plans_dir="./wellness_plans")
    agent = create_file_manager_agent(llm=specialist_llm, plans_dir="./wellness_plans")
    file_manager_node = create_file_manager_node(agent, name="file_manager")
    # Then: workflow.add_node("file_manager", file_manager_node)
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Default directory for wellness plans (override via get_file_manager_tools)
# ---------------------------------------------------------------------------
DEFAULT_PLANS_DIR = Path(__file__).resolve().parent / "wellness_plans"


def _ensure_plans_dir(plans_dir: Optional[Path] = None) -> Path:
    """Resolve and create plans directory if needed."""
    base = Path(plans_dir) if plans_dir is not None else DEFAULT_PLANS_DIR
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# File system tools (created with a specific plans_dir)
# ---------------------------------------------------------------------------

def get_file_manager_tools(plans_dir: Optional[Path] = None):
    """
    Return the list of file manager tools bound to `plans_dir`.
    Use this when you need to pass tools to create_agent or bind to an LLM.
    """
    base = _ensure_plans_dir(plans_dir)

    @tool
    def save_wellness_plan(filename: str, content: str) -> str:
        """Save a wellness plan to a markdown file. Use a short filename without path (e.g. 'my_plan' or 'weekly_plan')."""
        path = base / _ensure_md(filename)
        path.write_text(content, encoding="utf-8")
        return f"Saved wellness plan to {path.name}."

    @tool
    def load_wellness_plan(filename: str) -> str:
        """Load an existing wellness plan by filename (e.g. 'my_plan' or 'weekly_plan.md')."""
        path = base / _ensure_md(filename)
        if not path.exists():
            return f"No plan found with filename: {filename}"
        return path.read_text(encoding="utf-8")

    @tool
    def list_saved_plans() -> str:
        """List all saved wellness plans (markdown files in the plans directory)."""
        if not base.exists():
            return "No plans directory found; no plans saved yet."
        files = sorted(p.name for p in base.glob("*.md"))
        if not files:
            return "No saved wellness plans yet."
        return "Saved plans:\n" + "\n".join(f"  - {f}" for f in files)

    @tool
    def append_to_plan(filename: str, section: str, content: str) -> str:
        """Add a section to an existing wellness plan. Appends a new markdown section with the given title and content."""
        path = base / _ensure_md(filename)
        if not path.exists():
            return f"Cannot append: no plan found with filename '{filename}'."
        existing = path.read_text(encoding="utf-8")
        new_section = f"\n\n## {section}\n\n{content}\n"
        path.write_text(existing + new_section, encoding="utf-8")
        return f"Appended section '{section}' to {path.name}."

    return [
        save_wellness_plan,
        load_wellness_plan,
        list_saved_plans,
        append_to_plan,
    ]


def _ensure_md(name: str) -> str:
    """Ensure filename has .md extension."""
    return name if name.endswith(".md") else f"{name}.md"


# ---------------------------------------------------------------------------
# Agent and node factory (for use in LangGraph)
# ---------------------------------------------------------------------------

def create_file_manager_agent(llm, plans_dir: Optional[Path] = None, system_prompt: Optional[str] = None):
    """
    Create a File Manager agent that can save/load/list/append wellness plans.

    Args:
        llm: LangChain chat model (e.g. specialist_llm or ChatOpenAI(...)).
        plans_dir: Directory for plan files; default is DEFAULT_PLANS_DIR.
        system_prompt: Optional custom system prompt.

    Returns:
        An agent (e.g. from create_agent) you can invoke or wrap in a node.
    """
    from langchain.agents import create_agent

    tools = get_file_manager_tools(plans_dir)
    prompt = system_prompt or (
        "You are a File Manager for wellness plans. You can save new plans to markdown files, "
        "load existing plans, list all saved plans, and append new sections to a plan. "
        "Use the appropriate tool based on the user's request. Be concise and confirm actions."
    )
    return create_agent(model=llm, tools=tools, system_prompt=prompt)


def create_file_manager_node(agent, name: str = "file_manager"):
    """
    Wrap the File Manager agent as a LangGraph node.

    The returned node expects state with a "messages" key (list of BaseMessage)
    and returns {"messages": [response_message]}. Compatible with SupervisorState
    and similar state schemas that use "messages" and optional "next".

    Use in a graph:
        file_manager_node = create_file_manager_node(file_manager_agent, "file_manager")
        workflow.add_node("file_manager", file_manager_node)
        workflow.add_conditional_edges("supervisor", route_to_agent, {..., "file_manager": "file_manager"})
        workflow.add_edge("file_manager", END)
    """
    from langchain_core.messages import AIMessage

    def node(state):
        # state must have "messages"; "next" is optional
        result = agent.invoke({"messages": state["messages"]})
        last = result["messages"][-1]
        response_with_name = AIMessage(
            content=f"[{name.upper()}]\n\n{last.content}",
            name=name,
        )
        return {"messages": [response_with_name]}

    return node
