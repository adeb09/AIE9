from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graphs.simple_agent import graph


def display_messages(messages):
    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(f"[human] {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"[tool] {msg.name}: {msg.content[:200]}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                calls = ", ".join(tc["name"] for tc in msg.tool_calls)
                print(f"[ai] calling tools: {calls}")
            else:
                print(f"[ai] {msg.content}")
        else:
            print(f"[{msg.type}] {msg.content}")


def main():
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What are the recommended vaccinations for kittens? use retrieve_information tool."
                )
            ]
        }
    )
    display_messages(result["messages"])


if __name__ == "__main__":
    main()
