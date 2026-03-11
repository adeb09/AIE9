import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, Role, TextPart
from langchain.agents import create_agent
from langchain.tools import tool

# -- LangChain Tools ---------------------------------------------------------

CATALOG = [
    {"id": 1, "name": "Feather Wand", "price": 8.99, "category": "toys"},
    {"id": 2, "name": "Laser Pointer", "price": 12.99, "category": "toys"},
    {"id": 3, "name": "Cozy Donut Bed", "price": 34.99, "category": "beds"},
    {"id": 4, "name": "Salmon Crunchies", "price": 6.49, "category": "food"},
    {"id": 5, "name": "Cat Tree Tower", "price": 89.99, "category": "furniture"},
    {"id": 6, "name": "Catnip Mouse", "price": 4.99, "category": "toys"},
    {"id": 7, "name": "Self-Warming Mat", "price": 22.99, "category": "beds"},
    {"id": 8, "name": "Tuna Treats", "price": 5.99, "category": "food"},
]


@tool
def list_products(category: str | None = None) -> list[dict]:
    """List products from the cat shop catalog. Optionally filter by category: toys, beds, food, furniture."""
    if category:
        return [p for p in CATALOG if p["category"] == category.lower()]
    return CATALOG


@tool
def get_product(product_id: int) -> dict:
    """Get details for a specific product by its ID."""
    for p in CATALOG:
        if p["id"] == product_id:
            return p
    return {"error": "Product not found"}


# -- LangChain Agent ---------------------------------------------------------

SYSTEM_PROMPT = """You are the Cat Shop assistant. You help customers find the perfect products for their cats.
You have access to tools to browse the product catalog. Be friendly, helpful, and concise.
Always use your tools to look up products rather than guessing."""

def _build_agent():
    return create_agent(
        "openai:gpt-4.1-nano",
        tools=[list_products, get_product],
        system_prompt=SYSTEM_PROMPT,
    )


# -- A2A Agent Executor ------------------------------------------------------


class CatShopAgentExecutor(AgentExecutor):
    """Bridges the LangChain agent into the A2A protocol."""

    def __init__(self):
        self._agent = None

    @property
    def agent(self):
        if self._agent is None:
            self._agent = _build_agent()
        return self._agent

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_text = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if isinstance(part.root, TextPart):
                    user_text += part.root.text

        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": user_text}]}
        )

        # Extract the last agent message from the response
        output = result["messages"][-1].content

        response = Message(
            role=Role.agent,
            message_id=uuid.uuid4().hex,
            parts=[Part(root=TextPart(text=output))],
        )
        await event_queue.enqueue_event(response)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        response = Message(
            role=Role.agent,
            message_id=uuid.uuid4().hex,
            parts=[Part(root=TextPart(text="Request cancelled."))],
        )
        await event_queue.enqueue_event(response)
