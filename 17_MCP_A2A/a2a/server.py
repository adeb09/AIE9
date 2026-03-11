import uvicorn
import dotenv

from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
)

from agent_executor import CatShopAgentExecutor

dotenv.load_dotenv()

HOST = "0.0.0.0"
PORT = 9999

# -- Skills ------------------------------------------------------------------

recommend_skill = AgentSkill(
    id="recommend_products",
    name="Product Recommendations",
    description=(
        "Recommend cat products based on what the user is looking for. "
        "Supports categories: toys, beds, food, furniture."
    ),
    tags=["shopping", "cats", "recommendations"],
    examples=[
        "My cat needs a new toy",
        "What beds do you have?",
        "Recommend some treats",
        "Surprise me!",
    ],
    input_modes=["text/plain"],
    output_modes=["text/plain"],
)

# -- Agent Card ---------------------------------------------------------------

agent_card = AgentCard(
    name="Cat Shop Assistant",
    description="An A2A agent that recommends products from the Cat Shop catalog.",
    version="1.0.0",
    url=f"http://localhost:{PORT}/",
    capabilities=AgentCapabilities(
        streaming=False,
        push_notifications=False,
    ),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[recommend_skill],
)

# -- Server -------------------------------------------------------------------

request_handler = DefaultRequestHandler(
    agent_executor=CatShopAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

server_app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
)

app = server_app.build()

if __name__ == "__main__":
    print(f"Cat Shop A2A Server running at http://localhost:{PORT}")
    print(f"Agent Card: http://localhost:{PORT}/.well-known/agent-card.json")
    uvicorn.run(app, host=HOST, port=PORT)
