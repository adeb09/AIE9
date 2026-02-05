### YOUR CODE HERE ###
from typing import Annotated, TypedDict

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Step 1: Define a wellness profile schema
# Example attributes: name, age, goals, conditions, allergies, fitness_level, preferred_activities

## Wellness Schema
# name: str
# age: int
# goals: dict
# conditions: dict
# fitness_level: str
# preferred_activities: list[str]

# Step 2: Create helper functions to store and retrieve profiles
def store_wellness_profile(store, user_id: str, profile: dict) -> None:
    """Store a user's wellness profile."""
    profile_namespace = (user_id, "profile")

    # gather all profile keys and store values in store
    for key, value in profile.items():
        store.put(profile_namespace, key, value)


def get_wellness_profile(store, user_id: str) -> dict:
    """Retrieve a user's wellness profile."""
    profile_namespace = (user_id, "profile")
    return {item.key: item.value for item in store.search(profile_namespace)}

def get_wellness_profile_str(profile: dict):
    """Retrieve a user's wellness profile as a string (ready to utilize as context in prompt)"""
    return "\n".join([f"- {key}: {value}" for key, value in profile.items()])


# Step 3: Create two different user profiles
profile_a = {
    "name": "Alberto",
    "age": 42,
    "goals": {"primary": "losing weight", "secondary": "lowering cholesterol"},
    "conditions": {"diagnoses": ["high cholesterol", "type-2 diabetes"], "allergies": ["pollen"]},
    "fitness_level": "sedentary (barely any physical weekly activity)",
    "preferred_activities": ["walking", "hiking", "basketball"]
}
user_a = f"user_{profile_a.get('name').lower()}"

profile_b = {
    "name": "Carolina",
    "age": 25,
    "goals": {"primary": "getting tone", "secondary": "higher quality sleep"},
    "conditions": {"diagnoses": ["insomnia"], "allergies": ["dust mites"]},
    "fitness_level": "intermediate (4-6 hrs of physical activity each week)",
    "preferred_activities": ["pilates", "pickleball"]
}
user_b = f"user_{profile_b.get('name').lower()}"

user_profiles = [(user_a, profile_a), (user_b, profile_b)]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
store = InMemoryStore()

for user, profile in user_profiles:
    store_wellness_profile(store, user, profile)

## retrieve profiles
# for user, _ in user_profiles:
#     print(get_wellness_profile_str(get_wellness_profile(store, user)))
#     print("\n\n")


# Step 4: Build a personalized agent that uses profiles
class PersonalizedState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


def personalized_wellness_agent(state: PersonalizedState, store: BaseStore):
    user_id = state["user_id"]
    wellness_profile_str = get_wellness_profile_str(get_wellness_profile(store, user_id))
    system_msg = f"""You are a Personal Wellness Assistant. You know the following about this user:\n\n{wellness_profile_str}"""
    messages = [system_msg] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# build graph
act1_builder = StateGraph(PersonalizedState)
act1_builder.add_node("wellness_agent", personalized_wellness_agent)
act1_builder.add_edge(START, "wellness_agent")
act1_builder.add_edge("wellness_agent", END)

act1_graph = act1_builder.compile(store=store)

# Step 5: Test with different users - they should get different advice
q1 = "What sort of workout regime would you suggest for me?"
response1 = act1_graph.invoke(
    {
        "messages": [HumanMessage(content=q1)],
        "user_id": "user_alberto"
    },
    store=store
)
print(f"Alberto: {q1}")
print(f"Wellness Agent: {response1['messages'][-1].content}")

response2 = act1_graph.invoke(
    {
        "messages": [HumanMessage(content=q1)],
        "user_id": "user_carolina"
    },
    store=store
)
print(f"Carolina: {q1}")
print(f"Wellness Agent: {response2['messages'][-1].content}")