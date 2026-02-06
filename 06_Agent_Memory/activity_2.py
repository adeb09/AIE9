### YOUR CODE HERE ###
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# Create embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a store with semantic search enabled & for retrieving long-term memory
semantic_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,  # Dimension of text-embedding-3-small
    }
)
MAX_ITEMS = 10 ** 9 # max items for .search method for store

# Load and chunk the wellness document
loader = TextLoader("data/HealthWellnessGuide.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# Store chunks in semantic memory
knowledge_namespace = ("wellness", "knowledge")

for i, chunk in enumerate(chunks):
    semantic_store.put(
        knowledge_namespace,
        f"chunk_{i}",
        {"text": chunk.page_content, "source": "HealthWellnessGuide.txt"}
    )


# Create episodic memory with past successful interactions
episodic_namespace = ("agent", "episodes")

successful_episodes = [
    {
        "situation": "User asked about starting exercise with joint pain",
        "input": "I want to start exercising but my knees hurt. What should I do?",
        "output": """For knee-friendly exercise, I'd recommend starting with low-impact activities:

1. **Swimming or water aerobics** - Water supports your joints while providing great cardio
2. **Cycling** - Either stationary or regular bikes are gentle on knees
3. **Elliptical machine** - Provides cardio without impact

Start slowly with 15-20 minutes and gradually increase. If you experience pain, stop and consult a healthcare provider. Would you like specific exercises for any of these?""",
        "feedback": "User appreciated the specific alternatives and safety advice"
    },
    {
        "situation": "User struggling with sleep and stress",
        "input": "I can't fall asleep because I'm always thinking about work. Help!",
        "output": """Racing thoughts at bedtime is common, especially with work stress. Here's a proven approach:

**Before bed:**
- Write down tomorrow's to-do list (gets thoughts out of your head)
- Set a "worry time" earlier in the evening - not at bedtime

**At bedtime:**
- Try the 4-7-8 breathing technique: Inhale 4 counts, hold 7, exhale 8
- Use the "cognitive shuffle" - think of random, unrelated objects (apple, guitar, mountain...)

Which of these would you like to try first?""",
        "feedback": "User found the structured approach and specific techniques helpful"
    },
    {
        "situation": "User asking about healthy eating on a budget",
        "input": "How can I eat healthy when I don't have much money?",
        "output": """Eating healthy on a budget is absolutely possible! Here are practical strategies:

**Budget-friendly nutritious foods:**
- Eggs, beans, lentils (cheap protein)
- Frozen vegetables (same nutrition as fresh, longer lasting)
- Oats, rice, whole grain bread
- Canned fish (tuna, salmon)

**Money-saving tips:**
- Buy in bulk when possible
- Plan meals around sales
- Cook in batches and freeze portions

What's your typical weekly food budget? I can help you create a specific meal plan.""",
        "feedback": "User valued the practical, actionable advice without judgment"
    },
]

for i, episode in enumerate(successful_episodes):
    semantic_store.put(
        episodic_namespace,
        f"episode_{i}",
        {
            "text": episode["situation"],  # Used for semantic search
            **episode
        }
    )


# Step 1: Define wellness metrics schema and storage functions
def log_wellness_metric(store: BaseStore, user_id: str, date: str, metric_type: str, value: str, notes: str = ""):
    """Log a wellness metric for a user."""
    metric_namespace = (user_id, "metric")
    key = f"{metric_type}#{date}"
    value = {"value": value, "date": date, "notes": notes} if notes else {"value": value, "date": date}
    store.put(metric_namespace, key, value)


def get_wellness_history(store: BaseStore, user_id: str, metric_type: str = None, days: int = 7) -> list:
    """Get wellness history for a user."""
    metric_namespace = (user_id, "metric")
    to_return = []
    dates = set()
    for item in store.search(metric_namespace, limit=MAX_ITEMS):
        metric, date = item.key.split("#")
        dates.add(date)
        if len(dates) <= days and metric == metric_type:
            to_return.append(item.value)
    return to_return


# Step 2: Create sample wellness data for a user (simulate a week)
user_1 = "user_alberto"
metrics_alberto = [
    # dict of daily metrics
    {"date": "1/1", "mood": "happy", "energy": "rested", "sleep_quality": "deep", "notes": "beginning of the week starting off great!"},
    {"date": "1/2", "mood": "neutral", "energy": "energetic", "sleep_quality": "restless", "notes": "rough start to the week at work"},
    {"date": "1/3", "mood": "stressed", "energy": "focused", "sleep_quality": "interrupted", "notes": "needed to meet some deadlines today"},
    {"date": "1/4", "mood": "tired", "energy": "exhausted", "sleep_quality": "light", "notes": "in a midweek rut, can't wait for the weekend"},
    {"date": "1/5", "mood": "cheerful", "energy": "vibrant", "sleep_quality": "deep", "notes": "productive day with good flow"},
    {"date": "1/6", "mood": "frustrated", "energy": "drained", "sleep_quality": "light", "notes": "unexpected meeting drained energy"},
    {"date": "1/7", "mood": "relaxed", "energy": "calm", "sleep_quality": "restful", "notes": "weekend prep, feeling balanced"}
]

user_2 = "user_carolina"
metrics_carolina = [
    {"date":"1/1","mood":"Neutral","energy":"Alert","sleep_quality":"Light","notes":"Just started the week, feeling neutral."},
    {"date":"1/2","mood":"Joyful","energy":"Energetic","sleep_quality":"Deep","notes":"Morning sunshine boosted my mood."},
    {"date":"1/3","mood":"Stressed","energy":"Focused","sleep_quality":"Interrupted","notes":"Work deadline caused some stress."},
    {"date":"1/4","mood":"Tired","energy":"Lazy","sleep_quality":"Restless","notes":"Long meeting made me feel exhausted."},
    {"date":"1/5","mood":"Happy","energy":"Rested","sleep_quality":"Dreamy","notes":"Nice dinner, feels relaxed."},
    {"date":"1/6","mood":"Anxious","energy":"Exhausted","sleep_quality":"Caffeinated","notes":"Coffee kept me awake after the night shift."},
    {"date":"1/7","mood":"Neutral","energy":"Focused","sleep_quality":"Deep","notes":"Weekend prep for next week."}
]

users = [user_1, user_2]
metrics = [metrics_alberto, metrics_carolina]

###############################################################################
# load simulated data into in-memory datastore (simulating long-term storage) #
###############################################################################
for user_id, user_metrics in zip(users, metrics):
    for daily_log in user_metrics:
        mood = daily_log.get("mood", None)
        energy = daily_log.get("energy", None)
        sleep_quality = daily_log.get("sleep_quality", None)
        date = daily_log.get("date", None)
        notes = daily_log.get("notes", None)
        zipped = zip(["mood","energy","sleep_quality"], [mood,energy,sleep_quality])
        for metric, val in zipped:
            log_wellness_metric(semantic_store, user_id, date, metric, val, notes)

# Step 3: Build a wellness dashboard agent that:
#   - Retrieves user's wellness history
#   - Searches for relevant advice based on patterns
#   - Uses episodic memory for what worked before
#   - Generates a personalized summary
# create instance of openAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def summarize_conversation(messages: list, max_messages: int = 6) -> list:
    """Summarize older messages to manage context length."""
    if len(messages) <= max_messages:
        return messages

    # Keep the system message and last few messages
    system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
    content_messages = messages[1:] if system_msg else messages

    if len(content_messages) <= max_messages:
        return messages

    old_messages = content_messages[:-max_messages + 1]
    recent_messages = content_messages[-max_messages + 1:]

    # Summarize old messages
    summary_prompt = f"""Summarize this conversation in 2-3 sentences, 
capturing key wellness topics discussed and any important user information:

{chr(10).join([f'{type(m).__name__}: {m.content[:200]}' for m in old_messages])}"""

    summary = llm.invoke(summary_prompt)

    # Return: system + summary + recent messages
    result = []
    if system_msg:
        result.append(system_msg)
    result.append(SystemMessage(content=f"[Previous conversation summary: {summary.content}]"))
    result.extend(recent_messages)

    return result


# define state for this agentic system
class UnifiedState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    feedback: str


def unified_wellness_assistant(state: UnifiedState, store: BaseStore):
    """An assistant that uses all five memory types."""
    user_id = state["user_id"]
    user_message = state["messages"][-1].content

    # 1. PROCEDURAL: Get current instructions
    instructions_item = store.get(("agent", "instructions"), "wellness_assistant")
    base_instructions = instructions_item.value[
        "instructions"] if instructions_item else "You are a helpful wellness assistant."

    # 2. LONG-TERM: get wellness data
    mood = "\n".join(
        f"- Date: {item.get('date')} - Mood: {item.get('value')} - Notes: {item.get('notes')}"
        for item in get_wellness_history(store, user_id, "mood")
    )

    energy = "\n".join(
        f"- Date: {item.get('date')} - Energy: {item.get('value')} - Notes: {item.get('notes')}"
        for item in get_wellness_history(store, user_id, "energy")
    )

    sleep_quality = "\n".join(
        f"- Date: {item.get('date')} - Sleep quality: {item.get('value')} - Notes: {item.get('notes')}"
        for item in get_wellness_history(store, user_id, "sleep_quality")
    )

    wellness_history = "\n".join([mood, energy, sleep_quality])

    # 3. SEMANTIC: Search for relevant knowledge
    relevant_knowledge = store.search(("wellness", "knowledge"), query=user_message, limit=2)
    knowledge_text = "\n".join([f"- {r.value['text'][:200]}..." for r in
                                relevant_knowledge]) if relevant_knowledge else "No specific knowledge found."

    # 4. EPISODIC: Find similar past interactions
    similar_episodes = store.search(("agent", "episodes"), query=user_message, limit=1)
    if similar_episodes:
        ep = similar_episodes[0].value
        episode_text = f"Similar past interaction:\nUser: {ep.get('input', 'N/A')}\nResponse style: {ep.get('feedback', 'N/A')}"
    else:
        episode_text = "No similar past interactions found."

    # Build comprehensive system message
    system_message = f"""{base_instructions}

=== USER WELLNESS HISTORY ===
{wellness_history}

=== RELEVANT WELLNESS KNOWLEDGE ===
{knowledge_text}

=== LEARNING FROM EXPERIENCE ===
{episode_text}

Use all of this context to provide the best possible personalized response."""

    # 5. SHORT-TERM: Full conversation history is automatically managed by the checkpointer
    # Use summarization for long conversations
    trimmed_messages = summarize_conversation(state["messages"], max_messages=6)

    messages = [SystemMessage(content=system_message)] + trimmed_messages
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build the unified graph
act2_builder = StateGraph(UnifiedState)
act2_builder.add_node("wellness_assistant", unified_wellness_assistant)

act2_builder.add_edge(START, "wellness_assistant")
act2_builder.add_edge("wellness_assistant", END)

# Compile with semantic store
act2_graph = act2_builder.compile(
    store=semantic_store
)

# Step 4: Test the dashboard
# Example: "Give me a summary of my wellness this week"
# Example: "I've been feeling tired lately. What might help?"
q1 = "Give me a summary of my wellness this week?"
response1 = act2_graph.invoke(
    {
        "messages": [HumanMessage(content=q1)],
        "user_id": "user_alberto"
    },
    store=semantic_store
)
print(f"Alberto: {q1}")
print(f"Wellness Agent: {response1['messages'][-1].content}")


response1 = act2_graph.invoke(
    {
        "messages": [HumanMessage(content=q1)],
        "user_id": "user_carolina"
    },
    store=semantic_store
)
print(f"Carolina: {q1}")
print(f"Wellness Agent: {response1['messages'][-1].content}")