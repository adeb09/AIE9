"""An agent with guardrails using ``create_agent`` middleware.

Uses the modern ``langchain.agents.create_agent`` API with an
``AgentMiddleware`` subclass that validates inputs and outputs
via ``wrap_model_call`` — the recommended LangChain 1.0 pattern.

The middleware can **short-circuit** the model call entirely when
the user's input fails safety checks, and **replace** the model
response when the output fails content validation.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.models import get_chat_model
from app.tools import get_tool_belt
from app.guardrails import (
    create_guardrails_guard,
    validate_input,
    validate_output,
)


REFUSAL_INPUT = (
    "I'm sorry, but I can't process that request. "
    "Please ask a question related to cat health or feline care."
)
REFUSAL_OUTPUT = (
    "I apologize, but I'm unable to provide that response. "
    "Please rephrase your question."
)


class GuardrailsMiddleware(AgentMiddleware):
    """Middleware that wraps every model call with input & output guardrails.

    * **Input validation** — on the first model call (before any AI
      response exists), the latest ``HumanMessage`` is checked against
      topic-restriction and safety guards.  If validation fails the model
      call is **skipped** and a refusal is returned directly.

    * **Output validation** — after the model produces a final text
      response (not a tool-call), the content is checked for
      profanity, etc.  If validation fails the response is replaced
      with a safe refusal message.
    """

    def wrap_model_call(self, request, handler):
        messages = request.state["messages"]

        # --- Input guard (first iteration only) ---
        has_ai = any(isinstance(m, (AIMessage, ToolMessage)) for m in messages)
        if not has_ai:
            last_human = next(
                (m for m in reversed(messages) if isinstance(m, HumanMessage)),
                None,
            )
            if last_human is not None:
                guard = create_guardrails_guard(
                    valid_topics=[
                        "cat health", "feline care", "veterinary medicine",
                        "pet nutrition", "cat behavior",
                    ],
                    invalid_topics=[
                        "investment advice", "crypto", "gambling", "politics",
                    ],
                )
                result = validate_input(
                    guard, last_human.content, raise_on_failure=False,
                )
                if not result["validation_passed"]:
                    # Short-circuit: skip the model call entirely
                    return ModelResponse(
                        result=[AIMessage(content=REFUSAL_INPUT)],
                    )

        # --- Call the model ---
        response = handler(request)

        # --- Output guard (final text responses only) ---
        ai_msg = response.result[0] if response.result else None
        if (
            ai_msg
            and isinstance(ai_msg, AIMessage)
            and ai_msg.content
            and not getattr(ai_msg, "tool_calls", None)
        ):
            guard = create_guardrails_guard(
                enable_jailbreak_detection=False,
                enable_profanity_check=True,
            )
            result = validate_output(
                guard, ai_msg.content, raise_on_failure=False,
            )
            if not result["validation_passed"]:
                return ModelResponse(
                    result=[AIMessage(content=REFUSAL_OUTPUT)],
                )

        return response


# ---------------------------------------------------------------------------
# Build & export
# ---------------------------------------------------------------------------

graph = create_agent(
    model=get_chat_model(),
    tools=get_tool_belt(),
    middleware=[GuardrailsMiddleware()],
)
