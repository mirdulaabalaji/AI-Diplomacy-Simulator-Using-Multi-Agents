from schemas.state import NegotiationState
from agents.llm import invoke_llm

GAME_CLASSIFIER_PROMPT = """
You are a game theory expert.

SCENARIO:
{scenario}

Choose which game-theoretic models apply.

Choose ONLY from this list:
- Prisoner's Dilemma
- Repeated Game
- Free-Rider Problem
- Coalition Formation
- Nash Bargaining
- Deterrence Theory
- Brinkmanship Theory
- Escalation Control
- Information Asymmetry & Signaling

Return a comma-separated list.
Return ONLY the names.
"""


def classify_game(state: NegotiationState) -> NegotiationState:
    """
    Classifies which game theory models apply to the scenario.
    """

    response = invoke_llm(
        GAME_CLASSIFIER_PROMPT.format(scenario=state.scenario)
    )

    # response is already a STRING
    raw = response.strip()

    try:
        models = [m.strip() for m in raw.split(",") if m.strip()]
        state.game_models = models
    except Exception as e:
        raise ValueError(
            f"Invalid game model output from LLM: {raw}"
        ) from e

    return state