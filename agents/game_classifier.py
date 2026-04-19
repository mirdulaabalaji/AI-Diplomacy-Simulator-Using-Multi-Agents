from schemas.state import NegotiationState
from agents.llm import invoke_llm

# The full list of valid models — used for validation and fallback
VALID_MODELS = [
    "Prisoner's Dilemma",
    "Repeated Game",
    "Free-Rider Problem",
    "Coalition Formation",
    "Nash Bargaining",
    "Deterrence Theory",
    "Brinkmanship Theory",
    "Escalation Control",
    "Information Asymmetry & Signaling",
]

GAME_CLASSIFIER_PROMPT = """You are a game theory expert analyzing an international negotiation scenario.

SCENARIO:
{scenario}

Your task is to identify which game-theoretic models apply to this scenario.

You MUST choose from ONLY these exact names:
- Prisoner's Dilemma
- Repeated Game
- Free-Rider Problem
- Coalition Formation
- Nash Bargaining
- Deterrence Theory
- Brinkmanship Theory
- Escalation Control
- Information Asymmetry & Signaling

STRICT OUTPUT RULES:
- Return ONLY a comma-separated list of model names from the list above.
- Do NOT output single letters like "C" or "D".
- Do NOT output explanations, numbering, bullet points, or any other text.
- Do NOT output JSON.
- Choose between 2 and 5 models that best fit the scenario.

Example of correct output:
Prisoner's Dilemma, Nash Bargaining, Repeated Game

Example of wrong output:
C
1. Prisoner's Dilemma
Here are the models: Nash Bargaining
"""


def classify_game(state: NegotiationState) -> NegotiationState:
    """
    Classifies which game theory models apply to the scenario.
    Validates output against the known list and falls back gracefully.
    """
    response = invoke_llm(
        GAME_CLASSIFIER_PROMPT.format(scenario=state.scenario),
        agent_id="A",
    )

    raw = response.strip()

    # Strip any accidental preamble the model might add
    # e.g. "Here are the models: Nash Bargaining, ..."
    for preamble in ["here are the models:", "the applicable models are:",
                     "game theory models:", "models:"]:
        if raw.lower().startswith(preamble):
            raw = raw[len(preamble):].strip()

    # Parse comma-separated names
    candidates = [m.strip().strip("-").strip() for m in raw.split(",") if m.strip()]

    # Validate — only keep names that actually match our known list
    # Use case-insensitive matching so minor casing differences don't fail
    valid_lower = {m.lower(): m for m in VALID_MODELS}
    validated = []
    for candidate in candidates:
        match = valid_lower.get(candidate.lower())
        if match and match not in validated:
            validated.append(match)

    # If the LLM returned garbage (e.g. just "C"), fall back to the two
    # models that apply to virtually every negotiation scenario
    if not validated:
        print(f"[game_classifier] WARNING: LLM returned invalid output: '{raw}'. "
              f"Using fallback models.")
        validated = ["Prisoner's Dilemma", "Repeated Game"]

    state.game_models = validated
    print(f"[game_classifier] Models identified: {validated}")
    return state