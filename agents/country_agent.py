from schemas.state import NegotiationState, CountryMemory
from agents.llm import invoke_llm   # ✅ THIS WAS MISSING


def country_decide(
    state: NegotiationState,
    country_name: str,
    opponent_name: str
) -> str:
    """
    Uses LLM reasoning + memory to decide whether to
    Cooperate (C) or Defect (D).
    """

    country = state.countries[country_name]
    opponent = state.countries[opponent_name]

    # Fetch or initialize memory about opponent
    memory = country.memory.get(opponent_name)
    if memory is None:
        memory = CountryMemory(opponent=opponent_name)
        country.memory[opponent_name] = memory

    prompt = f"""
You are {country.name}, a rational country negotiating internationally.

SCENARIO:
{state.scenario}

YOUR PROFILE:
Goals: {country.goals}
Problems: {country.problems}
Resources: {country.resources}

OPPONENT PROFILE:
Opponent: {opponent.name}
Observed cooperation rate: {memory.cooperation_rate}
Betrayal count: {memory.betrayal_count}

NEGOTIATION HISTORY:
{state.history}

GAME MODELS IN PLAY:
{state.game_models}

RULES:
- Prefer cooperation if trust is high
- Retaliate if betrayal count is high
- Defect if short-term payoff dominates
- Consider long-term benefits (repeated game)

Choose ONE action only:
"C" for Cooperate
"D" for Defect

Return ONLY the single letter.
"""

    response = invoke_llm(prompt).strip().upper()

    if response not in {"C", "D"}:
        # Safe fallback
        response = "C" if country.trust >= 0.5 else "D"

    return response