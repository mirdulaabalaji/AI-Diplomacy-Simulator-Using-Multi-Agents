from graph.negotiation_graph import build_negotiation_graph
from schemas.state import NegotiationState, CountryState
import json


def get_user_input():
    print("Enter negotiation description (natural language):")
    text = input().strip()

    # Minimal fallback parser (your LLM parser already exists)
    from agents.input_parser import parse_natural_language_input
    return parse_natural_language_input(text)


def extract_actions(round_str):
    try:
        parts = round_str.split(":")[1].split(",")
        a = parts[0].split("=")[1].strip()
        b = parts[1].split("=")[1].strip()
        return a, b
    except Exception:
        return None, None


def is_successful_negotiation(state: NegotiationState):
    history = state.history

    if not history:
        return False, "No negotiation rounds occurred."

    # Rule 1: Final round mutual cooperation
    last_a, last_b = extract_actions(history[-1])
    if last_a == "C" and last_b == "C":
        return True, "Negotiation converged to mutual cooperation in the final round."

    # Rule 2: Late-stage convergence (2 of last 3 rounds)
    recent = history[-3:]
    coop = sum(
        1 for r in recent
        if extract_actions(r) == ("C", "C")
    )

    if coop >= 2:
        return True, "Negotiation showed late-stage cooperative convergence."

    return False, "Negotiation failed to reach a stable cooperative equilibrium."


def main():
    print("PYTHON IS RUNNING")
    print("DEBUG: main() started")

    # -------- INPUT --------
    state = get_user_input()

    # -------- RUN GRAPH --------
    graph = build_negotiation_graph()
    result = graph.invoke(state)

    # 🔥 CRITICAL FIX: normalize output
    if isinstance(result, dict):
        final_state = NegotiationState(**result)
    else:
        final_state = result

    # -------- JSON OUTPUT --------
    output = {
        "final_decision": final_state.history[-1] if final_state.history else None,
        "game_theory_models_used": final_state.game_models,
        "negotiation_rounds": final_state.round,
        "payoffs": {
            k: v.payoff for k, v in final_state.countries.items()
        },
        "goal_achievement": {
            k: round(v.trust, 2) for k, v in final_state.countries.items()
        },
        "problem_resolution": {
            k: round(v.trust, 2) for k, v in final_state.countries.items()
        },
        "negotiation_history": final_state.history,
    }

    print(json.dumps(output, indent=2))

    # -------- VERDICT --------
    success, explanation = is_successful_negotiation(final_state)

    if success:
        print(f"\n✅ Negotiation SUCCESSFUL: {explanation}")
    else:
        print(f"\n❌ Negotiation FAILED: {explanation}")

    print("\n✅ Negotiation complete")


if __name__ == "__main__":
    main()