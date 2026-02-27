from schemas.state import NegotiationState


def evaluate_round(
    state: NegotiationState,
    action_a: str,
    action_b: str,
    country_a: str,
    country_b: str,
) -> NegotiationState:
    """
    Updates payoffs, trust, memory, and termination condition.
    """

    state.round += 1
    state.history.append(
        f"Round {state.round}: {country_a}={action_a}, {country_b}={action_b}"
    )

    A = state.countries[country_a]
    B = state.countries[country_b]

    # --- PAYOFFS (Prisoner's Dilemma baseline) ---
    if action_a == "C" and action_b == "C":
        A.payoff += 3
        B.payoff += 3
        A.trust += 0.1
        B.trust += 0.1

        # ✅ TERMINATE IMMEDIATELY
        state.agreement_reached = True

    elif action_a == "C" and action_b == "D":
        A.payoff -= 1
        B.payoff += 4
        A.trust -= 0.1
        B.trust -= 0.05

    elif action_a == "D" and action_b == "C":
        A.payoff += 4
        B.payoff -= 1
        A.trust -= 0.05
        B.trust -= 0.1

    else:  # D, D
        A.payoff += 0
        B.payoff += 0
        A.trust -= 0.05
        B.trust -= 0.05

    # --- Clamp trust values ---
    A.trust = max(0.0, min(1.0, A.trust))
    B.trust = max(0.0, min(1.0, B.trust))

    return state