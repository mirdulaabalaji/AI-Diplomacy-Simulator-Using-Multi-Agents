"""
evaluator.py

Payoff and trust update logic for a 4-agent negotiation.

Coalition rules:
  - A country can only be in ONE coalition at a time.
  - Coalition forms when A proposes B AND B proposes A AND both cooperate.
  - If A is already in a coalition with X and now proposes Y:
      → A exits the coalition with X first (X loses their partner too).
      → Then A attempts to form a new coalition with Y.
  - Coalition bonus only applies to the confirmed, active coalition pair.
  - The final output proposed_ally reflects the last confirmed partner.

Game-theory concepts:
  - Prisoner's Dilemma baseline payoffs
  - Coalition Formation with exclusivity constraint
  - Free-Rider Penalty
  - Deterrence via trust erosion
  - Repeated Game via trust accumulation
  - Nash Bargaining equilibrium check
"""

from schemas.state import NegotiationState
from game_theory.payoff_matrices import (
    PRISONERS_DILEMMA,
    COALITION_BONUS,
    FREE_RIDER_PENALTY,
)
from game_theory.equilibrium import brinkmanship_risk

MAX_ROUNDS = 5


def _base_payoff(action: str, others_actions: list) -> float:
    payoff = 0.0
    for other in others_actions:
        payoff += PRISONERS_DILEMMA.get((action, other), (0, 0))[0]
    return payoff


def _update_memory(country, opponent_id: str, action: str, opponent_action: str):
    from schemas.state import CountryMemory
    if opponent_id not in country.memory:
        country.memory[opponent_id] = CountryMemory(opponent=opponent_id)
    mem = country.memory[opponent_id]
    if opponent_action == "C":
        mem.cooperation_rate = 0.9 * mem.cooperation_rate + 0.1 * 1.0
    else:
        mem.cooperation_rate = 0.9 * mem.cooperation_rate + 0.1 * 0.0
    if action == "C" and opponent_action == "D":
        mem.betrayal_count += 1
    if action == "C" and opponent_action == "C":
        mem.alliance_count += 1


def _resolve_coalitions(
    state: NegotiationState,
    actions: dict,
    proposed_allies: dict,
) -> tuple[list, dict]:
    """
    Resolves coalition changes for this round with the exclusivity constraint.

    Rules applied in order:
    1. A country proposing a NEW partner different from its current one
       → exits current coalition first (both sides lose the partner).
    2. A coalition only FORMS if both sides proposed each other this round
       AND both cooperated.
    3. An existing coalition PERSISTS if neither side proposed a different
       partner this round AND both cooperated.
    4. A coalition BREAKS if either member defected.

    Returns:
      - confirmed_coalitions: list of [id_a, id_b] pairs active this round
      - raw_payoffs_adjustment: dict of additional payoffs from coalitions
    """
    agent_ids = list(state.countries.keys())
    payoff_adj = {k: 0.0 for k in agent_ids}

    # Step 1: Handle coalition exits
    # If a country is proposing someone different from its current partner,
    # it is signalling it wants to leave — dissolve the old coalition.
    for agent_id in agent_ids:
        current_partner = state.countries[agent_id].current_coalition_partner
        new_proposal = proposed_allies.get(agent_id)

        if current_partner is not None and new_proposal != current_partner:
            # This agent wants out — dissolve both sides
            print(f"  💔 {state.countries[agent_id].name} is leaving coalition "
                  f"with {state.countries[current_partner].name}")
            state.countries[agent_id].current_coalition_partner = None
            # Also clear the old partner's side
            if state.countries[current_partner].current_coalition_partner == agent_id:
                state.countries[current_partner].current_coalition_partner = None

    # Step 2: Attempt to form or confirm coalitions
    confirmed = []
    already_paired = set()  # prevent a country being in 2 coalitions

    for i, id_a in enumerate(agent_ids):
        for id_b in agent_ids[i + 1:]:
            if id_a in already_paired or id_b in already_paired:
                continue

            both_cooperated = actions[id_a] == "C" and actions[id_b] == "C"
            a_wants_b = proposed_allies.get(id_a) == id_b
            b_wants_a = proposed_allies.get(id_b) == id_a

            current_a = state.countries[id_a].current_coalition_partner
            current_b = state.countries[id_b].current_coalition_partner

            # Case 1: NEW coalition — both proposed each other AND both cooperated
            if a_wants_b and b_wants_a and both_cooperated:
                state.countries[id_a].current_coalition_partner = id_b
                state.countries[id_b].current_coalition_partner = id_a
                state.countries[id_a].proposed_ally = id_b
                state.countries[id_b].proposed_ally = id_a
                confirmed.append([id_a, id_b])
                already_paired.add(id_a)
                already_paired.add(id_b)
                payoff_adj[id_a] += COALITION_BONUS
                payoff_adj[id_b] += COALITION_BONUS
                print(f"  🤝 NEW coalition formed: "
                      f"{state.countries[id_a].name} ↔ {state.countries[id_b].name}")

            # Case 2: EXISTING coalition persists — same partners, both cooperated
            elif (current_a == id_b and current_b == id_a and both_cooperated):
                confirmed.append([id_a, id_b])
                already_paired.add(id_a)
                already_paired.add(id_b)
                payoff_adj[id_a] += COALITION_BONUS
                payoff_adj[id_b] += COALITION_BONUS
                print(f"  ✅ Coalition maintained: "
                      f"{state.countries[id_a].name} ↔ {state.countries[id_b].name}")

            # Case 3: One side proposed but the other didn't — no coalition
            # Case 4: Either side defected — coalition breaks
            elif current_a == id_b and current_b == id_a and not both_cooperated:
                print(f"  💔 Coalition broken (defection): "
                      f"{state.countries[id_a].name} ↔ {state.countries[id_b].name}")
                state.countries[id_a].current_coalition_partner = None
                state.countries[id_b].current_coalition_partner = None

    return confirmed, payoff_adj


def evaluate_round(
    state: NegotiationState,
    actions: dict,
    proposed_allies: dict,
) -> NegotiationState:
    state.round += 1
    agent_ids = list(state.countries.keys())

    # ---- 1. Base payoffs ----
    raw_payoffs = {}
    for agent_id in agent_ids:
        others = [actions[k] for k in agent_ids if k != agent_id]
        raw_payoffs[agent_id] = _base_payoff(actions[agent_id], others)

    # ---- 2. Coalition resolution (with exclusivity) ----
    confirmed_coalitions, coalition_adj = _resolve_coalitions(
        state, actions, proposed_allies
    )
    state.active_coalitions = confirmed_coalitions

    for agent_id, bonus in coalition_adj.items():
        raw_payoffs[agent_id] += bonus

    # Log coalition state this round
    state.coalition_history.append({
        "round": state.round,
        "active": confirmed_coalitions,
        "memberships": {
            k: state.countries[k].current_coalition_partner
            for k in agent_ids
        }
    })

    # ---- 3. Free-rider penalty ----
    cooperators = [k for k in agent_ids if actions[k] == "C"]
    defectors = [k for k in agent_ids if actions[k] == "D"]

    if len(cooperators) >= 3 and len(defectors) >= 1:
        for d in defectors:
            raw_payoffs[d] -= FREE_RIDER_PENALTY

    # ---- 4. Apply payoffs and update trust ----
    for agent_id in agent_ids:
        country = state.countries[agent_id]
        country.payoff += raw_payoffs[agent_id]

        cooperating_opponents = sum(
            1 for k in agent_ids if k != agent_id and actions[k] == "C"
        )
        total_opponents = len(agent_ids) - 1

        if actions[agent_id] == "C":
            trust_delta = 0.05 * cooperating_opponents / total_opponents
            country.trust = min(1.0, country.trust + trust_delta)
        else:
            country.trust = max(0.0, country.trust - 0.08)

    # ---- 5. Update per-opponent memory ----
    for agent_id in agent_ids:
        for opp_id in agent_ids:
            if opp_id == agent_id:
                continue
            _update_memory(
                state.countries[agent_id],
                opp_id,
                actions[agent_id],
                actions[opp_id],
            )

    # ---- 6. History entry ----
    action_str = ", ".join(f"{k}={v}" for k, v in actions.items())
    coalition_str = ""
    if confirmed_coalitions:
        pairs = "+".join(f"[{a}↔{b}]" for a, b in confirmed_coalitions)
        coalition_str = f" | Coalitions: {pairs}"
    state.history.append(f"Round {state.round}: {action_str}{coalition_str}")

    # ---- 7. Agreement / failure check ----
    if all(a == "C" for a in actions.values()):
        state.agreement_reached = True

    if state.round >= MAX_ROUNDS and not state.agreement_reached:
        state.agreement_failed = True
        print(f"\n❌ Max rounds reached without full agreement — negotiation collapsed.")

    # ---- 8. Diagnostics ----
    avg_trust = sum(c.trust for c in state.countries.values()) / len(state.countries)
    escalation = brinkmanship_risk(base_risk=0.1, escalation_level=state.round)
    if avg_trust < 0.25 and escalation > 0.5:
        print(f"  ⚠️  Brinkmanship risk: {escalation:.2f} | avg trust: {avg_trust:.2f}")

    payoff_list = [c.payoff for c in state.countries.values()]
    nb_score = 1.0
    for p in payoff_list:
        nb_score *= max(0.0, p)
    print(
        f"  📊 Round {state.round} | "
        f"Payoffs: { {k: round(v.payoff, 1) for k, v in state.countries.items()} } "
        f"| NB score: {nb_score:.1f}"
    )

    return state