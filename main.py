from graph.negotiation_graph import build_negotiation_graph
from schemas.state import NegotiationState
from agents.llm import AGENT_MODELS
import json


def get_user_input():
    print("\n" + "="*60)
    print("  🌐 INTERNATIONAL NEGOTIATION SIMULATOR")
    print("     4 LLM agents | Game theory | Groq-powered")
    print("="*60)
    print("\nDescribe the negotiation scenario (natural language):")
    text = input("  > ").strip()
    from agents.input_parser import parse_natural_language_input
    return parse_natural_language_input(text)


def get_outcome(state: NegotiationState) -> dict:
    """
    Returns a structured outcome reflecting whether the negotiation
    succeeded, partially succeeded, or collapsed into defection.
    """
    history = state.history
    if not history:
        return {
            "result": "NO_ROUNDS",
            "label": "❌ No negotiation rounds occurred.",
            "detail": "The simulation did not run."
        }

    # Full agreement
    if state.agreement_reached:
        return {
            "result": "FULL_AGREEMENT",
            "label": "✅ Full agreement reached",
            "detail": "All four parties cooperated simultaneously — Nash equilibrium achieved."
        }

    # Explicit defection ending (hit max rounds without agreement)
    if state.agreement_failed:
        # Work out who the persistent defectors were
        defector_rounds = {}
        for entry in history:
            try:
                actions_part = entry.split(":")[1].split("|")[0]
                for pair in actions_part.split(","):
                    k, v = pair.strip().split("=")
                    if v.strip() == "D":
                        defector_rounds[k.strip()] = defector_rounds.get(k.strip(), 0) + 1
            except Exception:
                pass

        defector_names = [
            f"{state.countries[k].name} ({k}) — defected {n}x"
            for k, n in sorted(defector_rounds.items(), key=lambda x: -x[1])
        ]

        return {
            "result": "COLLAPSED",
            "label": "❌ Negotiation collapsed — no agreement reached",
            "detail": f"The simulation ran all {state.round} rounds without full cooperation.",
            "persistent_defectors": defector_names,
            "game_theory_note": (
                "This reflects a Prisoner's Dilemma trap: individually rational defection "
                "produced a collectively suboptimal outcome. "
                "Brinkmanship escalation or free-riding prevented stable cooperation."
            )
        }

    # Partial / majority convergence
    def majority_coop(round_str: str) -> bool:
        try:
            actions_part = round_str.split(":")[1].split("|")[0]
            pairs = [p.strip() for p in actions_part.split(",")]
            coop_count = sum(1 for p in pairs if p.split("=")[1].strip() == "C")
            return coop_count >= 3
        except Exception:
            return False

    recent = history[-3:]
    convergence = sum(1 for r in recent if majority_coop(r))
    if convergence >= 2:
        return {
            "result": "PARTIAL_AGREEMENT",
            "label": "⚠️ Partial agreement — majority cooperation sustained",
            "detail": "3 or more parties cooperated consistently but full consensus was not reached."
        }

    if state.active_coalitions:
        partners = " & ".join(
            f"{state.countries[a].name}+{state.countries[b].name}"
            for a, b in state.active_coalitions
        )
        return {
            "result": "COALITION_ONLY",
            "label": f"⚠️ Coalition formed but no full agreement: {partners}",
            "detail": "A bilateral alliance formed but other parties did not join."
        }

    return {
        "result": "COLLAPSED",
        "label": "❌ Negotiation collapsed — no agreement reached",
        "detail": "No stable cooperative equilibrium was reached within the round limit.",
        "game_theory_note": (
            "Defection dominated. This is consistent with a one-shot Prisoner's Dilemma "
            "outcome where trust never accumulated sufficiently for cooperation to take hold."
        )
    }


def main():
    print("DEBUG: main() started")

    state = get_user_input()

    print(f"\n📋 Scenario: {state.scenario}")
    print(f"🤖 Agents & Models:")
    for agent_id, country in state.countries.items():
        model = AGENT_MODELS.get(agent_id, "unknown")
        print(f"   {agent_id}: {country.name} — [{model}]")

    graph = build_negotiation_graph()
    result = graph.invoke(state)

    if isinstance(result, dict):
        final_state = NegotiationState(**result)
    else:
        final_state = result

    outcome = get_outcome(final_state)

    output = {
        "scenario": final_state.scenario,
        "outcome": outcome,                   # NEW: top-level outcome block
        "agents": {
            k: {
                "name": v.name,
                "model": AGENT_MODELS.get(k, "unknown"),
                "final_payoff": round(v.payoff, 2),
                "final_trust": round(v.trust, 2),
                "proposed_ally": v.proposed_ally,
            }
            for k, v in final_state.countries.items()
        },
        "game_theory_models_used": final_state.game_models,
        "negotiation_rounds": final_state.round,
        "active_coalitions": final_state.active_coalitions,
        "negotiation_history": final_state.history,
        "message_log": [
            {
                "round": m.round,
                "from": final_state.countries[m.sender].name,
                "to": final_state.countries[m.recipient].name,
                "message": m.content,
            }
            for m in final_state.message_log
        ],
        "agreement_reached": final_state.agreement_reached,
        "agreement_failed": final_state.agreement_failed,   # NEW
    }

    print("\n" + "="*60)
    print("  📜 FINAL REPORT")
    print("="*60)
    print(json.dumps(output, indent=2))

    print("\n" + "="*60)
    print(outcome["label"])
    if "detail" in outcome:
        print(outcome["detail"])
    if "persistent_defectors" in outcome:
        print("Persistent defectors:")
        for d in outcome["persistent_defectors"]:
            print(f"  • {d}")
    if "game_theory_note" in outcome:
        print(f"\n📖 {outcome['game_theory_note']}")
    print("="*60)

    with open("final_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n💾 Output saved to final_output.json")


if __name__ == "__main__":
    main()