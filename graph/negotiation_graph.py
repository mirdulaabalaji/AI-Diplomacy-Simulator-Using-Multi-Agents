from langgraph.graph import StateGraph, END
from schemas.state import NegotiationState
from agents.game_classifier import classify_game
from agents.country_agent import run_messaging_phase, country_decide
from agents.evaluator import evaluate_round

MAX_ROUNDS = 5


def build_negotiation_graph():
    graph = StateGraph(NegotiationState)

    graph.add_node("classify_game", classify_game)

    def messaging_step(state: NegotiationState) -> NegotiationState:
        if state.round >= MAX_ROUNDS:
            return state

        agent_names = {k: v.name for k, v in state.countries.items()}
        print(f"\n{'='*60}")
        print(f"  📬 MESSAGING PHASE (before Round {state.round + 1})")
        print(f"  Agents: {agent_names}")
        print(f"{'='*60}")

        state = run_messaging_phase(state)
        return state

    graph.add_node("messaging", messaging_step)

    def negotiation_step(state: NegotiationState) -> NegotiationState:
        if state.round >= MAX_ROUNDS:
            return state

        agent_ids = list(state.countries.keys())
        all_opponents = {
            agent_id: [k for k in agent_ids if k != agent_id]
            for agent_id in agent_ids
        }

        print(f"\n{'='*60}")
        print(f"  ⚖️  DECISION PHASE — Round {state.round + 1}")
        print(f"{'='*60}")

        actions = {}
        proposed_allies = {}

        for agent_id in agent_ids:
            action, ally = country_decide(
                state=state,
                country_id=agent_id,
                all_opponent_ids=all_opponents[agent_id],
            )
            actions[agent_id] = action
            proposed_allies[agent_id] = ally
            state.countries[agent_id].proposed_ally = ally

        state = evaluate_round(
            state=state,
            actions=actions,
            proposed_allies=proposed_allies,
        )

        return state

    graph.add_node("negotiate", negotiation_step)

    graph.set_entry_point("classify_game")
    graph.add_edge("classify_game", "messaging")
    graph.add_edge("messaging", "negotiate")

    def should_continue(state: NegotiationState):
        if state.agreement_reached:
            print("\n✅ Agreement reached — all parties cooperating!")
            return END
        if state.round >= MAX_ROUNDS:
            print(f"\n⏱️  Max rounds ({MAX_ROUNDS}) reached.")
            return END
        return "messaging"

    graph.add_conditional_edges("negotiate", should_continue)

    return graph.compile()