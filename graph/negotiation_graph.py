from langgraph.graph import StateGraph, END
from schemas.state import NegotiationState
from agents.game_classifier import classify_game
from agents.country_agent import country_decide
from agents.evaluator import evaluate_round

MAX_ROUNDS = 5


def build_negotiation_graph():
    graph = StateGraph(NegotiationState)

    # -------- NODE: classify game --------
    graph.add_node("classify_game", classify_game)

    # -------- NODE: negotiation step --------
    def negotiation_step(state: NegotiationState) -> NegotiationState:
        # Stop hard if max rounds reached
        if state.round >= MAX_ROUNDS:
            return state

        action_a = country_decide(state, "A", "B")
        action_b = country_decide(state, "B", "A")

        state = evaluate_round(
            state=state,
            action_a=action_a,
            action_b=action_b,
            country_a="A",
            country_b="B",
        )

        return state

    graph.add_node("negotiate", negotiation_step)

    # -------- FLOW --------
    graph.set_entry_point("classify_game")
    graph.add_edge("classify_game", "negotiate")

    # -------- TERMINATION LOGIC --------
    def should_continue(state: NegotiationState):
        """
        End if:
        - Mutual cooperation achieved (agreement_reached)
        - OR max rounds reached
        """
        if state.agreement_reached:
            return END

        if state.round >= MAX_ROUNDS:
            return END

        return "negotiate"

    graph.add_conditional_edges(
        "negotiate",
        should_continue
    )

    return graph.compile()