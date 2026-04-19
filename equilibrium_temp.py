"""
Mathematical equilibrium and utility calculations.
NO LLM logic here.
"""

from typing import Tuple


def repeated_game_utility(
    base_payoff: float,
    discount_factor: float = 0.9,
    rounds: int = 5
) -> float:
    """
    Calculates discounted utility for repeated games.
    """
    utility = 0.0
    for t in range(rounds):
        utility += (discount_factor ** t) * base_payoff
    return utility


def nash_bargaining_solution(
    utility_a: float,
    utility_b: float,
    disagreement_a: float = 0.0,
    disagreement_b: float = 0.0
) -> float:
    """
    Nash Bargaining objective value.
    """
    return (utility_a - disagreement_a) * (utility_b - disagreement_b)


def is_deterrence_stable(
    expected_attack_gain: float,
    expected_attack_cost: float
) -> bool:
    """
    Deterrence condition:
    Peace is stable if cost > gain.
    """
    return expected_attack_cost > expected_attack_gain


def brinkmanship_risk(
    base_risk: float,
    escalation_level: int
) -> float:
    """
    Computes probability of conflict under brinkmanship.
    """
    return min(1.0, base_risk + (0.1 * escalation_level))


def coalition_utility(
    base_payoff: float,
    coalition_size: int,
    bonus: float
) -> float:
    """
    Utility adjustment for coalition formation.
    """
    return base_payoff + (coalition_size * bonus)