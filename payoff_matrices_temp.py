"""
This file contains payoff matrices and constants
for different game theory models.

All values are normalized utilities.
"""

# Prisoner's Dilemma payoff matrix
# C = Cooperate, D = Defect
PRISONERS_DILEMMA = {
    ("C", "C"): (3, 3),
    ("C", "D"): (1, 4),
    ("D", "C"): (4, 1),
    ("D", "D"): (2, 2),
}

# Trade deal payoff template
TRADE_PAYOFF = {
    ("C", "C"): (4, 4),   # mutual free trade
    ("C", "D"): (1, 5),   # one-sided tariffs
    ("D", "C"): (5, 1),
    ("D", "D"): (2, 2),   # trade war
}

# Climate coalition bonus
COALITION_BONUS = 1.5

# Free rider penalty
FREE_RIDER_PENALTY = 1.0