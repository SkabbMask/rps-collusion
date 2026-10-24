"""N-player iterated Rock Paper Scissors, generalising the three-player game of Bonjour et al. (UAI 2022, Section 5.1)."""

MOVES = ("rock", "paper", "scissors")

# BEATS[x] is the move that x defeats.
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}


def payoffs(actions):
    """Payoff for each player, following the rules behind Table 1 of the paper.

    - everyone plays the same action    -> everyone gets 0
    - rock, paper and scissors all occur -> everyone gets 1 (no winner in a cycle)
    - exactly two distinct actions       -> every player holding the winning action gets 1
    """
    distinct = set(actions)
    if len(distinct) == 1:
        return [0] * len(actions)
    if len(distinct) == 3:
        return [1] * len(actions)
    x, y = distinct
    winner = x if BEATS[x] == y else y
    return [1 if a == winner else 0 for a in actions]


def assist_move(primary_move):
    """The move an assistant plays to guarantee the primary a point regardless of the third player."""
    return BEATS[primary_move]
