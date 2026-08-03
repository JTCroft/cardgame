"""Pure policy predicates for the post-game move-analysis subsystem.

These decide *whether* a position is worth analysing (and whether the
automatic worker's grace period has elapsed), with no side effects and no
dependency on room state, rendering, or the worker itself. They're shared
by both the rendering layer (deciding what to show a viewer) and the
analysis worker (deciding what to attempt), so they live in this leaf
module to keep those two from importing each other.
"""

import time

__all__ = ("analysis_feasible", "analysis_attemptable", "grace_period_over")


def analysis_feasible(game):
    """Whether a position is cheap enough that the review page should show
    "pending" rather than nothing at all while the worker hasn't reached it
    yet - calibrated against worst cases over sampled random games
    (recalibrated 2026-07 after the cached-scoring / single-pass-walk
    optimisations, which bought roughly 8-30x here). The worker itself
    (_analysis_worker_loop) uses the wider analysis_attemptable instead and
    isn't bound by this at all once backtracking past it, so this is purely
    about not promising a viewer "coming soon" for something that in
    practice never finishes. It walks the full remaining tree with no
    pruning, so it's only meaningful near the end of the game; as with
    Game.evaluate, the chance-node branching from face-down cards is the
    main cost driver.
    """
    # The next combinations out - (14, 5), (15, 4), (16, 4), (17, 3) -
    # were measured at 100-336s for a single position, past any sensible
    # budget; this ceiling sits exactly at that cliff edge.
    cards_left = 36 - len(game.moves)
    facedown = len(game.board.facedown_cards)
    if cards_left <= 9:
        return True
    if cards_left <= 10:
        return facedown <= 7
    if cards_left <= 11:
        return facedown <= 6
    if cards_left <= 13:
        return facedown <= 5
    if cards_left <= 14:
        return facedown <= 4
    if cards_left <= 16:
        return facedown <= 3
    return False


def analysis_attemptable(game):
    """The live worker's wider ceiling: during play there are minutes of
    thinking time rather than a post-game budget, so it may attempt one
    ring beyond analysis_feasible - the combinations measured in the
    ~2-6 minute range ((14,5): 104s, (15,4): 100s, (16,4): 336s,
    (17,3): 319s). Anything deeper runs into tens of minutes.
    """
    if analysis_feasible(game):
        return True
    cards_left = 36 - len(game.moves)
    facedown = len(game.board.facedown_cards)
    if cards_left <= 14:
        return facedown <= 5
    if cards_left <= 16:
        return facedown <= 4
    if cards_left <= 17:
        return facedown <= 3
    return False


def grace_period_over(deadline):
    """Whether the automatic post-game analysis worker's grace period (see
    game_flow._ANALYSIS_TIME_CAP/RoomState.analysis_deadline) has already
    ended - the gate for offering an explicit "Calculate" button in
    rendering._move_analysis_context, since before that the worker may still
    reach this position on its own. `deadline` is None only if the game
    somehow hasn't been recorded as finished yet, which shouldn't happen for
    a position review is even possible for; treated as "over" rather than
    wedging the button off forever on what would be a bug elsewhere.
    """
    return deadline is None or time.monotonic() >= deadline
