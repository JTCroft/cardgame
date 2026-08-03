"""View layer for rooms: turns a RoomState (or a frozen finished entry)
into the per-viewer template context, renders it, and pushes it out to
connected sockets. Also builds the lobby listing summaries and the
post-game move-analysis panel (including its outcome heatmap).

Depends only on the room-state model (rooms) and the pure analysis
policy predicates (analysis_policy) - never on the analysis worker, which
depends on *this* module (for broadcast_state) instead.
"""

import time
from collections import Counter

from flask import render_template

from ..game import Game
from .analysis_policy import analysis_feasible, grace_period_over
from .extensions import socketio
from .rooms import (
    encode_game_state,
    finished_rooms,
    finished_rooms_lock,
    rooms,
    rooms_lock,
)

__all__ = (
    "render_state",
    "render_room_review_state",
    "broadcast_state",
    "room_summary_locked",
    "lobby_active_summaries",
    "lobby_finished_summaries",
)


# Sequential ramp for the outcome heatmap's cells: dark (0% of outcomes,
# matching .move-analysis's own background so an empty cell reads as
# "nothing here") up to this site's existing accent blue (100%). A single
# hue carries likelihood; which side of the axis a cell sits on is what
# carries who it favours, so the two are never conflated in one channel.
_HEATMAP_BASE_RGB = (0x2B, 0x2B, 0x2B)
_HEATMAP_ACCENT_RGB = (0x7C, 0xB8, 0xFF)


def _heatmap_style(pct):
    """The cell's background - interpolated along the sequential ramp - and
    a text color for the percentage label printed on top of it, picked by
    the background's luminance so the label stays legible at both ends."""
    t = min(1.0, max(0.0, pct / 100))
    r, g, b = (
        round(base + (accent - base) * t)
        for base, accent in zip(_HEATMAP_BASE_RGB, _HEATMAP_ACCENT_RGB)
    )
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    text = "#161616" if luminance > 140 else "#f2f2f2"
    return f"#{r:02x}{g:02x}{b:02x}", text


# The heatmap's axis is fixed rather than sized to each position's actual
# range, so positions can be compared at a glance instead of each drawing
# its own scale. Outcomes beyond it collapse into the two end cells.
_HEATMAP_RANGE = 4


def _outcome_heatmap(distribution, mover_seat):
    """Turn a move's {mover_diff: weight} distribution (see analyse_moves)
    into the "who's winning" heatmap's template context: a fixed row of
    cells from -_HEATMAP_RANGE to +_HEATMAP_RANGE (P1 - P2), each annotated
    with its likelihood, plus two end cells pooling everything beyond that
    range - and the single most likely outcome, called out as the headline
    text. Face-down cards mean even optimal play from a fixed position can
    end in a spread of scores, not one number, which is what this is for.
    """
    total = sum(distribution.values())
    # distribution is in the mover's own perspective (self - opponent);
    # flip it onto the fixed P1 - P2 axis the template renders.
    sign = 1 if mover_seat == 1 else -1
    p1_distribution = Counter({sign * diff: weight for diff, weight in distribution.items()})

    def cell(label, pct, zero=False):
        color, text_color = _heatmap_style(pct)
        return {"label": label, "pct": pct, "zero": zero, "color": color, "text_color": text_color}

    low_pct = sum(w for d, w in p1_distribution.items() if d <= -_HEATMAP_RANGE - 1) / total * 100
    high_pct = sum(w for d, w in p1_distribution.items() if d >= _HEATMAP_RANGE + 1) / total * 100
    cells = [cell(f"≤-{_HEATMAP_RANGE + 1}", low_pct)]
    for diff in range(-_HEATMAP_RANGE, _HEATMAP_RANGE + 1):
        pct = p1_distribution.get(diff, 0) / total * 100
        cells.append(cell(f"{diff:+d}" if diff else "0", pct, zero=(diff == 0)))
    cells.append(cell(f"≥+{_HEATMAP_RANGE + 1}", high_pct))

    mode_diff, mode_weight = max(p1_distribution.items(), key=lambda item: (item[1], -abs(item[0])))
    return {
        "cells": cells,
        "mode_diff": mode_diff,
        "mode_pct": mode_weight / total * 100,
    }


def _move_analysis_context(cache, full_game, display_game, history_index, p1_name, p2_name,
                           inflight=frozenset(), calc_started=None, grace_period_over=False):
    """Build the move-comparison panel's template context for the position
    currently being reviewed, or None when there's nothing to show at all
    (final position, or no analysis recorded and the position is one the
    automatic worker hasn't given up on yet). Returns {"pending": True, ...}
    while the background computation hasn't reached a tractable position
    yet - or is mid-flight on this one - so the template can say it's on
    its way; {"calculable": True, ...} once the automatic post-game grace
    period (_ANALYSIS_TIME_CAP) has passed and this position was never
    reached - too deep for the worker's own unattended budget, but still
    computable on explicit request (see start_ondemand_analysis) - so the
    template can offer a "Calculate" button instead of showing nothing.
    """
    if cache is None or history_index >= len(full_game.moves):
        return None
    mover_seat = 1 if history_index % 2 == 0 else 2
    mover_name = (p1_name if mover_seat == 1 else p2_name) or f"Player {mover_seat}"
    opponent_name = (p2_name if mover_seat == 1 else p1_name) or f"Player {3 - mover_seat}"
    p1_display = p1_name or "Player 1"
    p2_display = p2_name or "Player 2"
    data = cache.get(history_index)
    if data is None:
        if history_index in inflight:
            started_ago = None
            if calc_started is not None and history_index in calc_started:
                started_ago = time.monotonic() - calc_started[history_index]
            return {"pending": True, "mover_name": mover_name, "started_ago": started_ago}
        # A feasible position the worker hasn't reached yet is "coming soon" -
        # but only while the grace period is still running. Once it's over the
        # worker has stopped for good, so a position it never got to (e.g. the
        # forward slot was starved of a pool thread - see _get_analysis_pools)
        # would otherwise be stuck showing "still being computed" forever with
        # no way to trigger it. Fall through to the "Calculate" button instead,
        # exactly as an infeasible position does.
        if analysis_feasible(display_game) and not grace_period_over:
            return {"pending": True, "mover_name": mover_name, "started_ago": None}
        if grace_period_over:
            return {"calculable": True, "mover_name": mover_name, "history_index": history_index}
        return None
    # A partial entry (see _stream_analysis) carries the moves solved so far
    # under a wrapper - "streaming" True while still calculating, False once a
    # deadline stopped it short - with the winner's heatmap not yet computed.
    # A finished analysis is a plain {(row, col): stats} dict (no "streaming"
    # key, since its keys are all (row, col) tuples). The rows below render
    # any of the three; only the heatmap, the note, and the still-uncomputed
    # placeholder rows differ.
    partial = isinstance(data, dict) and "streaming" in data
    in_progress = bool(partial and data["streaming"])
    done_count = total_count = None
    if partial:
        done_count, total_count = data["done"], data["total"]
        computed = data["moves"]
    else:
        computed = data
    played_marker = full_game.moves[history_index]
    rows = []
    # Same (eval, marker) tie-break analyse_moves itself uses to decide
    # "best" - not `combined`, which is a mean-based delta that can tie (or
    # even disagree in sign) between moves with different win/draw shapes.
    # Sorting by it too keeps the badged best move first in the table.
    for marker, move in sorted(
        computed.items(), key=lambda item: (item[1]["eval"], item[0]), reverse=True
    ):
        card = move["card"]
        # analyse_moves reports win_pct/loss_pct/defensive/offensive from
        # the mover's own perspective, which alternates with mover_seat as
        # you step through history - fixed to P1/P2 here so the columns
        # mean the same thing on every position instead of swapping sides
        # each time the mover changes (see _outcome_heatmap for the same
        # fix already applied to the heatmap's axis).
        if mover_seat == 1:
            p1_win_pct, p2_win_pct = move["win_pct"], move["loss_pct"]
            p1_change, p2_change = move["defensive"], move["offensive"]
        else:
            p1_win_pct, p2_win_pct = move["loss_pct"], move["win_pct"]
            p1_change, p2_change = move["offensive"], move["defensive"]
        rows.append(
            {
                "marker": marker,
                "card": card,
                # A face-down move's analysis averages over what it might
                # have been, but the one actually played has a known
                # outcome - the final board holds its revealed identity.
                "revealed": (
                    full_game.board[marker[0]][marker[1]]
                    if card.facedown and marker == played_marker
                    else None
                ),
                "combined": move["combined"],
                "p1_win_pct": p1_win_pct,
                "p2_win_pct": p2_win_pct,
                "draw_pct": move["draw_pct"],
                "p1_change": p1_change,
                "p2_change": p2_change,
                "best": move["best"],
                "played": marker == played_marker,
                "status": None,  # a solved row - see placeholders below
            }
        )
    # For a partial entry, list the moves not yet solved too, so the table
    # shows the full slate from the start rather than growing a row at a time -
    # tagged "calculating" while the walk is still running, "stopped" once a
    # deadline cut it short. display_game is the position being reviewed, so
    # its legal moves are exactly the ones the analysis covers.
    if partial:
        status = "calculating" if in_progress else "stopped"
        for marker in sorted(set(display_game.legal_moves) - set(computed)):
            card = display_game.board[marker[0]][marker[1]]
            rows.append(
                {
                    "marker": marker,
                    "card": card,
                    "revealed": (
                        full_game.board[marker[0]][marker[1]]
                        if card.facedown and marker == played_marker
                        else None
                    ),
                    "best": False,
                    "played": marker == played_marker,
                    "status": status,
                }
            )
    # The winner's distribution (and so the heatmap) is only there once the
    # analysis has fully finished - a partial entry shows the table alone.
    best_distribution = None
    if not partial:
        best = next(row for row in rows if row["best"])
        best_distribution = computed[best["marker"]]["distribution"]
    # Once every move is solved the walk is on its final, separate step - the
    # winner's outcome distribution (see iter_move_analyses / the native
    # `distribution` call), the one expensive piece - so the note flips from
    # "solving moves" to "generating heatmap" for that window.
    generating_heatmap = in_progress and done_count == total_count
    return {
        "pending": False,
        "streaming": in_progress and not generating_heatmap,
        "generating_heatmap": generating_heatmap,
        "incomplete": partial and not in_progress,
        "done_count": done_count,
        "total_count": total_count,
        "mover_name": mover_name,
        "opponent_name": opponent_name,
        "p1_name": p1_display,
        "p2_name": p2_display,
        "rows": rows,
        "heatmap": _outcome_heatmap(best_distribution, mover_seat) if best_distribution else None,
    }


def _room_context(code, room, player_id):
    """Build the (viewer-specific) template context for a room's state."""
    final_game = room.game
    game_over = room.game_over
    game_started = room.game_started
    my_seat = room.seat_of(player_id)

    display_game, history_index, history_total = room.displayed_game(player_id)
    viewing_history = game_over and history_index != history_total

    turn_seat = None if game_over else room.current_turn_seat

    # Placement phase: the marker is dealt but not yet placed, and both seats
    # are filled so seat 2 can choose its starting cell. your_placement marks
    # the viewer who gets the clickable central cells.
    needs_marker = display_game.needs_marker and room.both_seated
    your_placement = needs_marker and my_seat == 2

    winner = None
    if game_over:
        p1_score, p2_score = final_game.p1.score(), final_game.p2.score()
        if p1_score > p2_score:
            winner = 1
        elif p2_score > p1_score:
            winner = 2
        else:
            winner = 0  # draw

    seats_locked = game_started and not game_over

    show_game_over_modal = False
    if game_over and player_id not in room.game_over_seen:
        show_game_over_modal = True
        room.game_over_seen.add(player_id)

    rematch_requested_by_name = None
    rematch_requested_by_me = False
    if room.rematch_requested_by is not None:
        rematch_requested_by_me = room.rematch_requested_by == player_id
        rematch_requested_by_name = room.name_for_seat(room.seat_of(room.rematch_requested_by))

    move_analysis = None
    if game_over:
        move_analysis = _move_analysis_context(
            room.analysis,
            final_game,
            display_game,
            history_index,
            room.name_for_seat(1),
            room.name_for_seat(2),
            inflight=room.analysis_inflight,
            calc_started=room.analysis_calc_started,
            grace_period_over=grace_period_over(room.analysis_deadline),
        )

    # "Play from here" - a link to open a fresh solo game from the position
    # currently under review (see create_solo_room_from_state). Only offered
    # for a non-terminal reviewed position; the final one has nothing to play.
    play_from_state = (
        encode_game_state(display_game)
        if game_over and display_game.legal_moves
        else None
    )

    return {
        "code": code,
        "vs_computer": room.computer_seat is not None,
        "game": display_game,
        "my_seat": my_seat,
        "my_name": room.player_names.get(player_id),
        "is_spectator": my_seat is None,
        "both_seated": room.both_seated,
        "your_turn": my_seat is not None and my_seat == turn_seat and room.both_seated,
        "legal_moves": set() if (game_over or viewing_history) else display_game.legal_moves,
        "needs_marker": needs_marker,
        "your_placement": your_placement,
        "starting_positions": Game._valid_starting_positions,
        "game_over": game_over,
        "game_started": game_started,
        "seats_locked": seats_locked,
        "winner": winner,
        "p1_name": room.name_for_seat(1),
        "p2_name": room.name_for_seat(2),
        "p1_seated": room.occupied(1),
        "p2_seated": room.occupied(2),
        "p1_disconnected": room.seat_disconnected(1),
        "p2_disconnected": room.seat_disconnected(2),
        "p1_score": final_game.p1.score(),
        "p2_score": final_game.p2.score(),
        "seats_taken": len(room.seats) + (1 if room.computer_seat else 0),
        "taken_card": display_game.taken_card if display_game.moves else None,
        "history_index": history_index,
        "history_total": history_total,
        "viewing_history": viewing_history,
        "show_game_over_modal": show_game_over_modal,
        "rematch_pending": room.rematch_requested_by is not None,
        "rematch_requested_by_me": rematch_requested_by_me,
        "rematch_requested_by_name": rematch_requested_by_name,
        "move_analysis": move_analysis,
        "play_from_state": play_from_state,
    }


def _room_review_context(entry, viewer_id):
    """Build the template context for one viewer reviewing a frozen,
    finished room (a finished_rooms entry) - as opposed to _room_context,
    which reads the live, possibly-since-rematched RoomState. The Game
    object itself never changes here; only which move *this viewer* is
    currently scrubbed to does, tracked the same way as everywhere else -
    a per-viewer position in entry["history_index"] - so review reuses the
    exact same join/history_step/history_goto socket flow as live play and
    spectating, just pointed at a frozen entry instead of a live room.
    """
    game = entry["game"]
    total = len(game.moves)
    with finished_rooms_lock:
        history_index = entry["history_index"].get(viewer_id, total)
    history_index = max(0, min(total, history_index))
    display_game = game if history_index == total else game.undo(total - history_index)

    p1_score, p2_score = entry["p1_score"], entry["p2_score"]
    if p1_score > p2_score:
        winner = 1
    elif p2_score > p1_score:
        winner = 2
    else:
        winner = 0  # draw

    return {
        "code": entry["code"],
        "vs_computer": entry["is_solo"],
        "review": True,
        "game": display_game,
        "my_seat": None,
        "my_name": None,
        "is_spectator": True,
        "both_seated": True,
        "your_turn": False,
        "legal_moves": set(),
        "needs_marker": False,
        "your_placement": False,
        "starting_positions": (),
        "game_over": True,
        "game_started": True,
        "seats_locked": True,
        "winner": winner,
        "p1_name": entry["p1_name"],
        "p2_name": entry["p2_name"],
        "p1_seated": True,
        "p2_seated": True,
        "p1_disconnected": False,
        "p2_disconnected": False,
        "p1_score": p1_score,
        "p2_score": p2_score,
        "seats_taken": 2,
        "taken_card": display_game.taken_card if display_game.moves else None,
        "history_index": history_index,
        "history_total": total,
        "viewing_history": history_index != total,
        "show_game_over_modal": False,
        "rematch_pending": False,
        "rematch_requested_by_me": False,
        "rematch_requested_by_name": None,
        "entry_id": entry["id"],
        "move_analysis": _move_analysis_context(
            entry.get("analysis"),
            game,
            display_game,
            history_index,
            entry["p1_name"],
            entry["p2_name"],
            inflight=entry.get("analysis_inflight", frozenset()),
            calc_started=entry.get("analysis_calc_started"),
            grace_period_over=grace_period_over(entry.get("analysis_deadline")),
        ),
        # See _room_context - a link to play on from the reviewed position.
        "play_from_state": (
            encode_game_state(display_game) if display_game.legal_moves else None
        ),
    }


def render_room_review_state(entry, viewer_id):
    context = _room_review_context(entry, viewer_id)
    return render_template("_game_state.html.jinja2", **context)


def render_state(code, room, player_id):
    context = _room_context(code, room, player_id)
    return render_template("_game_state.html.jinja2", **context)


def broadcast_state(code, room):
    """Push a freshly rendered, viewer-specific board to everyone in the room.

    Each viewer's status bar/seat buttons depend on who *they* are (their
    own seat, whether they can join a vacant one, etc.), so unlike the lobby
    listing below this can't be sent as one shared broadcast - it's rendered
    and sent individually per connected socket.
    """
    with room.lock:
        sid_players = dict(room.sid_players)
    for sid, player_id in sid_players.items():
        html = render_state(code, room, player_id)
        socketio.emit("state", {"html": html}, to=sid)


def room_summary_locked(code, room):
    """Like _room_summary, but assumes the caller already holds room.lock
    (needed so a finished-game snapshot can be taken atomically with the
    move that just finished it - see game_flow.record_room_finished_locked).
    """
    game = room.game
    # An unplaced game (fresh room awaiting a marker) has no legal moves but
    # isn't finished - otherwise a room with a vacant seat would be filtered
    # out of the lobby as "finished" (see lobby_active_summaries).
    game_over = not game.needs_marker and not game.legal_moves
    game_started = room.game_started
    seated_ids = set(room.seats.values())
    spectators = sum(1 for pid in room.sid_players.values() if pid not in seated_ids)
    connected = len(room.sid_players) > 0
    if game_over:
        status = "finished"
    elif game_started:
        status = "in progress"
    else:
        status = "waiting for players"
    return {
        "code": code,
        "is_solo": room.computer_seat is not None,
        # A multiplayer room with a free seat is joinable; once both are taken
        # (even before the first move, while placing the marker) it's spectate
        # only. Drives the lobby's Join vs Spectate label - "waiting for
        # players" (no moves yet) is NOT the same as "has a free seat".
        "has_open_seat": room.computer_seat is None and len(room.seats) < 2,
        "has_moves": game_started,
        "p1_name": room.name_for_seat(1),
        "p2_name": room.name_for_seat(2),
        "status": status,
        "spectators": spectators,
        "connected": connected,
        "p1_score": game.p1.score() if (game_started or game_over) else None,
        "p2_score": game.p2.score() if (game_started or game_over) else None,
    }


def _room_summary(code, room):
    with room.lock:
        return room_summary_locked(code, room)


def lobby_active_summaries():
    """Active games - multiplayer rooms and solo-vs-computer games alike -
    for the /rooms list's "Ongoing games" table. A room/solo game are the
    same underlying thing (see module docstring), so one list covers both;
    the template distinguishes them via each summary's "is_solo" flag
    (e.g. to link to room_view vs spectate_solo). Solo games with no moves
    yet are left off - there's nothing meaningful to watch, and every
    /play visit otherwise creates one of these; a multiplayer room with no
    moves yet still shows (as "waiting for players"), since that's exactly
    what a second player needs to see to find it.
    """
    with rooms_lock:
        codes = list(rooms.keys())
    summaries = []
    for code in codes:
        room = rooms.get(code)
        if room is None:
            continue
        summary = _room_summary(code, room)
        if summary["status"] == "finished":
            continue
        if summary["is_solo"] and not summary["has_moves"]:
            continue
        summaries.append(summary)
    status_order = {"waiting for players": 0, "in progress": 1}
    summaries.sort(key=lambda r: (status_order[r["status"]], r["code"]))
    return summaries


def lobby_finished_summaries():
    """The most recently finished games - rooms and solo games alike, most-
    recent first - frozen snapshots taken as each one finished (see
    game_flow.record_room_finished_locked), so a later rematch/new game
    doesn't change or remove its entry here.
    """
    with finished_rooms_lock:
        return list(reversed(finished_rooms))
