"""View layer for rooms: turns a RoomState (or a frozen finished entry) into
per-viewer template context, renders it, and pushes it to connected sockets.
Also builds the lobby listing summaries and the post-game move-analysis panel
(including its outcome heatmap).
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


# Sequential ramp for the heatmap cells: dark (0% of outcomes) to accent blue
# (100%). A single hue carries likelihood.
_HEATMAP_BASE_RGB = (0x2B, 0x2B, 0x2B)
_HEATMAP_ACCENT_RGB = (0x7C, 0xB8, 0xFF)


def _heatmap_style(pct):
    """Cell background interpolated along the ramp, plus a luminance-picked
    text color for the label so it stays legible at both ends."""
    t = min(1.0, max(0.0, pct / 100))
    r, g, b = (
        round(base + (accent - base) * t)
        for base, accent in zip(_HEATMAP_BASE_RGB, _HEATMAP_ACCENT_RGB)
    )
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    text = "#161616" if luminance > 140 else "#f2f2f2"
    return f"#{r:02x}{g:02x}{b:02x}", text


# Fixed heatmap axis, so positions can be compared at a glance. Outcomes
# beyond it collapse into the two end cells.
_HEATMAP_RANGE = 4


def _outcome_heatmap(distribution, mover_seat):
    """Turn a move's {mover_diff: weight} distribution into the heatmap's
    template context: a fixed row of cells from -_HEATMAP_RANGE to
    +_HEATMAP_RANGE (P1 - P2), two end cells pooling everything beyond, and
    the single most likely outcome as the headline text.
    """
    total = sum(distribution.values())
    # Flip from the mover's perspective onto the fixed P1 - P2 axis.
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
    being reviewed, or None when there's nothing to show.

    Returns {"pending": True, ...} while analysis hasn't reached a tractable
    position yet or is mid-flight; {"calculable": True, ...} once the grace
    period has passed and this position was never reached, so the template
    can offer a "Calculate" button (see start_ondemand_analysis).
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
        # A feasible position the worker hasn't reached is "coming soon", but
        # only while the grace period runs. Once it's over, fall through to the
        # "Calculate" button rather than spin forever.
        if analysis_feasible(display_game) and not grace_period_over:
            return {"pending": True, "mover_name": mover_name, "started_ago": None}
        if grace_period_over:
            return {"calculable": True, "mover_name": mover_name, "history_index": history_index}
        return None
    # A partial entry (see _stream_analysis) wraps the moves solved so far,
    # "streaming" True while calculating. A finished analysis is a plain
    # {(row, col): stats} dict. The rows below render either.
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
    # Sort by the same (eval, marker) tie-break analyse_moves uses for "best",
    # so the badged best move stays first in the table.
    for marker, move in sorted(
        computed.items(), key=lambda item: (item[1]["eval"], item[0]), reverse=True
    ):
        card = move["card"]
        # analyse_moves reports pcts from the mover's perspective; fix to P1/P2
        # so the columns mean the same on every position.
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
                # The played face-down move has a known identity on the final board.
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
    # For a partial entry, list the not-yet-solved moves too, tagged
    # "calculating" while running, "stopped" once a deadline cut it short.
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
    # The heatmap is only available once analysis fully finished; a partial
    # entry shows the table alone.
    best_distribution = None
    if not partial:
        best = next(row for row in rows if row["best"])
        best_distribution = computed[best["marker"]]["distribution"]
    # Every move solved but not final: the note flips to "generating heatmap".
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

    # Placement phase: marker dealt but unplaced, both seats filled so seat 2
    # can choose. your_placement marks the viewer with the clickable cells.
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

    # "Play from here": link to a fresh solo game from the reviewed position
    # (see create_solo_room_from_state). Only for a non-terminal position.
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
    """Build the template context for one viewer reviewing a frozen finished
    room (a finished_rooms entry), rather than the live RoomState _room_context
    reads. The Game never changes; only this viewer's scrubbed-to position
    (entry["history_index"]) does, so review reuses the live history flow.
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

    Each viewer's status bar/seat buttons depend on who they are, so this is
    rendered and sent individually per connected socket, not shared.
    """
    with room.lock:
        sid_players = dict(room.sid_players)
    for sid, player_id in sid_players.items():
        html = render_state(code, room, player_id)
        socketio.emit("state", {"html": html}, to=sid)


def room_summary_locked(code, room):
    """Like _room_summary, but assumes the caller holds room.lock, so a
    finished-game snapshot is atomic with the move that finished it (see
    game_flow.record_room_finished_locked).
    """
    game = room.game
    # Guard on the marker: an unplaced game has no legal moves but isn't over.
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
        # Drives the lobby's Join vs Spectate label; distinct from "waiting for
        # players" (which is about moves, not free seats).
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
    """Active games - rooms and solo games alike - for the /rooms "Ongoing
    games" table; the template distinguishes them via "is_solo". Solo games
    with no moves yet are left off (nothing to watch), but a multiplayer room
    with none still shows as "waiting for players" so a joiner can find it.
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
    """The most recently finished games, most-recent first - frozen snapshots
    taken as each one finished, so a later rematch doesn't change its entry.
    """
    with finished_rooms_lock:
        return list(reversed(finished_rooms))
