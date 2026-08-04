"""Flask + Flask-SocketIO front end for playing Cross Kings online.

Two players share a game at /room/<4 letter code>; visiting auto-claims a
vacant seat, later visitors spectate. /play is the same room model keyed by
the visitor's own player id, with the computer seated in the other seat. A
solo game is just a room whose second seat is computer-played. /rooms lists
live rooms; finished games are reviewable from a frozen snapshot.

There's no server-side session: each browser generates a player id (UUID)
and display name in localStorage, sent up with every event. The initial
render is a generic shell; the real per-viewer board is filled in over the
socket once the page's JS supplies its identity.

State is in-memory and per-process. This module owns the HTTP routes and
Socket.IO handlers; supporting concerns are split into siblings: rooms.py
(state + lifecycle helpers + identity validation), rendering.py (per-viewer
render/broadcast + view models), analysis_worker.py (post-game analysis),
analysis_policy.py (feasibility/grace-period predicates), game_flow.py
(computer turn + freezing finished games), extensions.py (shared socketio).
"""

import importlib.metadata
import os
import random
import secrets

from flask import (
    Flask,
    Response,
    abort,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_socketio import join_room as sio_join_room
from flask_socketio import leave_room as sio_leave_room
from jinja2 import ChoiceLoader, PackageLoader

from ..analysis_native import NATIVE_AVAILABLE as _ANALYSIS_NATIVE
from ..cards import Card
from ..game import Game
from ..solver_native import NATIVE_AVAILABLE as _SOLVER_NATIVE
from .analysis_policy import grace_period_over
from .analysis_worker import ensure_analysis_worker, start_ondemand_analysis
from .extensions import app_holder, socketio
from .game_flow import maybe_play_computer_move, record_room_finished_locked
from .rendering import (
    broadcast_state,
    lobby_active_summaries,
    lobby_finished_summaries,
    render_room_review_state,
    render_state,
)
from .rooms import (
    MAX_NAME_LENGTH,
    create_solo_room_from_state,
    deal_fresh_locked,
    decode_game_state,
    find_finished_room_entry,
    finished_rooms_lock,
    get_or_create_room,
    normalize_code,
    normalize_player_id,
    random_unused_code,
    resolve_or_create_room_for_join,
    resolve_room_key,
    rooms,
    sid_index,
    sid_index_lock,
    status_counts,
)

# flask-socketio broadcast group for the /rooms lobby listing (one shared
# render for every viewer). Distinct from our 4-letter game room codes.
_LOBBY_GROUP = "lobby"


def _render_lobby():
    return render_template(
        "_lobby_state.html.jinja2",
        rooms=lobby_active_summaries(),
        finished_games=lobby_finished_summaries(),
    )


def _how_to_play_context():
    """Data for the illustrated rules page: a dealt board plus, for each
    diagram, the marker cell, highlighted legal moves and taken cells (gaps).
    The scoring diagrams are hand-built card rows."""
    game = Game.deal()
    board = game.board

    # A reachable position for the "passing over gaps" diagram: the marker sits
    # in a gap with further gaps past it. Any facedown resolution will do.
    gap_game = game
    for move in [(2, 3), (4, 3), (4, 1), (1, 1), (1, 3)]:
        gap_game = gap_game.move(*move)[0]

    cards = lambda strs: [Card.from_str(s) for s in strs]
    return {
        "board": board,
        "central_cells": {(2, 2), (2, 3), (3, 2), (3, 3)},
        "move_marker": Game.starting_position,
        "move_legal": game.legal_moves,
        "gap_board": gap_game.board,
        "gap_marker": gap_game.marker,
        "gap_taken": set(gap_game.moves),
        "gap_legal": gap_game.legal_moves,
        "run_cards": cards(["4C", "5C", "6C"]),
        "set_cards": cards(["7H", "7D", "7C"]),
        "king_cards": cards(["5H", "6H", "KS"]),
        "points_table": [(1, 0), (2, 0), (3, 3), (4, 5), (5, 7), (6, 9), (7, 11), (8, 13)],
    }


def _broadcast_lobby():
    socketio.emit("lobby_state", {"html": _render_lobby()}, to=_LOBBY_GROUP)


def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("CARDGAME_SECRET_KEY", secrets.token_hex(16))

    # Chain in the package-wide templates/ folder (the card/board macros
    # shared with the Jupyter _repr_html_ output) alongside Flask's default.
    app.jinja_loader = ChoiceLoader(
        [
            app.jinja_loader,
            PackageLoader(package_name="cardgame", package_path="../../templates"),
        ]
    )

    @app.context_processor
    def inject_app_version():
        return {"app_version": importlib.metadata.version("cardgame")}

    @app.get("/")
    def index():
        return render_template("index.html.jinja2")

    @app.get("/how-to-play")
    def how_to_play():
        return render_template("how_to_play.html.jinja2", **_how_to_play_context())

    @app.get("/favicon.ico")
    def favicon():
        return redirect(url_for("static", filename="favicon.ico"))

    @app.get("/status")
    def status():
        # Health/diagnostics JSON. `native` reports whether the Rust core
        # backs the solver (computer opponent) and post-game analysis.
        return {
            "status": "ok",
            "version": importlib.metadata.version("cardgame"),
            "native": {
                "solver": _SOLVER_NATIVE,
                "analysis": _ANALYSIS_NATIVE,
            },
            **status_counts(),
        }

    @app.get("/robots.txt")
    def robots_txt():
        # Only the two landing pages are indexable; games/reviews/rooms are
        # transient or per-user.
        body = (
            "User-agent: *\n"
            "Allow: /$\n"
            "Allow: /rooms$\n"
            "Disallow: /play\n"
            "Disallow: /review/\n"
            "Disallow: /room/\n"
            "\n"
            "Sitemap: https://crosskings.net/sitemap.xml\n"
        )
        return Response(body, mimetype="text/plain")

    @app.get("/sitemap.xml")
    def sitemap_xml():
        urls = [url_for("index", _external=True), url_for("rooms_view", _external=True)]
        entries = "".join(f"  <url><loc>{u}</loc></url>\n" for u in urls)
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            f"{entries}"
            "</urlset>\n"
        )
        return Response(body, mimetype="application/xml")

    @app.get("/rooms")
    def rooms_view():
        return render_template(
            "rooms.html.jinja2",
            rooms=lobby_active_summaries(),
            finished_games=lobby_finished_summaries(),
        )

    @app.post("/create-room")
    def create_room():
        code = random_unused_code()
        return redirect(url_for("room_view", code=code))

    @app.post("/join-room")
    def join_room_form():
        code = normalize_code(request.form.get("code", ""))
        if code is None:
            return redirect(url_for("index"))
        return redirect(url_for("room_view", code=code))

    @app.get("/play")
    def play():
        # No player id server-side yet; the page's JS joins over the socket.
        return render_template("play.html.jinja2")

    @app.get("/play-from/<state>")
    def play_from(state):
        # "Play from here": seed a solo game from a URL-encoded position.
        # Validated eagerly here; the JS creates the room on its "join".
        game = decode_game_state(state)
        if game is None or not game.legal_moves:
            abort(404, description="That saved position could not be loaded.")
        return render_template("play.html.jinja2", load_state=state)

    @app.get("/play/<player_id>")
    def spectate_solo(player_id):
        target_id = normalize_player_id(player_id)
        if target_id is None:
            abort(404, description="Invalid player id.")
        return render_template("play.html.jinja2", spectate_player_id=target_id)

    @app.get("/review/<entry_id>")
    def review(entry_id):
        # Dispatch on the frozen entry's is_solo flag (solo vs room template).
        entry = find_finished_room_entry(entry_id)
        if entry is None:
            abort(404, description="That finished game could no longer be found.")
        if entry["is_solo"]:
            return render_template("play.html.jinja2", review_entry_id=entry_id)
        return render_template("room.html.jinja2", code=entry["code"], review_entry_id=entry_id)

    @app.get("/room/<code>")
    def room_view(code):
        normalized = normalize_code(code)
        if normalized is None:
            abort(404, description="Room codes must be 4 letters, e.g. /room/ABCD")
        get_or_create_room(normalized)
        return render_template("room.html.jinja2", code=normalized)

    @app.errorhandler(404)
    def not_found(error):
        message = getattr(error, "description", None) or "That page doesn't exist."
        return render_template("error.html.jinja2", code=404, message=message), 404

    @app.errorhandler(500)
    def server_error(error):
        message = "Something went wrong on our end. Please try again."
        return render_template("error.html.jinja2", code=500, message=message), 500

    return app


app = create_app()
socketio.init_app(app, async_mode="threading")
# Background threads in rooms.py render templates; they need the app to
# enter an app context (see extensions.app_holder).
app_holder["app"] = app


@socketio.on("join")
def handle_join(data):
    data = data or {}
    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        socketio.emit("error_message", {"message": "Missing player id."}, to=request.sid)
        return

    # review_entry_id names a frozen finished room for read-only review - a
    # stateless path with no live RoomState to join.
    review_entry_id = (data.get("review_entry_id") or "").strip()
    if review_entry_id:
        entry = find_finished_room_entry(review_entry_id)
        if entry is None:
            socketio.emit(
                "error_message",
                {"message": "That finished game could no longer be found."},
                to=request.sid,
            )
            return
        socketio.emit(
            "state", {"html": render_room_review_state(entry, player_id)}, to=request.sid
        )
        return

    name = (data.get("name") or "").strip()[:MAX_NAME_LENGTH]
    # Solo "start a new game" flow. Client flags: `start` (first join of a
    # fresh page load), `force` (confirmed the new-game prompt), `load_state`
    # (a "Play from here" seed position). A game in progress gets a Resume/new
    # choice (solo_prompt) rather than being clobbered.
    start = bool(data.get("start"))
    force = bool(data.get("force"))
    load_state = (data.get("load_state") or "").strip()
    seed_game = None
    if load_state:
        seed_game = decode_game_state(load_state)
        if seed_game is None or not seed_game.legal_moves:
            socketio.emit(
                "error_message",
                {"message": "That saved position could not be loaded."},
                to=request.sid,
            )
            return

    raw_code = data.get("code", "")
    is_own_solo = normalize_code(raw_code) is None and (
        normalize_player_id(raw_code) or player_id
    ) == player_id

    pending_prompt = None  # {load_state} -> emit solo_prompt after broadcasting
    deal_fresh = False
    if is_own_solo:
        existing = rooms.get(player_id)
        in_progress = finished = False
        if existing is not None and existing.computer_seat is not None:
            with existing.lock:
                finished = existing.game_over
                in_progress = existing.game_started and not existing.game_over
        if force and seed_game is not None:
            code, room = create_solo_room_from_state(player_id, seed_game)
        elif force:
            code, room = player_id, get_or_create_room(player_id, solo=True)
            deal_fresh = True
        elif start and in_progress:
            # Don't clobber a game in progress - resume it and ask (below).
            code, room = player_id, existing
            pending_prompt = {"load_state": load_state or None}
        elif start and finished and seed_game is None:
            # Landing on a finished game via Play -> straight to a new one.
            code, room = player_id, get_or_create_room(player_id, solo=True)
            deal_fresh = True
        elif seed_game is not None:
            code, room = create_solo_room_from_state(player_id, seed_game)
        else:
            code, room = player_id, get_or_create_room(player_id, solo=True)
    else:
        code, room, error = resolve_or_create_room_for_join(raw_code, player_id)
        if room is None:
            socketio.emit("error_message", {"message": error}, to=request.sid)
            return

    sid = request.sid
    with room.lock:
        if deal_fresh:
            deal_fresh_locked(room)
        room.sid_players[sid] = player_id
        # Pick up the player's client-stored name, if any.
        if name:
            room.player_names[player_id] = name

        # Auto-claim a vacant seat. A multiplayer seat needs a name first
        # (needs_name prompts and retries); a solo room's seat is exempt.
        vacant_seat = next(
            (seat for seat in (1, 2) if not room.occupied(seat)), None
        )
        eligible = (
            vacant_seat is not None
            and room.seat_of(player_id) is None
            and not (room.game_started and not room.game_over)
        )
        can_auto_seat = name or room.computer_seat is not None
        needs_name = eligible and not can_auto_seat
        if eligible and can_auto_seat:
            room.seats[vacant_seat] = player_id
    with sid_index_lock:
        sid_index[sid] = (code, player_id)

    if needs_name:
        # Withhold the board behind the name prompt; the retry renders it.
        socketio.emit("need_name", {}, to=sid)
    else:
        broadcast_state(code, room)
        # Offer the joiner Resume vs new-game for a game in progress.
        if pending_prompt is not None:
            socketio.emit("solo_prompt", pending_prompt, to=sid)
    # A solo computer in seat 2 places the marker once the human is seated.
    maybe_play_computer_move(code, room)
    ensure_analysis_worker(code, room)
    _broadcast_lobby()


@socketio.on("set_name")
def handle_set_name(data):
    data = data or {}
    name = (data.get("name") or "").strip()[:MAX_NAME_LENGTH]
    if not name:
        socketio.emit("error_message", {"message": "Please enter a name."}, to=request.sid)
        return
    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        socketio.emit("error_message", {"message": "Missing player id."}, to=request.sid)
        return

    sid = request.sid
    with sid_index_lock:
        info = sid_index.get(sid)
    if info is not None:
        code, _player_id = info
        room = rooms.get(code)
        if room is not None:
            with room.lock:
                room.player_names[player_id] = name
                is_seated = room.seat_of(player_id) is not None
            # Only broadcast for a seated player; an unseated visitor's name
            # isn't visible to others yet.
            if is_seated:
                broadcast_state(code, room)
                _broadcast_lobby()

    socketio.emit("name_set", {"name": name}, to=sid)


@socketio.on("claim_seat")
def handle_claim_seat(data):
    data = data or {}
    code = normalize_code(data.get("code", ""))
    if code is None:
        return
    room = rooms.get(code)
    if room is None:
        return

    name = (data.get("name") or "").strip()
    if not name:
        socketio.emit("need_name", {}, to=request.sid)
        return

    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    try:
        seat = int(data.get("seat"))
    except (TypeError, ValueError):
        return

    ok, error = room.claim_seat(seat, player_id)
    if not ok:
        socketio.emit("error_message", {"message": error}, to=request.sid)
        return

    broadcast_state(code, room)
    # Claiming may have completed a solo room's seating, letting the computer
    # place the marker.
    maybe_play_computer_move(code, room)
    _broadcast_lobby()


@socketio.on("vacate_seat")
def handle_vacate_seat(data):
    data = data or {}
    code = normalize_code(data.get("code", ""))
    if code is None:
        return
    room = rooms.get(code)
    if room is None:
        return

    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    ok, error = room.vacate_seat(player_id)
    if not ok:
        socketio.emit("error_message", {"message": error}, to=request.sid)
        return

    broadcast_state(code, room)
    _broadcast_lobby()


@socketio.on("swap_seats")
def handle_swap_seats(data):
    data = data or {}
    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    # Works for solo rooms too (keyed by player id, not a 4-letter code).
    code = resolve_room_key(data.get("code", ""), player_id)
    room = rooms.get(code)
    if room is None:
        return

    ok, error = room.swap_seats(player_id)
    if not ok:
        socketio.emit("error_message", {"message": error}, to=request.sid)
        return

    broadcast_state(code, room)
    _broadcast_lobby()
    # The computer may now hold seat 1 (moves first), which no played move
    # would otherwise trigger.
    maybe_play_computer_move(code, room)


@socketio.on("kick_seat")
def handle_kick_seat(data):
    data = data or {}
    code = normalize_code(data.get("code", ""))
    if code is None:
        return
    room = rooms.get(code)
    if room is None:
        return

    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    try:
        seat = int(data.get("seat"))
    except (TypeError, ValueError):
        return

    ok, error = room.kick_seat(seat)
    if not ok:
        socketio.emit("error_message", {"message": error}, to=request.sid)
        return

    broadcast_state(code, room)
    _broadcast_lobby()


@socketio.on("move")
def handle_move(data):
    data = data or {}
    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        socketio.emit("error_message", {"message": "Missing player id."}, to=request.sid)
        return
    code = resolve_room_key(data.get("code", ""), player_id)
    room = rooms.get(code)
    if room is None:
        socketio.emit("error_message", {"message": "That room doesn't exist."}, to=request.sid)
        return
    my_seat = room.seat_of(player_id)

    with room.lock:
        game = room.game
        if not room.both_seated:
            socketio.emit(
                "error_message",
                {"message": "Waiting for both players to join before the game can start."},
                to=request.sid,
            )
            return
        if my_seat is None or my_seat != room.current_turn_seat:
            socketio.emit("error_message", {"message": "It isn't your turn."}, to=request.sid)
            return
        try:
            row = int(data["row"])
            col = int(data["col"])
        except (KeyError, TypeError, ValueError):
            socketio.emit("error_message", {"message": "Invalid move."}, to=request.sid)
            return
        if (row, col) not in game.legal_moves:
            socketio.emit("error_message", {"message": "That isn't a legal move."}, to=request.sid)
            return

        outcomes = game.move(row, col)
        # Moving onto a face-down card reveals a uniformly-random unseen card
        # (as in Game.random_move).
        room.game = random.choice(outcomes)
        if room.game_over:
            record_room_finished_locked(code, room)

    broadcast_state(code, room)
    _broadcast_lobby()
    maybe_play_computer_move(code, room)
    ensure_analysis_worker(code, room)
    _broadcast_lobby()


@socketio.on("place_marker")
def handle_place_marker(data):
    data = data or {}
    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        socketio.emit("error_message", {"message": "Missing player id."}, to=request.sid)
        return
    code = resolve_room_key(data.get("code", ""), player_id)
    room = rooms.get(code)
    if room is None:
        socketio.emit("error_message", {"message": "That room doesn't exist."}, to=request.sid)
        return
    my_seat = room.seat_of(player_id)

    with room.lock:
        game = room.game
        if not room.both_seated:
            socketio.emit(
                "error_message",
                {"message": "Waiting for both players to join before the game can start."},
                to=request.sid,
            )
            return
        if not game.needs_marker:
            socketio.emit("error_message", {"message": "The marker is already placed."}, to=request.sid)
            return
        # The marker is placed by the player not moving first - seat 2.
        if my_seat != 2:
            socketio.emit("error_message", {"message": "It isn't yours to place."}, to=request.sid)
            return
        try:
            row = int(data["row"])
            col = int(data["col"])
        except (KeyError, TypeError, ValueError):
            socketio.emit("error_message", {"message": "Invalid placement."}, to=request.sid)
            return
        if (row, col) not in game._valid_starting_positions:
            socketio.emit(
                "error_message",
                {"message": "The marker must start on a central card."},
                to=request.sid,
            )
            return
        room.game = game.place_marker(row, col)

    broadcast_state(code, room)
    _broadcast_lobby()
    # If the computer is Player 1, it now makes the first move.
    maybe_play_computer_move(code, room)


@socketio.on("history_step")
def handle_history_step(data):
    data = data or {}
    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    try:
        delta = int(data.get("delta", 0))
    except (TypeError, ValueError):
        return

    review_entry_id = (data.get("review_entry_id") or "").strip()
    if review_entry_id:
        entry = find_finished_room_entry(review_entry_id)
        if entry is None:
            socketio.emit(
                "error_message",
                {"message": "That finished game could no longer be found."},
                to=request.sid,
            )
            return
        with finished_rooms_lock:
            total = len(entry["game"].moves)
            current = entry["history_index"].get(player_id, total)
            entry["history_index"][player_id] = max(0, min(total, current + delta))
        socketio.emit(
            "state", {"html": render_room_review_state(entry, player_id)}, to=request.sid
        )
        return

    code = resolve_room_key(data.get("code", ""), player_id)
    room = rooms.get(code)
    if room is None or not room.game_over:
        return

    with room.lock:
        total = len(room.game.moves)
        current = room.history_index.get(player_id, total)
        room.history_index[player_id] = max(0, min(total, current + delta))

    socketio.emit("state", {"html": render_state(code, room, player_id)}, to=request.sid)


@socketio.on("history_goto")
def handle_history_goto(data):
    data = data or {}
    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError):
        return

    review_entry_id = (data.get("review_entry_id") or "").strip()
    if review_entry_id:
        entry = find_finished_room_entry(review_entry_id)
        if entry is None:
            socketio.emit(
                "error_message",
                {"message": "That finished game could no longer be found."},
                to=request.sid,
            )
            return
        with finished_rooms_lock:
            total = len(entry["game"].moves)
            entry["history_index"][player_id] = max(0, min(total, index))
        socketio.emit(
            "state", {"html": render_room_review_state(entry, player_id)}, to=request.sid
        )
        return

    code = resolve_room_key(data.get("code", ""), player_id)
    room = rooms.get(code)
    if room is None or not room.game_over:
        return

    with room.lock:
        total = len(room.game.moves)
        room.history_index[player_id] = max(0, min(total, index))

    socketio.emit("state", {"html": render_state(code, room, player_id)}, to=request.sid)


@socketio.on("calculate_move")
def handle_calculate_move(data):
    """Explicit "Calculate" click for a position the automatic post-game
    worker gave up on. Always re-renders, so a redundant click while it's
    already running still confirms it's under way.
    """
    data = data or {}
    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError):
        return
    sid = request.sid

    review_entry_id = (data.get("review_entry_id") or "").strip()
    if review_entry_id:
        entry = find_finished_room_entry(review_entry_id)
        if entry is None:
            return
        if not grace_period_over(entry.get("analysis_deadline")):
            return
        start_ondemand_analysis(
            entry["analysis"],
            entry["analysis_inflight"],
            entry["analysis_calc_started"],
            finished_rooms_lock,
            entry["game"],
            index,
            on_done=lambda: socketio.emit(
                "state", {"html": render_room_review_state(entry, player_id)}, to=sid
            ),
        )
        socketio.emit("state", {"html": render_room_review_state(entry, player_id)}, to=sid)
        return

    code = resolve_room_key(data.get("code", ""), player_id)
    room = rooms.get(code)
    if room is None:
        return
    with room.lock:
        if not room.game_over or not grace_period_over(room.analysis_deadline):
            return
        game, game_id = room.game, room.game_id
        cache, inflight, calc_started = room.analysis, room.analysis_inflight, room.analysis_calc_started

    def on_done():
        # Skip if the room has since rematched; the result waits on /review.
        current = rooms.get(code)
        if current is not None and current.game_id == game_id:
            broadcast_state(code, current)

    start_ondemand_analysis(cache, inflight, calc_started, room.lock, game, index, on_done=on_done)
    broadcast_state(code, room)


@socketio.on("request_rematch")
def handle_request_rematch(data):
    data = data or {}
    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    code = resolve_room_key(data.get("code", ""), player_id)
    room = rooms.get(code)
    if room is None:
        return

    my_seat = room.seat_of(player_id)
    if my_seat is None:
        socketio.emit(
            "error_message", {"message": "Only players can request a rematch."}, to=request.sid
        )
        return

    with room.lock:
        if not room.game_over:
            socketio.emit(
                "error_message", {"message": "The game hasn't finished yet."}, to=request.sid
            )
            return
        if room.computer_seat is not None:
            # The computer always accepts immediately.
            deal_fresh_locked(room)
        elif room.rematch_requested_by is not None and room.rematch_requested_by != player_id:
            # The other player already asked - treat this as accepting.
            deal_fresh_locked(room)
        else:
            room.rematch_requested_by = player_id

    broadcast_state(code, room)
    _broadcast_lobby()
    # A fresh solo rematch may put the computer on the move.
    maybe_play_computer_move(code, room)


@socketio.on("respond_rematch")
def handle_respond_rematch(data):
    data = data or {}
    player_id = normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    code = resolve_room_key(data.get("code", ""), player_id)
    room = rooms.get(code)
    if room is None:
        return

    my_seat = room.seat_of(player_id)
    accept = bool(data.get("accept"))

    with room.lock:
        if my_seat is None:
            return
        if room.rematch_requested_by is None or room.rematch_requested_by == player_id:
            return  # nothing to respond to, or you're the one who asked
        if accept:
            deal_fresh_locked(room)
        room.rematch_requested_by = None

    broadcast_state(code, room)
    _broadcast_lobby()


@socketio.on("join_lobby")
def handle_join_lobby():
    sio_join_room(_LOBBY_GROUP)
    socketio.emit("lobby_state", {"html": _render_lobby()}, to=request.sid)


@socketio.on("leave_lobby")
def handle_leave_lobby():
    sio_leave_room(_LOBBY_GROUP)


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    with sid_index_lock:
        info = sid_index.pop(sid, None)
    if info is not None:
        code, _player_id = info
        room = rooms.get(code)
        if room is not None:
            with room.lock:
                room.sid_players.pop(sid, None)
            # Refresh so watchers get the "Kick" button for the abandoned seat.
            broadcast_state(code, room)
            _broadcast_lobby()


def main():
    """Entry point for running a local dev server: `cardgame-web`."""
    host = os.environ.get("CARDGAME_HOST", "127.0.0.1")
    port = int(os.environ.get("CARDGAME_PORT", "5000"))
    debug = os.environ.get("CARDGAME_DEBUG", "").lower() in ("1", "true", "yes")
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
