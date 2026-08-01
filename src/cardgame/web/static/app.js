    // Identity lives entirely in this browser's localStorage rather than a
    // server-side session/cookie: a UUID generated once on first visit
    // (player_id), plus whatever display name the player has chosen
    // (player_name). Both are sent up with every relevant Socket.IO event
    // via withIdentity() below, which is how the server recognises "the
    // same visitor" across page loads, reconnects, and even different
    // rooms - see app.py's module docstring for the reasoning.
    function generateUuid() {
        if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
        // Fallback for browsers/contexts without crypto.randomUUID.
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
            const r = (Math.random() * 16) | 0;
            const v = c === "x" ? r : (r & 0x3) | 0x8;
            return v.toString(16);
        });
    }
    function getPlayerId() {
        let id = localStorage.getItem("player_id");
        if (!id) {
            id = generateUuid();
            localStorage.setItem("player_id", id);
        }
        return id;
    }
    function getPlayerName() {
        return localStorage.getItem("player_name") || "";
    }
    const PLAYER_ID = getPlayerId();
    function withIdentity(data) {
        return Object.assign({ player_id: PLAYER_ID, name: getPlayerName() }, data || {});
    }

    const socket = io();
    // lastAction remembers the most recent requireNameThen() call so that,
    // *if* the server comes back asking for a name, we know what to retry
    // once one's been given - see the "need_name" handler below, which is
    // what actually promotes it to pendingAction. Only promoted attempts
    // get retried on name_set; an attempt that didn't need a name (e.g.
    // you already have one, or there was no seat to claim anyway) must
    // NOT get silently replayed the next time you happen to change your
    // name via the settings button.
    let lastAction = null;
    let pendingAction = null;

    document.getElementById("current-name-label").textContent = getPlayerName() || "Set name";

    function toggleNavMenu() {
        document.getElementById("nav-menu").classList.toggle("open");
    }
    function closeNavMenu() {
        document.getElementById("nav-menu").classList.remove("open");
    }
    document.addEventListener("click", (event) => {
        const menu = document.getElementById("nav-menu");
        if (!menu.classList.contains("open")) return;
        if (menu.contains(event.target) || event.target.closest(".hamburger")) return;
        closeNavMenu();
    });

    function openNameModal() {
        document.getElementById("name-modal-input").value = getPlayerName();
        document.getElementById("name-modal").classList.add("open");
    }
    function closeNameModal() {
        pendingAction = null;
        document.getElementById("name-modal").classList.remove("open");
    }
    function submitName() {
        const input = document.getElementById("name-modal-input");
        const name = input.value.trim();
        if (!name) return;
        localStorage.setItem("player_name", name);
        socket.emit("set_name", withIdentity({ name: name }));
    }
    // Fires an event that might need a display name first - if the server
    // responds with "need_name", this attempt (event + its *original*,
    // identity-free data) is what gets retried once one's set. `data`
    // must not already have name/player_id baked in, or a retry would
    // replay today's (possibly still-empty) name instead of picking up
    // the fresh one - see withIdentity() above, always applied fresh both
    // times.
    function requireNameThen(event, data) {
        lastAction = { event: event, data: data };
        socket.emit(event, withIdentity(data));
    }

    socket.on("need_name", () => {
        pendingAction = lastAction;
        openNameModal();
    });
    socket.on("name_set", (payload) => {
        document.getElementById("name-modal").classList.remove("open");
        const label = document.getElementById("current-name-label");
        if (label) label.textContent = payload.name;
        if (pendingAction) {
            const action = pendingAction;
            pendingAction = null;
            socket.emit(action.event, withIdentity(action.data));
        }
    });
    socket.on("error_message", (payload) => {
        const el = document.getElementById("flash-container");
        if (el) el.innerHTML = "<div class=\"flash\">" + payload.message + "</div>";
    });

    // Shared by any page with a #game-state div - currently room.html.jinja2
    // (live rooms) and play.html.jinja2 (solo play/spectate/review). Inert
    // on pages without one, like index.html.jinja2 or rooms.html.jinja2.
    //
    // ROOM_KEY names which room this page is about - either a 4-letter
    // multiplayer room code or another (or your own) player's id naming a
    // solo-vs-computer room; REVIEW_ENTRY_ID names a frozen, finished game
    // instead, for read-only review. A room and a solo game are the same
    // thing server-side (see app.py's module docstring), so one set of
    // functions below covers both - room.html.jinja2/play.html.jinja2 just
    // assign these two before they're used (safe: the "connect" event is
    // always async, firing only after all synchronous <script> execution,
    // including their own scripts block, has already run).
    let ROOM_KEY = null;
    let REVIEW_ENTRY_ID = null;
    // A "Play from here" saved position (play.html.jinja2, /play-from route).
    // Sent up with the *first* join to seed a fresh solo game, then cleared
    // so a later reconnect's re-join doesn't reset the game back to it.
    let LOAD_STATE = null;

    // The routing fields alone, with no identity attached yet - passed to
    // requireNameThen() for "join" so a retry (after a name prompt) can
    // attach a *fresh* identity rather than replaying a stale, still-empty
    // one baked in at the first attempt.
    function targetFields() {
        const fields = { code: ROOM_KEY };
        if (REVIEW_ENTRY_ID) fields.review_entry_id = REVIEW_ENTRY_ID;
        if (LOAD_STATE) fields.load_state = LOAD_STATE;
        return fields;
    }
    function withTarget(data) {
        return withIdentity(Object.assign(targetFields(), data));
    }

    socket.on("connect", () => {
        // Solo games (see app.py's module docstring) never need a name to
        // auto-seat you - only multiplayer rooms do, and only when there's
        // actually a vacant seat to claim - so it's fine to always route
        // through requireNameThen() here: the server simply won't ask for
        // one in the cases that don't need it.
        if (ROOM_KEY || REVIEW_ENTRY_ID) requireNameThen("join", targetFields());
        // The seed only applies to the first join; drop it (and scrub it from
        // the address bar) so a reconnect - or a refresh - resumes the game
        // in progress instead of resetting it back to the loaded position.
        if (LOAD_STATE) {
            LOAD_STATE = null;
            history.replaceState({}, "", "/play");
        }
    });

    function sendMove(row, col) {
        socket.emit("move", withIdentity({ code: ROOM_KEY, row: row, col: col }));
    }
    function requestRematch() {
        socket.emit("request_rematch", withIdentity({ code: ROOM_KEY }));
    }
    function respondRematch(accept) {
        socket.emit("respond_rematch", withIdentity({ code: ROOM_KEY, accept: accept }));
    }
    // Swapping seats is meaningful both in a multiplayer room and in a solo
    // game vs. the computer (it swaps which seat the computer's in too) -
    // see rooms.RoomState.swap_seats - so unlike claim/leave/kick (room.html.jinja2
    // only), this lives here where both room.html.jinja2 and play.html.jinja2 can use it.
    function swapSeats() {
        socket.emit("swap_seats", withIdentity({ code: ROOM_KEY }));
    }
    function historyStep(delta) {
        socket.emit("history_step", withTarget({ delta: delta }));
    }
    function historyGoto(index) {
        socket.emit("history_goto", withTarget({ index: index }));
    }
    // "Calculate" button in the move-analysis panel, for a position the
    // automatic post-game worker gave up on (see
    // rooms.start_ondemand_analysis) - the server pushes a fresh "state"
    // straight back once it's done (no polling needed here), the same way
    // it replies to history_step/history_goto above.
    function calculateMove(index) {
        socket.emit("calculate_move", withTarget({ index: index }));
    }

    // Ticks the move-analysis panel's "still being computed (Ns so far)"
    // counter between server pushes, which only arrive when the position
    // changes or a calculation finishes - not once a second - so without
    // this the elapsed time would only ever update on those occasions.
    // Re-anchored from the server-rendered data-started-ago baseline every
    // time #game-state is replaced (see the "state" handler below), so it
    // stays accurate to the server's own clock rather than drifting.
    let analysisTimerInterval = null;
    function applyAnalysisTimer() {
        if (analysisTimerInterval) {
            clearInterval(analysisTimerInterval);
            analysisTimerInterval = null;
        }
        const el = document.querySelector(".analysis-timer");
        if (!el) return;
        const startedAt = Date.now() - parseFloat(el.dataset.startedAgo) * 1000;
        const format = (seconds) => {
            seconds = Math.max(0, Math.round(seconds));
            const m = Math.floor(seconds / 60);
            const s = seconds % 60;
            return m > 0 ? m + "m " + String(s).padStart(2, "0") + "s" : s + "s";
        };
        analysisTimerInterval = setInterval(() => {
            const live = document.querySelector(".analysis-timer");
            if (!live) {
                clearInterval(analysisTimerInterval);
                analysisTimerInterval = null;
                return;
            }
            live.textContent = format((Date.now() - startedAt) / 1000);
        }, 1000);
    }

    socket.on("state", (payload) => {
        const el = document.getElementById("game-state");
        if (el) el.innerHTML = payload.html;
        applyActiveHand();
        applyAnalysisTimer();
    });
    socket.on("disconnect", () => {
        const el = document.getElementById("flash-container");
        if (el) el.innerHTML = "<div class=\"flash\">Connection lost, trying to reconnect...</div>";
    });
    function openGameOverModal() {
        const el = document.getElementById("game-over-modal");
        if (el) el.classList.add("open");
    }
    function closeGameOverModal() {
        const el = document.getElementById("game-over-modal");
        if (el) el.classList.remove("open");
    }
    // Mobile hand panel: exactly one player's hand is shown at a time, as a
    // persistent toggle rather than a drawer that auto-closes - selection
    // lives only here in JS, so it has to be re-applied every time a
    // "state" push replaces #game-state's innerHTML wholesale (including
    // right after the very first render, defaulting to seat 1).
    let activeHandSeat = 1;
    function selectHand(seat) {
        activeHandSeat = seat;
        applyActiveHand();
    }
    function applyActiveHand() {
        for (const seat of [1, 2]) {
            const slot = document.getElementById("hand-slot-" + seat);
            const tab = document.getElementById("hand-tab-" + seat);
            const active = seat === activeHandSeat;
            if (slot) slot.classList.toggle("open", active);
            if (tab) tab.classList.toggle("active", active);
        }
    }
