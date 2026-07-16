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
    // Solo-vs-computer pages set this (play.html.jinja2): whether to ask
    // the server for live-eval mode on join. null (every other page) means
    // "send no flag at all", leaving the room's mode untouched.
    let LIVE_EVAL = null;

    // The routing fields alone, with no identity attached yet - passed to
    // requireNameThen() for "join" so a retry (after a name prompt) can
    // attach a *fresh* identity rather than replaying a stale, still-empty
    // one baked in at the first attempt.
    function targetFields() {
        const fields = { code: ROOM_KEY };
        if (REVIEW_ENTRY_ID) fields.review_entry_id = REVIEW_ENTRY_ID;
        if (LIVE_EVAL !== null) fields.live_eval = LIVE_EVAL;
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
    function historyStep(delta) {
        socket.emit("history_step", withTarget({ delta: delta }));
    }
    function historyGoto(index) {
        socket.emit("history_goto", withTarget({ index: index }));
    }

    socket.on("state", (payload) => {
        const el = document.getElementById("game-state");
        if (el) el.innerHTML = payload.html;
        applyActiveHand();
    });
    // Live-eval pushes update their own container (only present on
    // /play/live pages), independent of #game-state re-renders.
    socket.on("live_eval", (payload) => {
        const el = document.getElementById("live-eval");
        if (el) el.innerHTML = payload.html;
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
