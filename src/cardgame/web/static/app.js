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

    // Standalone name prompt (used for the server's "need_name" flow).
    function openNameModal() {
        document.getElementById("name-modal-input").value = getPlayerName();
        document.getElementById("name-modal").classList.add("open");
    }
    function closeNameModal() {
        pendingAction = null;
        document.getElementById("name-modal").classList.remove("open");
    }
    function submitName() {
        saveName(document.getElementById("name-modal-input").value);
    }

    // Settings modal (name + card back), opened from the nav menu.
    function openSettingsModal() {
        document.getElementById("settings-name-input").value = getPlayerName();
        applyCardBack();
        document.getElementById("settings-modal").classList.add("open");
    }
    function closeSettingsModal() {
        document.getElementById("settings-modal").classList.remove("open");
    }
    // The Settings modal's single "Done" button: save the name (if any) and
    // close. Card back is applied live on each pick, so it needs no saving.
    function saveSettings() {
        saveName(document.getElementById("settings-name-input").value);
        closeSettingsModal();
    }
    function saveName(raw) {
        const name = (raw || "").trim();
        if (!name) return;
        localStorage.setItem("player_name", name);
        socket.emit("set_name", withIdentity({ name: name }));
    }

    // Card-back appearance: a per-viewer cosmetic preference kept in
    // localStorage and applied as body data-attributes that style.css keys the
    // face-down patterns off. Nothing here touches game state.
    const CARD_BACK_COLORS = ["red", "green", "blue"];
    const CARD_BACK_PATTERNS = ["stripes", "crosshatch", "gradient", "plain"];
    function getCardBack() {
        const color = localStorage.getItem("card_back_color");
        const pattern = localStorage.getItem("card_back_pattern");
        return {
            color: CARD_BACK_COLORS.includes(color) ? color : "red",
            pattern: CARD_BACK_PATTERNS.includes(pattern) ? pattern : "stripes",
        };
    }
    function applyCardBack() {
        const back = getCardBack();
        document.body.dataset.backColor = back.color;
        document.body.dataset.backPattern = back.pattern;
        document.querySelectorAll("#back-grid .back-cell").forEach((b) =>
            b.classList.toggle("selected",
                b.dataset.color === back.color && b.dataset.pattern === back.pattern));
    }
    const backGridEl = document.getElementById("back-grid");
    if (backGridEl) {
        backGridEl.addEventListener("click", (event) => {
            const btn = event.target.closest(".back-cell");
            if (!btn) return;
            localStorage.setItem("card_back_color", btn.dataset.color);
            localStorage.setItem("card_back_pattern", btn.dataset.pattern);
            applyCardBack();
        });
    }
    applyCardBack();
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
    socket.on("name_set", () => {
        document.getElementById("name-modal").classList.remove("open");
        document.getElementById("settings-modal").classList.remove("open");
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

    // Only the first join of this page load is a genuine "I came here to play"
    // intent (the `start` flag driving the solo resume/new-game flow); later
    // joins are socket reconnects, which must just resume the game in progress.
    let firstJoin = true;
    socket.on("connect", () => {
        // Clear any "Connection lost" flash left over from a prior drop.
        const flash = document.getElementById("flash-container");
        if (flash) flash.innerHTML = "";
        // Solo games (see app.py's module docstring) never need a name to
        // auto-seat you - only multiplayer rooms do, and only when there's
        // actually a vacant seat to claim - so it's fine to always route
        // through requireNameThen() here: the server simply won't ask for
        // one in the cases that don't need it.
        if (ROOM_KEY || REVIEW_ENTRY_ID) {
            const extra = firstJoin ? { start: true } : {};
            firstJoin = false;
            requireNameThen("join", Object.assign(targetFields(), extra));
        }
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
    function placeMarker(row, col) {
        socket.emit("place_marker", withIdentity({ code: ROOM_KEY, row: row, col: col }));
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

    // After a full re-render the pointer may already sit over a freshly
    // created cell button, so the browser paints its :hover state with no
    // mouse movement - e.g. the cell you clicked to place the marker looks
    // pre-selected on your next turn. Suppress board hover until a real move.
    function suppressBoardHover() {
        const board = document.querySelector(".board");
        if (board) board.classList.add("no-hover");
    }
    document.addEventListener("mousemove", () => {
        const board = document.querySelector(".board");
        if (board) board.classList.remove("no-hover");
    });

    socket.on("state", (payload) => {
        const el = document.getElementById("game-state");
        if (el) el.innerHTML = payload.html;
        applyActiveHand();
        applyAnalysisTimer();
        suppressBoardHover();
        applySpectatorBadge();
    });

    // Surface the spectating indicator in the sticky nav (its label is carried
    // by a hidden element in the state fragment), so it never takes vertical
    // space above the board and disturb the fit-to-height scaling.
    function applySpectatorBadge() {
        const badge = document.getElementById("spectator-badge");
        if (!badge) return;
        const src = document.getElementById("spectator-status");
        if (src) {
            badge.textContent = src.dataset.label;
            badge.hidden = false;
        } else {
            badge.hidden = true;
        }
    }
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

    // Solo resume/new-game prompt. The server sends "solo_prompt" when a "start"
    // intent landed on a game in progress (see handle_join): show the current
    // game with a choice to Resume or start the new action. A bare payload means
    // Play (New game); one carrying a load_state means Play-from-here (Start from
    // this position). Both confirm by re-joining with force so the server starts
    // fresh; Resume just dismisses.
    function closeResumeModal() {
        document.getElementById("resume-modal").classList.remove("open");
    }
    function openResumeModal(newLabel, onNew) {
        const btn = document.getElementById("resume-modal-new");
        btn.textContent = newLabel;
        btn.onclick = () => {
            closeResumeModal();
            onNew();
        };
        document.getElementById("resume-modal").classList.add("open");
    }
    socket.on("solo_prompt", (payload) => {
        const loadState = payload && payload.load_state;
        if (loadState) {
            openResumeModal("Start from this position", () => {
                socket.emit("join", withIdentity({ code: ROOM_KEY, start: true, force: true, load_state: loadState }));
            });
        } else {
            openResumeModal("New game", () => {
                socket.emit("join", withIdentity({ code: ROOM_KEY, start: true, force: true }));
            });
        }
    });
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
