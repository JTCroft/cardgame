"""Client-supplied visitor identity.

There's no server-side session in this app - see app.py's module docstring
for why. Every visitor generates their own player id (a UUID) and picks a
display name client-side (localStorage), sending both up with relevant
Socket.IO events; this module just validates/normalizes what arrives.
"""

import re

# Client-generated ids are UUIDs (see getPlayerId() in base.html.jinja2),
# but any short id-shaped token is accepted rather than validating strictly
# against RFC 4122 - the format only matters here to keep it a safe, bounded
# dict key.
_PLAYER_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_MAX_NAME_LENGTH = 24


def _normalize_player_id(raw):
    """Return a validated client-supplied player id, or None if missing/malformed."""
    if not raw or not isinstance(raw, str) or not _PLAYER_ID_RE.match(raw):
        return None
    return raw
