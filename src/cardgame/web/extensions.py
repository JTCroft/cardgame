"""The shared Flask-SocketIO instance.

Split out from app.py (which does the actual `socketio.init_app(app)`) so
that `rooms.py` and `solo.py` can import `socketio` to emit events from
their own broadcast helpers without creating a circular import with app.py -
app.py is what constructs the Flask app and imports *from* those modules.
"""

from flask_socketio import SocketIO

socketio = SocketIO()
