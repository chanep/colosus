import logging

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, send, emit
from match_controller import MatchController

_client_sid = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cucamona'

socketio = SocketIO(app)
logging.basicConfig(level=logging.ERROR)


def on_status_update(status):
    print('server on_status_update')
    if _client_sid is not None:
        # socketio.emit('status_update', status, room=_client_sid)
        socketio.emit('status_update', status, room=_client_sid)


match_c = MatchController(on_status_update)


@socketio.on('connect')
def handle_connect():
    global _client_sid
    _client_sid = request.sid
    print('Client connected ' + str(_client_sid))


@socketio.on('disconnect')
def handle_disconnect():
    global _client_sid
    _client_sid = None
    print('Client disconnected')


@socketio.on('new_game')
def handle_new_game(data):
    print("new_game:" + str(data))
    match_c.new_match(data['blackHuman'], data['whiteHuman'], data['iterations'])


@socketio.on('move')
def handle_move(data):
    print("move:" + str(data))
    match_c.move(data['rank'], data['file'])


if __name__ == '__main__':
    socketio.run(app, port=5003, debug=False, log_output=False)
