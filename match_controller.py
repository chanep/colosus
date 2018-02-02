from flask import request, jsonify
import numpy as np

from colosus.game.position import Position
from colosus.game.side import Side
from colosus.match import Match, PlayerSettings
from colosus.player_type import PlayerType

_match = Match()
_weights_filename = None


def new_match():
    iterations = request.json['iterations']
    black_human = request.json['black_human']
    white_human = request.json['white_human']
    if black_human:
        black = PlayerSettings(PlayerType.HUMAN)
    else:
        black = PlayerSettings(PlayerType.COLOSUS, iterations, _weights_filename)
    if white_human:
        white = PlayerSettings(PlayerType.HUMAN)
    else:
        white = PlayerSettings(PlayerType.COLOSUS, iterations, _weights_filename)

    _match.new_game(black, white, initial_pos=None, move_callback=_move_callback, match_initialized_callback=_match_initialized_callback)

    return "match initializing!"

def move(rank, file):



def _move_callback(match: Match, move=None, value=None):
    pass


def _match_initialized_callback(match: Match):
    pass


def _create_match_status(match: Match, last_move=None, value=None, error: str=None):
    winner = None
    if match.is_end():
        winner = match.position.side_to_move.change()

    return {
        'position': _position_dto(match.position),
        'human_turn': match.is_human_turn(),
        'last_move': last_move,
        'winner': winner,
        'value': value,
        'error': error
    }


def _position_dto(position: Position):
    p_str = ['X', 'O']
    ranks = []
    for r in range(Position.B_SIZE):
        rank = []
        for f in range(Position.B_SIZE):
            if position.piece_at(Side.WHITE, r, f):
                rank.append(p_str[Side.WHITE])
            elif position.piece_at(Side.BLACK, r, f):
                rank.append(p_str[Side.BLACK])
            else:
                rank.append('-')
        ranks.append(rank)
    return ranks
