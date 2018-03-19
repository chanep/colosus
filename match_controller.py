from flask import request, jsonify
import numpy as np

from colosus.Illegal_move import IllegalMove
from colosus.config import MatchConfig
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.game.square import Square
from colosus.match import Match, PlayerSettings
from colosus.player_type import PlayerType


class MatchController:
    def __init__(self, config: MatchConfig, status_update_callback):
        self.config = config
        self._status_update_callback = status_update_callback
        self._match = Match(config)

    def new_match(self, black_human, white_human, iterations=None):
        if black_human:
            black = PlayerSettings(PlayerType.HUMAN)
        else:
            black = PlayerSettings(PlayerType.COLOSUS, iterations, self.config.weights_filename)
        if white_human:
            white = PlayerSettings(PlayerType.HUMAN)
        else:
            white = PlayerSettings(PlayerType.COLOSUS, iterations, self.config.weights_filename)

        self._match.new_game(black, white, initial_pos=None, move_callback=self._on_move,
                             match_initialized_callback=self._on_match_initialized)

    def move(self, rank, file):
        move = Square.square(rank, file)
        try:
            self._match.move(move)
        except IllegalMove as err:
            status = self._create_match_status(self._match, error=err.value)
            self._status_update_callback(status)

    def _on_move(self, match: Match, move=None, value=None, pv=None):
        status = self._create_match_status(match, move, value, pv)
        self._status_update_callback(status)

    def _on_match_initialized(self, match: Match):
        status = self._create_match_status(match)
        self._status_update_callback(status)

    def _create_match_status(self, match: Match, last_move=None, value=None, pv=None, error: str=None):
        winner = None
        depth = None
        win_line = []
        if match.is_end():
            winner = match.position.side_to_move.change()
            for (rank, file) in match.position.win_line():
                win_line.append({'rank': rank, 'file': file})
        if last_move is not None:
            last_move = self._move_to_dict(last_move)

        if pv is not None:
            depth = len(pv)

        return {
            'board': self._position_dto(match.position),
            'humanTurn': match.is_human_turn(),
            'lastMove': last_move,
            'winner': winner,
            'value': value,
            'depth': depth,
            'error': error,
            'sideToMove': match.position.side_to_move,
            'inProgress': match.in_progress,
            'winLine': win_line
        }

    def _move_to_dict(self, move: int):
        rank, file = Square.to_rank_file(move)
        return {'rank': rank, 'file': file}

    def _position_dto(self, position: Position):
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
