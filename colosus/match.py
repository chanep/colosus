from concurrent.futures import ThreadPoolExecutor

from colosus.Illegal_move import IllegalMove
from colosus.config import PlayerConfig
from colosus.game.position import Position
from colosus.player import Player
from colosus.player_type import PlayerType


class PlayerSettings:
    def __init__(self, type: PlayerType, iterations: int=None, weights_filename: str=None):
        self.type = type
        self.iterations = iterations
        self.weights_filename = weights_filename


class Match:
    def __init__(self):
        self.position = None
        self.players = [None, None]
        self._move_callback = None
        self._match_initialized_callback = None
        self.initialized = False
        self._executor = ThreadPoolExecutor()

    def new_game(self, black: PlayerSettings, white: PlayerSettings, initial_pos: Position=None,
                 move_callback=None, match_initialized_callback=None):
        if initial_pos is None:
            self.position = Position()
        else:
            self.position = initial_pos

        self._move_callback = move_callback
        self._match_initialized_callback = match_initialized_callback

        player_settings = [black, white]

        self._executor.submit(self._initialize, player_settings)

    def _initialize(self, player_settings):
        for i in range(len(player_settings)):
            player_sets = player_settings[i]
            if player_sets.type == PlayerType.COLOSUS:
                self.players[i] = Player(PlayerConfig())
                self.players[i].new_game(self.position.clone(), player_sets.iterations, player_sets.weights_filename)
        self.initialized = True
        if self._match_initialized_callback is not None:
            self._match_initialized_callback(self)

    def _current_player(self) -> Player:
        return self.players[self.position.side_to_move]

    def _colosus_thinks_async(self):
        self._executor.submit(self._colosus_thinks)

    def _colosus_thinks(self):
        c_player = self._current_player()
        policy, value, move, old_state, new_state = c_player.move()
        self._move_done(move, value)

    def move(self, move):
        if not self.initialized or not self.is_human_turn():
            return
        if not self.position.is_legal(move):
            raise IllegalMove(move)
        self._move_done(move)

    def is_human_turn(self):
        side = self.position.side_to_move
        return self.players[side] is None

    def is_end(self):
        return self.position.is_end

    def _move_done(self, move, value=None):
        self.position = self.position.move(move)
        if self._move_callback is not None:
            self._move_callback(self, move=move, value=value)
        self._move_done(move, self.position.clone(), value)
        if not self.is_end() and not self.is_human_turn():
            self._colosus_thinks()





