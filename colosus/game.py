from concurrent.futures import ThreadPoolExecutor

from colosus.config import PlayerConfig
from colosus.game.position import Position
from colosus.player import Player
from colosus.player_type import PlayerType

class PlayerSettings:
    def __init__(self, type: PlayerType, iterations: int=None, weights_filename: str=None):
        self.type = type
        self.iterations = iterations
        self.weights_filename = weights_filename


class Game:
    def __init__(self):
        self.position = None
        self.players = [None, None]
        self._move_callback = None
        self.initialized = False
        self._executor = ThreadPoolExecutor()

    def new_game(self, player1: PlayerSettings, player2: PlayerSettings, initial_pos: Position=None, move_callback=None):
        if initial_pos is None:
            self.position = Position()
        else:
            self.position = initial_pos

        self._move_callback = move_callback

        player_settings = [player1, player2]

        self._executor.submit(self._initialize, player_settings)

    def _initialize(self, player_settings):
        for i in range(len(player_settings)):
            player_sets = player_settings[i]
            if player_sets.type == PlayerType.COLOSUS:
                self.players[i] = Player(PlayerConfig())
                self.players[i].new_game(self.position.clone(), player_sets.iterations, player_sets.weights_filename)
        self.initialized = True

    def _current_player(self) -> Player:
        return self.players[self.position.side_to_move]

    def _colosus_thinks_async(self):
        colosus_move_future = self._executor.submit(self._colosus_thinks)

    def _colosus_thinks(self):
        c_player = self._current_player()
        policy, value, move, old_state, new_state = c_player.move()

    def move(self, move):
        if not self.initialized or not self._is_human_turn():
            return

        self._move_done(move, self.position)

    def _is_human_turn(self):
        side = self.position.side_to_move
        return self.players[side] is None

    def _move_done(self, move, position, value=None):
        if self._move_callback is not None:
            self._move_callback(move=move, position=position)





