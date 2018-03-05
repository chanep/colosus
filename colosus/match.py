from concurrent.futures import ThreadPoolExecutor

from colosus.Illegal_move import IllegalMove
from colosus.colosus_model import ColosusModel
from colosus.config import PlayerConfig, ColosusConfig, MatchConfig
from colosus.game.position import Position
from colosus.player import Player
from colosus.player_mp import PlayerMp
from colosus.player_type import PlayerType


class PlayerSettings:
    def __init__(self, type: PlayerType, iterations: int=None, weights_filename: str=None):
        self.type = type
        self.iterations = iterations
        self.weights_filename = weights_filename


class Match:
    def __init__(self, config: MatchConfig):
        self.config = config
        self.position = None
        self.players = [None, None]
        self._move_callback = None
        self._match_initialized_callback = None
        self.initialized = False
        self._executor = ThreadPoolExecutor()
        self._thinking_future = None
        self.in_progress = False

    def new_game(self, black: PlayerSettings, white: PlayerSettings, initial_pos: Position=None,
                 move_callback=None, match_initialized_callback=None):
        if self._thinking_future is not None:
            self._thinking_future.cancel()

        if initial_pos is None:
            self.position = Position()
        else:
            self.position = initial_pos

        self._move_callback = move_callback
        self._match_initialized_callback = match_initialized_callback

        player_settings = [black, white]

        self.in_progress = True

        f = self._executor.submit(self._initialize, player_settings)
        f.add_done_callback(self._initialize_done)

    def _initialize(self, player_settings):
        for i in range(len(player_settings)):
            player_sets = player_settings[i]
            if player_sets.type == PlayerType.COLOSUS:
                colosus = ColosusModel(self.config.colosus_config)
                colosus.build()
                if player_sets.weights_filename is not None:
                    colosus.load_weights(player_sets.weights_filename)
                self.players[i] = PlayerMp(self.config.player_config, colosus) if self.config.mp else \
                    Player(self.config.player_config, colosus)
                self.players[i].new_game(self.position.clone(), player_sets.iterations)
            else:
                self.players[i] = None
        self.initialized = True

    def _initialize_done(self, future):
        exception = future.exception()
        if exception is not None:
            raise exception
        if self._match_initialized_callback is not None:
            self._match_initialized_callback(self)
        self._start_thinking_if_applies()

    def _current_player(self) -> Player:
        return self.players[self.position.side_to_move]

    def _colosus_thinks(self):
        c_player = self._current_player()
        policy, value, move, old_state, new_state = c_player.move()
        return policy, value, int(move), old_state, new_state

    def _colosus_thinks_done(self, future):
        policy, value, move, old_state, new_state = future.result()
        print("colosus thinks done " + str(move))
        value = new_state.Q
        pv = old_state.principal_variation()
        self._do_move(move, value, pv)

    def move(self, move):
        if not self.initialized or not self.is_human_turn():
            return
        if not self.position.is_legal(move):
            message = "Illegal Move"
            if self.position.move_count == 0:
                message = "First move must be on the center (8, 8)"
            elif self.position.move_count == 2:
                message = "Second black move must be at least 5 squares away from the center. e.g. (9, 13)"
            raise IllegalMove(message)
        self._do_move(move)

    def is_human_turn(self):
        side = self.position.side_to_move
        return self.players[side] is None and not self.position.is_end

    def is_end(self):
        return self.position.is_end

    def _do_move(self, move, value=None, pv=None):
        self.position = self.position.move(move)
        if self.position.is_end:
            self.in_progress = False
        if self._move_callback is not None:
            self._move_callback(self, move=move, value=value, pv=pv)
        opponent = self.players[self.position.side_to_move]
        if opponent is not None:
            opponent.opponent_move(move)
        self._start_thinking_if_applies()

    def _start_thinking_if_applies(self):
        if not self.is_end() and not self.is_human_turn():
            self._thinking_future = self._executor.submit(self._colosus_thinks)
            self._thinking_future.add_done_callback(self._colosus_thinks_done)








