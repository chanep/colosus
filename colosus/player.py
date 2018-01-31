from colosus.colosus_model import ColosusModel
from colosus.config import PlayerConfig
from colosus.game.position import Position
from colosus.searcher import Searcher
from colosus.state import State


class Player:
    def __init__(self, config: PlayerConfig):
        self.config = config
        self.iterations = None
        self.state = None
        self.searcher = None

    def new_game(self, initial_pos: Position, iterations_per_move: int, weights_filename: str):
        self.iterations = iterations_per_move

        colosus_config = self.config.colosus_config
        colosus = ColosusModel(colosus_config)
        colosus.build()
        if weights_filename is not None:
            colosus.load_weights(weights_filename)

        self.state = State(initial_pos, None, None, colosus, self.config.state_config)
        self.searcher = Searcher(self.config.search_config)

    def move(self):
        policy, value, move, new_state = self.searcher.search(self.state, self.iterations)
        old_state = self.state
        self.state = new_state
        return policy, value, move, old_state, new_state

    def opponent_move(self, move):
        if self.state.children() is not None and self.state.children()[move] is not None:
            self.state = self.state.children()[move]
        else:
            position = self.state.position().move(move)
            self.state = State(position, None, None, self.state.colosus, self.config.state_config)

