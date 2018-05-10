from colosus.colosus_model import ColosusModel
from colosus.config import PlayerConfig
from colosus.game.position import Position
from colosus.searcher import Searcher
from colosus.searcher_mp import SearcherMp
from colosus.state import State


class PlayerMp:
    def __init__(self, config: PlayerConfig, colosus: ColosusModel):
        self.config = config
        self.iterations = None
        self.state = None
        self.searcher = None
        self.colosus = colosus

    def new_game(self, initial_pos: Position, iterations_per_move: int):
        self.iterations = iterations_per_move
        self.state = State(initial_pos, None, None, self.colosus, self.config.state_config)
        self.searcher = SearcherMp(self.config.search_config)

    def move(self):
        policy, policy_temp, value, move, new_state = self.searcher.search(self.state, self.iterations)
        old_state = self.state
        self.state = new_state
        return policy, value, move, old_state, new_state

    def opponent_move(self, move):
        if self.state.children() is not None and self.state.children()[move] is not None:
            self.state = self.state.children()[move]
            self.state.parent = None
        else:
            position = self.state.position().move(move)
            self.state = State(position, None, None, self.state.colosus, self.config.state_config)
