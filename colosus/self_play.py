import numpy as np

from colosus.colosus_model import ColosusModel
from .game.position import Position
from .state import State
from .searcher import Searcher

class SelfPlay:
    def play(self, games: int, iterations_per_move: int, initial_pos: Position):
        colosus = ColosusModel()
        colosus.build()

        # initial_state = State(initial_pos, None, None, colosus)
        searcher = Searcher()
        for i in range(games):
            state = State(initial_pos, None, None, colosus)
            print("initial state N: " + str(state.N))
            end = False
            while not end:
                policy, value, move, new_state = searcher.search(state, iterations_per_move)
                new_state.parent = None
                end = new_state.position.is_end
                state = new_state
                mc = state.position.move_count
                if mc % 20 == 0:
                    print("move count: " + str(state.position.move_count))
            print("fin game " + str(i))
            state.print()
            print("position score: " + str(state.position.score))