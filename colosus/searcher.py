import numpy as np

from colosus.config import SearchConfig
from .state import State


class Searcher:
    def __init__(self, config: SearchConfig):
        self.config = config

    def search(self, root_state: State, iterations: int) -> (np.array, float, int, State):
        if iterations == 1:
            return root_state.play_static_policy(self._get_temperature(root_state.position().move_count))

        for i in range(iterations):
            root_state.select()
        policy, value, move, new_state = root_state.play(self._get_temperature(root_state.position().move_count))
        return policy, value, move, new_state

    def _get_temperature(self, move_count):
        if move_count <= self.config.move_count_temp0:
            if isinstance(self.config.temp0, list):
                temp_prob = np.array(self.config.temp0[0])
                index = np.random.choice(len(temp_prob), 1, p=temp_prob)[0]
                return self.config.temp0[1][index]
            else:
                self.config.temp0
        else:
            return self.config.tempf

