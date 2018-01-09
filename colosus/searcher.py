import numpy as np
from .state import State


class Searcher:
    def search(self, root_state: State, iterations: int) -> (np.array, float, int, State):
        for i in range(iterations):
            root_state.select()
        policy = root_state.get_policy(self._get_temperature(root_state))
        value = root_state.Q
        move, new_state = root_state.play(policy)
        return policy, value, move, new_state

    def _get_temperature(self, state: State):
        if state.position.move_count <= 10:
            return 1.0
        else:
            return 0.1
