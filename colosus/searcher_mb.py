import numpy as np

from colosus.config import SearchConfigMb
from .state_mb import StateMb


class SearcherMb:
    root_state: StateMb

    def __init__(self, config: SearchConfigMb):
        self.config = config
        self.root_state = None

    def search(self, root_state: StateMb, iterations: int) -> (np.array, float, int, StateMb):
        self.root_state = root_state

        if iterations == 1:
            return root_state.play_static_policy(self._get_temperature(root_state.position().move_count))

        for i in range(iterations):
            root_state.select()
        policy, temp_policy, value, move, new_state = root_state.play(self._get_temperature(root_state.position().move_count))
        return policy, temp_policy, value, move, new_state

    def _get_temperature(self, move_count):
        if move_count <= self.config.move_count_temp0:
            if isinstance(self.config.temp0, list):
                temp_prob = np.array(self.config.temp0[0])
                index = np.random.choice(len(temp_prob), 1, p=temp_prob)[0]
                return self.config.temp0[1][index]
            else:
                return self.config.temp0
        else:
            return self.config.tempf


    def _execute_mb(self):
        _gather_mb()

    def _gather_mb(self):
        mini_batch = []
        collisions = self.config.max_collisions
        mb_size = self.config.mb_size
        while mb_size < len(mini_batch) and collisions >= 0:
            state = self._pick_state_to_extend()



    def _pick_state_to_extend(self):
        if self.root_state.is_leaf:
            return self.root_state

        state = self.root_state
        while True:
            best_child = state.get_best_child()

            if best_child.is_leaf:
                return best_child

            state = best_child


