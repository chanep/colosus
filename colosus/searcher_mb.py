from typing import List

import numpy as np

from colosus.colosus_model import ColosusModel
from colosus.config import SearchConfig
from .state_mb import StateMb
import time


class Stats:
    def __init__(self):
        self.nodes = 0
        self.nn_computations = 0
        self.out_of_orders = 0
        self.collisions = 0
        self.mini_batches = []

    def inc_nn_computation(self, mb_size):
        self.nn_computations += 1
        self.mini_batches.append(mb_size)

    def inc_out_of_orders(self):
        self.out_of_orders += 1

    def inc_collisions(self):
        self.collisions += 1

    def reset(self):
        self.nodes = 0
        self.nn_computations = 0
        self.out_of_orders = 0
        self.collisions = 0
        self.mini_batches = []

    def print(self):
        print(f"nodes: {self.nodes}")
        print(f"nn computations: {self.nn_computations}")
        print(f"mini batch size (avg/min/max): {sum(self.mini_batches) / len(self.mini_batches) : .1f}/"
              f"{min(self.mini_batches)}/{max(self.mini_batches)}")
        print(f"out_of_orders: {self.out_of_orders}")
        print(f"collisions: {self.collisions}")


class SearcherMb:
    root_state: StateMb

    def __init__(self, config: SearchConfig, colosus: ColosusModel):
        self.config = config
        self.colosus = colosus
        self.root_state = None
        self._nodes = 0
        self._iterations = 0
        self._time_per_move = 0
        self._start_time = 0
        self.stats = Stats()

    def search(self, root_state: StateMb, iterations: int = 0, time_per_move: float = 1) -> (np.array, float, int, StateMb):
        self.root_state = root_state
        self._nodes = 0
        self._iterations = iterations
        self._time_per_move = time_per_move
        self._start_time = time.time()
        self.stats.reset()

        if iterations == 1:
            self._execute_mb_iteration(1)
            return root_state.play_static_policy(self._get_temperature(root_state.position().move_count))

        while not self._should_stop():
            self._execute_mb_iteration(self.config.mb_size)

        self.stats.nodes = self._nodes

        policy, temp_policy, value, move, new_state = root_state.play(self._get_temperature(root_state.position().move_count))
        return policy, temp_policy, value, move, new_state

    def _should_stop(self):
        if self._iterations == 0:
            elapsed = time.time() - self._start_time
            return elapsed > self._time_per_move
        else:
            return self._nodes >= self._iterations

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

    def _execute_mb_iteration(self, mb_size):
        mini_batch = self._gather_mb(mb_size)
        if len(mini_batch) > 0:
            self._run_nn_computation(mini_batch)

    def _gather_mb(self, mb_size):
        # print("gather " + str(self._nodes))
        mini_batch = []
        collisions = self.config.max_collisions
        while mb_size > len(mini_batch) and collisions > 0 and not self._should_stop():
            state = self._pick_state_to_extend()
            if state.is_end():
                # print("is_end")
                value = state.position().score
                state.backup(-value)
                self._nodes += 1
                self.stats.inc_out_of_orders()
            else:
                if state.N_in_flight == 0:
                    self._nodes += 1
                    mini_batch.append(state)
                else:                           # collision
                    collisions -= 1
                    self.stats.inc_collisions()

                state.increment_n_in_flight()

        return mini_batch

    def _pick_state_to_extend(self) -> StateMb:
        if self.root_state.is_leaf:
            return self.root_state

        state = self.root_state
        while True:
            best_child = state.get_best_child()

            if best_child.is_leaf:
                return best_child

            state = best_child

    def _run_nn_computation(self, mini_batch: List[StateMb]):
        model_positions = []
        for state in mini_batch:
            model_positions.append(state.position().to_model_position())

        policies, values = self.colosus.predict_on_batch(model_positions)

        self.stats.inc_nn_computation(len(mini_batch))

        for i in range(len(mini_batch)):
            state = mini_batch[i]
            state.apply_policy_and_value(policies[i], values[i], is_mini_batch=True)
