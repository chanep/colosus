from typing import List
import math
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


class HashTable:
    def __init__(self):
        self.entries = {}
        self.hits = 0
        self.searches = 0

    def store(self, hash):
        self.entries[hash] = 1

    def search(self, hash):
        self.searches += 1
        entry = self.entries.get(hash)
        if entry is not None:
            self.hits += 1
        return entry


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
        self.hash_table = HashTable()

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

    def _get_remaining_nodes(self):
        if self._iterations != 0:
            return self._iterations - self._nodes
        else:
            elapsed = time.time() - self._start_time
            if elapsed == 0:
                return 10000000
            nodes_per_second = self._nodes / elapsed
            remaining = self._time_per_move - elapsed
            return int(remaining * nodes_per_second)

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

                    # hash = state.position().hash()
                    # entry = self.hash_table.search(hash)
                    # if entry is None:
                    #     self.hash_table.store(hash)

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
            best_child = self._get_best_child(state)

            if best_child.is_leaf:
                return best_child

            state = best_child

    def _get_best_child(self, state: StateMb) -> StateMb:
        best_child = None
        max_visits = 0
        best_score = -10000
        factor = self.config.cpuct * math.sqrt(state.N + state.N_in_flight)
        children = state.children()

        if state.is_root() and state.noise is None and self.config.noise_factor > 0:
            state.noise = np.random.dirichlet([self.config.noise_alpha] * len(children))

        fpu = (-state.Q + self.config.fpuRoot) if state.is_root() else -state.Q - 1.2 * math.sqrt(state.P)

        if self.config.smart_pruning_factor > 0 and state.is_root():
            max_visits = max((c.N if c is not None else 0) for c in children)

        for i in range(len(children)):
            child = children[i]  # 20%
            if child is not None:
                if max_visits > 0 and self._nodes > 0 and (
                        (max_visits - child.N) * self.config.smart_pruning_factor) > max(1, self._get_remaining_nodes()):
                    # print('continue...')
                    # print(f"nodes: {self._nodes}, max visits: {max_visits}, N: {child.N}, remaining: {self._get_remaining_nodes()}")
                    # print("elapsed: " + str(time.time() - self._start_time))
                    continue

                if state.noise is not None:
                    child_p = (1 - self.config.noise_factor) * child.P + self.config.noise_factor * state.noise[i]
                else:
                    child_p = child.P

                child_score = (child.Q if child.N > 0 else fpu) + ((factor * child_p) / (1 + child.N + child.N_in_flight))  # 55%

                if child_score > best_score:
                    best_score = child_score
                    best_child = child

        return best_child

    def _run_nn_computation(self, mini_batch: List[StateMb]):
        model_positions = []
        for state in mini_batch:
            model_positions.append(state.position().to_model_position())

        policies, values = self.colosus.predict_on_batch(model_positions)

        self.stats.inc_nn_computation(len(mini_batch))

        for i in range(len(mini_batch)):
            state = mini_batch[i]
            state.apply_policy_and_value(policies[i], values[i], is_mini_batch=True)
