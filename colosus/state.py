import math
import types

import numpy as np

from colosus.config import StateConfig
from .game.position import Position
from .colosus_model import ColosusModel


class State:

    def __init__(self, position: Position, p, parent: 'State', colosus: ColosusModel, config: StateConfig):
        self.config = config

        self.parent = parent
        self._position = position
        self.colosus = colosus
        self.P = p

        self.N = 0
        self.W = 0
        self.Q = 0.0
        self.is_leaf = True
        self._is_end = None
        self.children = []
        self.noise = None

    @staticmethod
    def apply_temperature(policy, temperature):
        temp_policy = np.power(policy, 1 / temperature)
        return temp_policy / np.sum(temp_policy)

    def get_policy(self):
        policy_len = len(self.children)
        policy = np.zeros(policy_len)
        for i in range(policy_len):
            child = self.children[i]
            if child is not None:
                policy[i] = child.N / self.N
        return policy

    def play(self, temperature) -> (int, 'State'):
        policy = self.get_policy()
        value = - self.Q
        temp_policy = self.apply_temperature(policy, temperature)
        move = np.random.choice(len(temp_policy), 1, p=temp_policy)[0]
        new_root_state = self.children[move]
        self.noise = None
        return policy, value, move, new_root_state

    def select(self):
        self._resolve_position()

        if self.is_leaf:
            self.expand()
        else:
            selected_child = None
            best_score = -10000
            factor = self.config.cpuct * math.sqrt(self.N)

            if self.is_root() and self.noise is None and self.config.noise_factor > 0:
                self.noise = np.random.dirichlet([self.config.noise_alpha] * len(self.children))

            for i in range(len(self.children)):
                child = self.children[i]
                if child is not None:
                    if self.noise is not None:
                        child_p = (1 - self.config.noise_factor) * child.P + self.config.noise_factor * self.noise[i]
                    else:
                        child_p = child.P
                    child_score = child.Q + ((factor * child_p) / (1 + child.N))
                    if child_score > best_score:
                        best_score = child_score
                        selected_child = child

            selected_child.select()

    def expand(self):
        if self._is_end:
            value = self._position.score
        else:
            legal_moves = self._position.legal_moves()
            if len(legal_moves) == 0:
                raise Exception('No legal moves but position is not end')
            else:
                self.is_leaf = False
                policy, value = self.colosus.predict(self._position.to_model_position())
                legal_policy = self.colosus.legal_policy(policy, legal_moves)
                self.children = [None] * len(policy)
                for move in legal_moves:
                    child_pos = lambda _, m = move: self._position.move(m)
                    # child_pos = self._position.move(move)
                    child = self.__class__(child_pos, legal_policy[move], self, self.colosus, self.config)
                    self.children[move] = child
        self.backup(-value)

    def backup(self, v):
        self.W += v
        self.N += 1
        self.Q = self.W / self.N
        if self.parent is not None:
            self.parent.backup(-v)

    def print(self):
        print("N: {}, W: {}, Q: {}".format(self.N, self.W, self.Q))
        self._position.print()

    def is_root(self):
        return self.parent is None

    def _resolve_position(self):
        if isinstance(self._position, types.FunctionType):
            self._position = self._position(None)
        self._is_end = self._position.is_end

    def position(self):
        self._resolve_position()
        return self._position

