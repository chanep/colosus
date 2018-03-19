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
        self._children = None
        self._prev_position = None
        self._move = None
        self.noise = None
        self.legal_policy = None

    @staticmethod
    def apply_temperature(policy, temperature):
        temp_policy = np.power(policy, 1 / temperature)
        return temp_policy / np.sum(temp_policy)

    def get_policy(self):
        policy_len = len(self.children())
        policy = np.zeros(policy_len)
        for i in range(policy_len):
            child = self.children()[i]
            if child is not None:
                policy[i] = child.N / self.N
        return policy

    def play(self, temperature) -> (int, float, int, 'State'):
        policy = self.get_policy()
        if temperature < 0.1:
            move = np.argmax(policy)
        else:
            temp_policy = self.apply_temperature(policy, temperature)
            move = np.random.choice(len(temp_policy), 1, p=temp_policy)[0]
        new_root_state = self.children()[move]
        new_root_state.parent = None
        self.noise = None
        value = -self.Q
        return policy, value, move, new_root_state

    def play_static_policy(self, temperature) -> (int, float, int, 'State'):
        self.select()
        policy = np.nan_to_num(np.array(self.legal_policy, np.float), 0)
        value = - self.Q
        if temperature < 0.1:
            move = np.argmax(policy)
        else:
            temp_policy = self.apply_temperature(policy, temperature)
            move = np.random.choice(len(temp_policy), 1, p=temp_policy)[0]
        new_root_state = self.__class__(self.position().move(move), policy[move], None, self.colosus, self.config)
        new_root_state.Q = value
        self.noise = None
        return policy, value, move, new_root_state

    def select(self):
        if self.is_leaf:
            self.expand()
        else:
            selected_child = None
            best_score = -10000
            factor = self.config.cpuct * math.sqrt(self.N)

            if self.is_root() and self.noise is None and self.config.noise_factor > 0:
                self.noise = np.random.dirichlet([self.config.noise_alpha] * len(self.children()))

            for i in range(len(self.children())):
                child = self.children()[i]
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
        if self.position().is_end:
            value = self.position().score
        else:
            legal_moves = self.position().legal_moves()
            if len(legal_moves) == 0:
                raise Exception('No legal moves but position is not end')
            else:
                self.is_leaf = False
                policy, value = self.colosus.predict(self.position().to_model_position())
                legal_policy = self._get_legal_policy(policy, legal_moves)
                self.legal_policy = [None] * len(policy)
                for m in legal_moves:
                    self.legal_policy[m] = legal_policy[m]

        self.backup(-value)

    def principal_variation(self):
        pv = []
        state = self
        while not state.is_leaf:
            best_move = None
            best_N = 0
            best_child = None
            for m in range(len(state.children())):
                child: State = state.children()[m]
                if child is not None and child.N > best_N:
                    best_N = child.N
                    best_move = m
                    best_child = child
            if best_move is not None:
                pv.append(best_move)
                state = best_child
            else:
                break
        return pv

    @staticmethod
    def _get_legal_policy(policy, legal_moves):
        legal_policy = np.zeros_like(policy)
        for m in legal_moves:
            legal_policy[m] = policy[m]
        return legal_policy / np.sum(legal_policy)

    def backup(self, v):
        self.W += v
        self.N += 1
        self.Q = self.W / self.N
        if self.parent is not None:
            self.parent.backup(-v * self.config.backup_factor)

    def children(self):
        if self._children is not None:
            return self._children
        if self.legal_policy is None:
            return None

        self._children = [None] * len(self.legal_policy)
        for move in range(len(self.legal_policy)):
            p = self.legal_policy[move]
            if p is not None:
                child = self.__class__(None, p, self, self.colosus, self.config)
                child._prev_position = self._position
                child._move = move
                self._children[move] = child
        self.legal_policy = None
        return self._children

    def position(self):
        if self._position is not None:
            return self._position
        if self._prev_position is None:
            raise Exception("State _position and _position are both None")

        self._position = self._prev_position.move(self._move)
        self._prev_position = None
        return self._position

    def print(self):
        print("N: {}, W: {}, Q: {}".format(self.N, self.W, self.Q))
        self.position().print()

    def is_root(self):
        return self.parent is None
