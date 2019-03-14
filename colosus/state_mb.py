import math
import types
from typing import List

import numpy as np

from colosus.config import StateConfig
from colosus.game.square import Square
from .game.position import Position
from .colosus_model import ColosusModel


class StateMb:
    _children: List['StateMb']

    def __init__(self, position: Position, p, parent: 'StateMb', config: StateConfig):
        self.config = config

        self.parent = parent
        self._position = position
        self.P = p
        self.N = 0
        self.W = 0
        self.Q = 0
        self.N_in_flight = 0
        self.is_leaf = True

        self._children = None
        self._prev_position = None
        self._move = None
        self.noise = None
        self.legal_policy = None
        self.policy = None

    def get_policy2(self):
        policy_len = len(self.children())
        policy = np.zeros(policy_len)
        for i in range(policy_len):
            child = self.children()[i]
            if child is not None:
                policy[i] = child.N / self.N
        return policy

    def get_policy(self):
        policy_len = len(self.children())
        policy = np.zeros(policy_len)
        for i in range(policy_len):
            child = self.children()[i]
            if child is not None:
                policy[i] = child.N / self.N + (child.P / math.sqrt(self.N))
        return policy / np.sum(policy)

    def play(self, temperature) -> (int, float, int, 'StateMb'):
        policy = self.get_policy()
        if temperature < 0.1:
            move = np.argmax(policy)
            temp_policy = np.zeros_like(policy)
            temp_policy[move] = 1.0
        else:
            play_policy = np.clip(policy + self.config.policy_offset, 0.0, None)
            play_policy = play_policy / np.sum(play_policy)

            temp_policy = self.apply_temperature(play_policy, temperature)
            move = np.random.choice(len(temp_policy), 1, p=temp_policy)[0]
        new_root_state = self.children()[move]
        new_root_state.parent = None
        self.noise = None
        value = -self.Q
        return policy, temp_policy, value, move, new_root_state

    def play_static_policy(self, temperature) -> (int, float, int, 'StateMb'):
        self.children()

        if temperature < 0.1:
            move = np.argmax(self.legal_policy)
            temp_policy = np.zeros_like(self.legal_policy)
            temp_policy[move] = 1.0
        else:
            temp_policy = self.apply_temperature(self.legal_policy, temperature)
            move = np.random.choice(len(temp_policy), 1, p=temp_policy)[0]

        value = - self.Q
        new_root_state = self.children()[move]
        new_root_state.Q = value
        self.noise = None
        return self.legal_policy, temp_policy, value, move, new_root_state

    def is_root(self):
        return self.parent is None

    def is_end(self):
        return self.position().is_end

    def get_best_child(self) -> 'StateMb':
        best_child = None
        best_score = -10000
        factor = self.config.cpuct * math.sqrt(self.N + self.N_in_flight)
        children = self.children()

        if self.is_root() and self.noise is None and self.config.noise_factor > 0:
            self.noise = np.random.dirichlet([self.config.noise_alpha] * len(children))

        fpu = -self.Q if self.is_root() else -self.Q - 1.2 * math.sqrt(self.P)

        for i in range(len(children)):
            child = children[i]  # 20%
            if child is not None:
                if self.noise is not None:
                    child_p = (1 - self.config.noise_factor) * child.P + self.config.noise_factor * self.noise[i]
                else:
                    child_p = child.P

                child_score = (child.Q if child.N > 0 else fpu) + ((factor * child_p) / (1 + child.N + child.N_in_flight))  # 55%

                if child_score > best_score:
                    best_score = child_score
                    best_child = child

        return best_child

    def get_q(self, default_q):
        return self.Q if self.N > 0 else default_q

    def position(self) -> Position:
        if self._position is not None:
            return self._position
        if self._prev_position is None:
            raise Exception("State _position and _position are both None")

        self._position = self._prev_position.move(self._move)
        self._prev_position = None
        return self._position

    def children(self):
        if self._children is not None:
            return self._children
        if self.policy is None:
            return None

        legal_moves = self.position().legal_moves()
        if len(legal_moves) == 0:
            raise Exception('No legal moves but position is not end')
        else:
            legal_policy = self._get_legal_policy(self.policy, legal_moves)
            self.legal_policy = [None] * len(self.policy)
            for m in legal_moves:
                self.legal_policy[m] = legal_policy[m]

        self._children = [None] * len(self.legal_policy)
        for move in range(len(self.legal_policy)):
            p = self.legal_policy[move]
            if p is not None:
                child = self.__class__(None, p.item(), self, self.config)
                child._prev_position = self._position
                child._move = move
                self._children[move] = child
        # self.legal_policy = None
        return self._children

    def backup(self, v, is_mini_bach=False):
        self.W += v
        self.N += 1
        if is_mini_bach:
            self.N_in_flight = 0
        self.Q = self.W / self.N
        if self.parent is not None:
            self.parent.backup(-v * self.config.backup_factor, is_mini_bach)

    def increment_n_in_flight(self):
        self.N_in_flight += 1
        if self.parent is not None:
            self.parent.increment_n_in_flight()

    def apply_policy_and_value(self, policy, value, is_mini_batch=True):
        self.policy = policy
        self.is_leaf = False
        self.backup(-value, is_mini_batch)


    @staticmethod
    def apply_temperature(policy, temperature):
        temp_policy = np.power(policy, 1 / temperature)
        return temp_policy / np.sum(temp_policy)

    def principal_variation(self):
        pv = []
        state = self
        while not state.is_leaf:
            best_move = None
            best_N = 0
            best_child = None
            for m in range(len(state.children())):
                child: StateMb = state.children()[m]
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

    def print(self):
        print("N: {}, W: {}, Q: {}".format(self.N, self.W, self.Q))
        self.position().print()


    @staticmethod
    def print_policy(policy, limit):
        move_policy = []
        for m in range(len(policy)):
            m_str = Square.to_string(m)
            move_policy.append((m_str, policy[m]))
        sorted_policy = sorted(move_policy, key=lambda t: t[1], reverse=True)
        for i in range(limit):
            m = sorted_policy[i]
            print("{} - {}".format(m[0], m[1]))