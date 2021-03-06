import math
import types
from typing import List

import numpy as np

from colosus.config import StateConfig
from colosus.game.square import Square
from .game.position import Position


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

    def get_policy(self, offset=0):
        policy_len = len(self.children())
        policy = np.zeros(policy_len)
        for i in range(policy_len):
            child = self.children()[i]
            if child is not None:
                policy[i] = max(0, (child.N + offset)) / self.N + (child.P / math.sqrt(self.N))
        return policy / np.sum(policy)

    def play(self, temperature) -> (int, float, int, 'StateMb'):
        policy = self.get_policy(self.config.policy_offset)
        play_policy = self.get_policy(self.config.policy_offset + self.config.play_policy_offset)
        if temperature < 0.1:
            move = np.argmax(play_policy)
            temp_policy = np.zeros_like(play_policy)
            temp_policy[move] = 1.0
        else:
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

    def children(self) -> List['StateMb']:
        if self._children is not None:
            return self._children
        if self.policy is None:
            return None

        legal_moves = self.position().legal_moves()
        if len(legal_moves) == 0:
            raise Exception('No legal moves but position is not end')
        else:
            self.legal_policy = self._get_legal_policy(self.policy, legal_moves)

        self._children = [None] * len(self.legal_policy)
        for move in legal_moves:
            p = self.legal_policy[move]
            child = self.__class__(None, p.item(), self, self.config)
            child._prev_position = self._position
            child._move = move
            self._children[move] = child
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
        print("N: {}, W: {}, Q: {}, P: {}".format(self.N, self.W, self.Q, self.P))
        self.position().print()

    def print_children_stats(self, count=10):
        sorted_children = sorted(self.children(), key=lambda c: -1 if c is None else c.N, reverse=True)
        for i in range(count):
            c = sorted_children[i]
            if c is not None:
                print(f"{Square.to_string(c._move)} - N: {c.N}, Q: {c.Q}, W: {c.W}, P: {c.P}")

    def sort_policy(self, policy):
        move_policy = []
        for m in range(len(policy)):
            m_str = Square.to_string(m)
            move_policy.append((m_str, policy[m]))
        return sorted(move_policy, key=lambda t: t[1], reverse=True)

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