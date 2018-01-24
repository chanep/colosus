import math
import numpy as np

from colosus.config import StateConfig
from .game.position import Position
from .colosus_model import ColosusModel


class State:

    def __init__(self, position: Position, p, parent: 'State', colosus: ColosusModel, config: StateConfig):
        self.config = config

        self.parent = parent
        self.position = position
        self.colosus = colosus
        self.P = p

        self.N = 0
        self.W = 0
        self.Q = 0.0
        self.is_leaf = True
        self.is_end = position.is_end
        self._children = None
        self.noise = None
        self.legal_policy = None

    def get_policy(self, temperature):
        policy_len = len(self.children())
        inv_temp = 1 / temperature
        policy = np.zeros(policy_len)
        total_visit = math.pow(self.N, inv_temp)
        for i in range(policy_len):
            child = self.children()[i]
            if child is not None:
                child_visit = math.pow(child.N, inv_temp)
                policy[i] = child_visit / total_visit
        return policy / np.sum(policy)

    def play(self, policy) -> (int, 'State'):
        move = np.random.choice(len(policy), 1, p=policy)[0]
        new_root_state = self.children()[move]
        self.noise = None
        return move, new_root_state

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
        if self.is_end:
            value = self.position.score
        else:
            legal_moves = self.position.legal_moves()
            if len(legal_moves) == 0:  # stalemate
                value = 0
                self.is_end = True
                self.position.is_end = True
                self.position.score = 0
            else:
                self.is_leaf = False
                policy, value = self.colosus.predict(self.position.to_model_position())
                legal_policy = self.colosus.legal_policy(policy, legal_moves)
                self.legal_policy = [None] * len(policy)
                for m in legal_moves:
                    self.legal_policy[m] = legal_policy[m]

        self.backup(-value)

    def backup(self, v):
        self.W += v
        self.N += 1
        self.Q = self.W / self.N
        if self.parent is not None:
            self.parent.backup(-v)

    def children(self):
        if self._children is not None:
            return self._children
        if self.legal_policy is None:
            raise Exception("State legal_policy and children are both None")

        self._children = [None] * len(self.legal_policy)
        for move in range(len(self.legal_policy)):
            p = self.legal_policy[move]
            if p is not None:
                child_pos = self.position.move(move)
                child = self.__class__(child_pos, p, self, self.colosus, self.config)
                self._children[move] = child
        self.legal_policy = None
        return self._children

    def print(self):
        print("N: {}, W: {}, Q: {}".format(self.N, self.W, self.Q))
        self.position.print()

    def is_root(self):
        return self.parent is None
