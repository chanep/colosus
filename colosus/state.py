import math
import numpy as np

from .game.position import Position
from .colosus_model import ColosusModel


class State:
    cpuct = 1.41

    def __init__(self, position: Position, p, parent: 'State', colosus: ColosusModel):
        self.parent = parent
        self.position = position
        self.colosus = colosus
        self.P = p

        self.N = 0
        self.W = 0
        self.Q = 0.0
        self.is_leaf = True
        self.is_end = position.is_end
        self.children = []
        self.noise = None

    def build_initial(self, position_ini: Position):
        return self.__class__(position_ini.clone(), None, None, self.colosus)

    def get_policy(self, temperature):
        policy_len = len(self.children)
        inv_temp = 1 / temperature
        policy = np.zeros(policy_len)
        total_visit = math.pow(self.N, inv_temp)
        for i in range(policy_len):
            child = self.children[i]
            if child is not None:
                child_visit = math.pow(child.N, inv_temp)
                policy[i] = child_visit / total_visit
        return policy / np.sum(policy)

    def play(self, policy) -> (int, 'State'):
        move = np.random.choice(len(policy), 1, p=policy)[0]
        new_root_state = self.children[move]
        self.noise = None
        return move, new_root_state

    def select(self):
        if self.is_leaf:
            self.expand()
        else:
            selected_child = None
            best_score = -10000
            factor = self.__class__.cpuct * math.sqrt(self.N)

            if self.is_root() and self.noise is None:
                self.noise = np.random.dirichlet([0.3] * len(self.children))

            for i in range(len(self.children)):
                child = self.children[i]
                if child is not None:
                    if self.is_root():
                        child_p = 0.75 * child.P + 0.25 * self.noise[i]
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
                self.children = [None] * len(policy)
                for move in legal_moves:
                    child_pos = self.position.move(move)
                    child = self.__class__(child_pos, legal_policy[move], self, self.colosus)
                    self.children[move] = child
        self.backup(-value)

    def backup(self, v):
        self.W += v
        self.N += 1
        self.Q = self.W / self.N
        if self.parent is not None:
            self.parent.backup(-v * 0.999)

    def print(self):
        print("N: {}, W: {}, Q: {}".format(self.N, self.W, self.Q))
        self.position.print()

    def is_root(self):
        return self.parent is None
