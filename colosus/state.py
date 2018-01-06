import math

from .game.position import Position
from .colosus_model import ColosusModel



class State:
    cpuct = 1.41

    def __init__(self, position: Position, p, colosus: ColosusModel, parent: 'State'):
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

    def select(self):
        if self.is_leaf:
            self.expand()
        else:
            selected_child = None
            best_score = -10000
            factor = State.cpuct * math.sqrt(self.N)

            for child in self.children:
                child_score = child.Q + ((factor * child.P) / (1 + child.N))
                if child_score > best_score:
                    best_score = child_score
                    selected_child = child

            selected_child.select()

    def expand(self):
        self.is_leaf = False
        if self.is_end:
            value = self.position.score
        else:
            policy, value = self.colosus.predict(self.position.board)
            legal_moves = self.position.legal_moves()
            legal_policy = self.colosus.legal_policy(policy, legal_moves)
            for i in range(len(legal_moves)):
                child_pos = self.position.move(legal_moves[i])
                child = State(child_pos, legal_policy[i], self.colosus, self)
                self.children.append(child)
        self.backup(value)

    def backup(self, v):
        self.W += v
        self.N += 1
        self.Q = self.W / self.N
        if self.parent is not None:
            self.parent.backup(-v)

