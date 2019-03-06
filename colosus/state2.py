import math
import numpy as np

from colosus.state import State
from .game.position import Position
from .colosus_model import ColosusModel


class State2(State):
    pass
    # def select(self):
    #     if self.is_leaf:
    #         self.expand()
    #     else:
    #         selected_child = None
    #         best_score = -10000
    #         factor = self.config.cpuct * math.sqrt(self.N)
    #
    #         if self.is_root() and self.noise is None and self.config.noise_factor > 0:
    #             self.noise = np.random.dirichlet([self.config.noise_alpha] * len(self.children()))
    #
    #         for i in range(len(self.children())):
    #             child = self.children()[i]
    #             if child is not None:
    #                 if self.noise is not None:
    #                     child_p = (1 - self.config.noise_factor) * child.P + self.config.noise_factor * self.noise[i]
    #                 else:
    #                     child_p = child.P
    #                 child_score = child.Q + self.config.cpuct * child_p * math.exp(- 0.1 * child.N) + self.config.cpuct * math.sqrt(self.N) / (1 + child.N)
    #                 if child_score > best_score:
    #                     best_score = child_score
    #                     selected_child = child
    #
    #         selected_child.select()

