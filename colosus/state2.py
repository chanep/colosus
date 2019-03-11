import math
import numpy as np

from colosus.state import State
from .game.position import Position
from .colosus_model import ColosusModel
from colosus.config import StateConfig


class State2(State):
    def __init__(self, position: Position, p, parent: 'State', colosus: ColosusModel, config: StateConfig):
        self.config = config

        self.parent = parent
        self._position = position
        self.colosus = colosus
        self.P = p

        self.N = 0
        self.W = 0
        self.Q = 0 if parent is None else (-parent.Q if parent.parent is None else -parent.Q - 1.0 * math.sqrt(parent.P))
        self.is_leaf = True
        self._children = None
        self._prev_position = None
        self._move = None
        self.noise = None
        self.legal_policy = None
        self.policy = None

