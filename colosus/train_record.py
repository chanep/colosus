import numpy as np

from colosus.game.position import Position


class TrainRecord:
    def __init__(self, position: Position, policy: np.array, value: float):
        self.position = position
        self.policy = policy
        self.value = value
