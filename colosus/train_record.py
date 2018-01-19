import numpy as np

from colosus.game.model_position import ModelPosition


class TrainRecord:
    def __init__(self, position: ModelPosition, policy: np.array, value: float):
        self.position = position
        self.policy = policy
        self.value = value
