import numpy as np

from colosus.game import model_position


class TrainRecord:
    def __init__(self, position: model_position, policy: np.array, value: float):
        self.position = position
        self.policy = policy
        self.value = value
