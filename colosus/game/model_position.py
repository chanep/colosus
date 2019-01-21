import numpy as np


class ModelPosition:
    def __init__(self, board: np.ndarray):
        self.board = board

    def hash(self):
        return hash(self.board.data.tobytes())

