import pickle
from typing import List

from colosus.game.rotator import Rotator
from colosus.train_record import TrainRecord


class TrainRecordSet:
    def __init__(self):
        self.records = []

    def append(self, record: TrainRecord):
        self.records.append(record)

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def do_rotations(self):
        rotator = Rotator()
        with_rotations = []
        for r in self.records:
            with_rotations.extend(rotator.rotations(r))
        self.records = with_rotations

    @classmethod
    def load_from_file(cls, filename) -> 'TrainRecordSet':
        with open(filename, "rb") as f:
            return pickle.load(f)
