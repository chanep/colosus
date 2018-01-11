import pickle

from colosus.train_record import TrainRecord


class TrainRecordSet:
    def __init__(self):
        self.records = []

    def append(self, record: TrainRecord):
        self.records.append(record)

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
