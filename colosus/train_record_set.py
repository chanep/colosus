import pickle
import random
from typing import List

import os

from colosus.game.rotator import Rotator
from colosus.train_record import TrainRecord


class TrainRecordSet:
    records: List[TrainRecord]

    def __init__(self, records: List[TrainRecord] = None):
        if records is not None:
            self.records = records
        else:
            self.records = []

    def append(self, record: TrainRecord):
        self.records.append(record)

    def extend(self, records: List[TrainRecord]):
        self.records.extend(records)

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
    def merge_and_rotate(cls, merged_filename: str, file_count: int):
        merged_filename_parts = merged_filename.split(".")
        records = []
        filenames = []
        for i in range(file_count):
            filename = merged_filename_parts[0] + "_" + str(i) + "." + merged_filename_parts[1]
            filenames.append(filename)
            record_set = cls.load_from_file(filename)
            record_set.do_rotations()
            records.extend(record_set.records)
        random.shuffle(records)
        merged = TrainRecordSet(records)
        merged.save_to_file(merged_filename)
        for f in filenames:
            os.remove(f)

    @classmethod
    def duplications(cls, filename: str, value: float) -> (int, int, int):
        recordset = cls.load_from_file(filename)
        final_positions = {}
        total_positions = 0
        for r in recordset.records:
            if abs(r.value) > value:
                total_positions += 1
                h = r.position.hash()
                if h not in final_positions.keys():
                    final_positions[h] = 1
        total = total_positions / 8
        different = (len(final_positions.keys()) / 8)
        duplicated = total - different
        return total, different, duplicated

    @classmethod
    def load_from_file(cls, filename) -> 'TrainRecordSet':
        with open(filename, "rb") as f:
            return pickle.load(f)
