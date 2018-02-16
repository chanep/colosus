import random
import unittest

import pickle
import numpy as np

from colosus.game.side import Side
from colosus.train_record import TrainRecord
from colosus.train_record_set import TrainRecordSet
from colosus.game.position import Position


class TrainRecordSetTestCase(unittest.TestCase):
    def test_normalize_policy(self):
        record_set = TrainRecordSet.load_from_file("c_gtp_4_310_300.dat")
        for r in record_set.records:
            r.policy = r.policy / np.sum(r.policy)
        record_set.save_to_file("c_gtpn_4_310_300.dat")

    def test_sample(self):
        record_set = TrainRecordSet.load_from_file("c_12_800_1600.dat")
        records_count = len(record_set.records)
        sample_count = int(records_count / 4)
        sample = record_set.records[0:sample_count]
        sample_set = TrainRecordSet(sample)
        sample_set.save_to_file("c_12_1_200_1600.dat")

    def test_merge(self):
        files = [
            'c_15_600_1600.dat',
            'c_15_2_100_1600.dat',
            'c_15_3_100_1600.dat'
        ]

        recordset = TrainRecordSet()

        for f in files:
            r = TrainRecordSet.load_from_file(f)
            recordset.extend(r.records)

        random.shuffle(recordset.records)

        recordset.save_to_file('c_15_800_1600.dat')

    def test_truncate(self):
        input = 'c_16_800_1600.dat'
        output = 'c_16_400_1600.dat'
        percent = 50

        recordset = TrainRecordSet.load_from_file(input)
        count = int((len(recordset.records) * percent) / 100)
        recordset.records = recordset.records[0:count]
        recordset.save_to_file(output)

    def test_save_and_load(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 3)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 4)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 5)

        policy = np.array([0.5, 0.5])

        value = 0.8

        record = TrainRecord(pos.to_model_position(), policy, value)

        record_set = TrainRecordSet()
        record_set.append(record)

        filename = "bin2.dat"
        record_set.save_to_file(filename)

        record_set2 = TrainRecordSet.load_from_file(filename)

        self.assertEqual(value, record_set2.records[0].value)

    def test_save_and_load_multiple(self):

        x = "hola"
        y = "chau"
        filename = "bin2.dat"

        with open(filename, "wb") as f:
            pickle.dump(x, f)
            pickle.dump(y, f)

        with open(filename, "rb") as f:
            try:
                print(pickle.load(f))
                print(pickle.load(f))
                print(pickle.load(f))
            except EOFError:
                ...


if __name__ == '__main__':
    unittest.main()