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
            'e_09_2000_800.dat',
            'e_10_2000_800.dat',
            'e_08_2000_800.dat'
        ]

        recordset = TrainRecordSet()

        for f in files:
            r = TrainRecordSet.load_from_file(f)
            recordset.extend(r.records)

        random.shuffle(recordset.records)

        recordset.save_to_file('e_0810_2000_800.dat')


    def test_truncate(self):
        input = 'd_50_2000_800.dat'
        output = 'd_50_100_800.dat'
        percent = 5

        recordset = TrainRecordSet.load_from_file(input)
        count = int((len(recordset.records) * percent) / 100)
        recordset.records = recordset.records[0:count]
        recordset.save_to_file(output)

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

    def test_duplications(self):
        train_filename = "d_50c_2000_800.dat"
        total, different, duplicated = TrainRecordSet.duplications(train_filename, 0.9)
        print("final positions:")
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))


if __name__ == '__main__':
    unittest.main()