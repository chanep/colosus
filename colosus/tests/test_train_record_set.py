import unittest

import pickle
import numpy as np

from colosus.game.piece import Piece
from colosus.game.side import Side
from colosus.train_record import TrainRecord
from colosus.train_record_set import TrainRecordSet
from colosus.game.position import Position


class TrainRecordSetTestCase(unittest.TestCase):
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