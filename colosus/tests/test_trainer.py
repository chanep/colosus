import unittest
import numpy as np

from colosus.train_record_set import TrainRecordSet
from colosus.trainer import Trainer
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.game.piece import Piece


class Person:
    def __init__(self):
        self.name = "esteban"

    def __del__(self):
        print("del")


class SelfPlayTestCase(unittest.TestCase):
    def test_train(self):
        train_filename = "tz_2_1200_800.dat"
        weights_filename = "wz_2_1200_800.h5"
        # weights_filename = "wpp_3_1600_800.h5"
        prev_weights_filename = "wz_1_1200_800.h5"
        # prev_weights_filename = None

        trainer = Trainer()
        trainer.train(train_filename, weights_filename, 30, prev_weights_filename)

    def test_save_rotated_records(self):
        input_filename = "tz_2_150_800.dat"
        rotated_filename = "tz_2_1200_800.dat"
        recordset = TrainRecordSet.load_from_file(input_filename)
        recordset.do_rotations()
        recordset.save_to_file(rotated_filename)


    def test_generator(self):

        def generator():
            yield 3
            yield 4
            yield 5

        for x in generator():
            print(x)

        def p(x, y):
            print("x: {}, y: {}".format(x, y))

        def get_input():
            return 3, 4

        p(*get_input())

    def test_destructor(self):
        p = Person()
        print(p.name)




if __name__ == '__main__':
    unittest.main()