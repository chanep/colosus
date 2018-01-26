import unittest
import numpy as np

from colosus.config import TrainerConfig
from colosus.train_record_set import TrainRecordSet
from colosus.trainer import Trainer
from colosus.game.position import Position
from colosus.game.side import Side


class Person:
    def __init__(self):
        self.name = "esteban"

    def __del__(self):
        print("del")


class SelfPlayTestCase(unittest.TestCase):
    def test_train(self):
        # train_filename = "c_2_800_800.dat"
        # weights_filename = "c_2_800_800.h5"
        train_filename = "c_2_2480_256.dat"
        weights_filename = "x.h5"
        # weights_filename = "wpp_3_1600_800.h5"
        prev_weights_filename = "c_1_2000_256.h5"
        # prev_weights_filename = None

        trainer = Trainer(TrainerConfig())
        trainer.train(train_filename, weights_filename, 10, prev_weights_filename)

    def test_save_rotated_records(self):
        input_filename = "c_1_200_30.dat"
        rotated_filename = "c_1_1600_30.dat"
        recordset = TrainRecordSet.load_from_file(input_filename)
        recordset.do_rotations()
        recordset.save_to_file(rotated_filename)

    def test_merge_records(self):
        merged_filename = "c_2_310_256.dat"
        TrainRecordSet.merge_and_rotate(merged_filename, 4)

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