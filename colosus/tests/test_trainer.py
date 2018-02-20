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
        train_filename = "c_18_1000_1600.dat"
        weights_filename = "cnone_reg3-5_18_1000_1600.h5"
        # weights_filename = "wpp_3_1600_800.h5"
        prev_weights_filename = "cnone_reg3-5_18_1000_1600.h5"
        # prev_weights_filename = None

        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.0001
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 2, prev_weights_filename)

    def test_save_rotated_records(self):
        input_filename = "c_1_200_30.dat"
        rotated_filename = "c_1_1600_30.dat"
        recordset = TrainRecordSet.load_from_file(input_filename)
        recordset.do_rotations()
        recordset.save_to_file(rotated_filename)

    def test_merge_records(self):
        merged_filename = "c_18_2_500_1600.dat"
        TrainRecordSet.merge_and_rotate(merged_filename, 16)

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