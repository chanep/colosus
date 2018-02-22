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
        weights_filename = "x.h5"
        # weights_filename = "wpp_3_1600_800.h5"
        prev_weights_filename = "c_18_1000_1600.h5"
        # prev_weights_filename = None

        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.0001
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 15, prev_weights_filename)

    def test_train_multi(self):

        print("training cnone_17_1000_1600.h5...")
        train_filename = "c_17_1000_1600.dat"
        weights_filename = "cnone_17_1000_1600.h5"
        prev_weights_filename = None
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.00005
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 50, prev_weights_filename)
        print("cnone_17_1000_1600.h5 done!\n")

        print("training zcnone_17_1000_1600.h5...")
        train_filename = "zc_17_1000_1600.dat"
        weights_filename = "zcnone_17_1000_1600.h5"
        prev_weights_filename = None
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.00005
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 50, prev_weights_filename)
        print("cnone_17_1000_1600.h5 done!\n")

        print("training c_17_1000_1600.h5...")
        train_filename = "c_17_1000_1600.dat"
        weights_filename = "c_17_1000_1600.h5"
        prev_weights_filename = "c_16_800_1600.h5"
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.00002
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 20, prev_weights_filename)
        print("c_17_1000_1600.h5 done!\n")

        print("training zc_17_1000_1600.h5...")
        train_filename = "zc_17_1000_1600.dat"
        weights_filename = "zc_17_1000_1600.h5"
        prev_weights_filename = "c_16_800_1600.h5"
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.00002
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 20, prev_weights_filename)
        print("zc_17_1000_1600.h5 done!\n")

        print("training cres3_17_1000_1600.h5...")
        train_filename = "c_17_1000_1600.dat"
        weights_filename = "cres3_17_1000_1600.h5"
        prev_weights_filename = None
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.00005
        trainer_config.colosus_config.residual_blocks = 3
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 50, prev_weights_filename)
        print("cres3_17_1000_1600.h5 done!\n")


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
