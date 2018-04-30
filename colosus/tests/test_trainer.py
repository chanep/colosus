import unittest
import numpy as np

from colosus.config import TrainerConfig
from colosus.tests.test_evaluator import EvaluatorTestCase
from colosus.train_record_set import TrainRecordSet
from colosus.trainer import Trainer
from colosus.game.position import Position
from colosus.game.side import Side
import os.path



class Person:
    def __init__(self):
        self.name = "esteban"

    def __del__(self):
        print("del")


class TrainerTestCase(unittest.TestCase):
    def test_train(self):
        print("training c_44_7300_800_60.h5...")
        train_filename = "c_44_7300_800_60.dat"
        weights_filename = "c_44_7300_800_60.h5"
        prev_weights_filename = "c_44_7300_800.h5"
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.00003
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 2, prev_weights_filename)

        # train_filename = "c_40_4000_1600.dat"
        # weights_filename = "c_40_4000_1600.h5"
        # prev_weights_filename = "c_40_4000_1600.h5"
        # trainer_config = TrainerConfig()
        # trainer_config.colosus_config.lr = 0.00003
        # trainer = Trainer(trainer_config)
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)

    def test_train_multi(self):
        print("training c_45_10100_800.h5...")
        train_filename = "c_45_10100_800.dat"
        weights_filename = "cn_45_10100_800.h5"
        prev_weights_filename = None
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.0001
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 2, prev_weights_filename)

        train_filename = "c_45_10100_800.dat"
        weights_filename = "cn_45_10100_800.h5"
        prev_weights_filename = "cn_45_10100_800.h5"
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.00005
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 2, prev_weights_filename)

        train_filename = "c_45_10100_800.dat"
        weights_filename = "cn_45_10100_800.h5"
        prev_weights_filename = "cn_45_10100_800.h5"
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.00002
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 4, prev_weights_filename)

        train_filename = "c_45_10100_800.dat"
        weights_filename = "cn_45_10100_800.h5"
        prev_weights_filename = "cn_45_10100_800.h5"
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.00001
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 5, prev_weights_filename)

    def test_train_all(self):
        train_filenames = [
            "c_1_500_1600.dat",
            "c_2_500_1600.dat",
            "c_3_500_1600.dat",
            "c_4_600_1600.dat",
            "c_5_600_1600.dat",
            "c_6_600_1600.dat",
            "c_7_800_2000.dat",
            "c_8_600_2000.dat",
            "c_9_800_1600.dat",
            "c_10_800_1200.dat",
            "c_11_800_1600.dat",
            "c_12_800_1600.dat",
            "c_13_800_1600.dat",
            "c_14_400_1600.dat",
            "c_15_800_1600.dat",
            "c_16_800_1600.dat",
            "c_17_1000_1600.dat",
            "c_18_1000_1600.dat",
            "c_19_1200_1600.dat",
            "c_20_1000_1600.dat",
            "c_21_1200_1600.dat",
            "c_22_1200_1600.dat",
            "c_23_1000_1600.dat",
            "c_24_1000_1600.dat",
            "c_25_1000_1600.dat",
            "c_26_1100_1600.dat",
            "c_27_1100_1600.dat",
            "c_28_1100_1600.dat",
            "c_29_1100_1600.dat",
            "c_30_1100_1600.dat"
        ]

        for f in train_filenames:
            if not os.path.isfile(f):
                print(f"file {f} not exists")
                raise Exception()

        trainer_config = TrainerConfig()
        trainer_config.colosus_config.residual_blocks = 4
        trainer = Trainer(trainer_config)

        def get_weights_fname(tf):
            return "xx" + tf.split(".")[0] + ".h5"

        for i in range(0, len(train_filenames)):
            if i == 0:
                prev_weights_filename = None
                trainer_config.colosus_config.lr = 0.0003
                epochs = 10
            elif i < 11:
                prev_weights_filename = get_weights_fname(train_filenames[i - 1])
                trainer_config.colosus_config.lr = 0.0001
                epochs = 10
            elif i < 21:
                prev_weights_filename = get_weights_fname(train_filenames[i - 1])
                trainer_config.colosus_config.lr = 0.00005
                epochs = 10
            else:
                prev_weights_filename = get_weights_fname(train_filenames[i - 1])
                trainer_config.colosus_config.lr = 0.00005
                epochs = 15

            train_filename = train_filenames[i]
            weights_filename = get_weights_fname(train_filename)
            print(f"training with {train_filename}...")
            trainer.train(train_filename, weights_filename, epochs, prev_weights_filename)

        train_filename = train_filenames[-1]
        weights_filename = get_weights_fname(train_filename)
        trainer_config.colosus_config.lr = 0.00002
        trainer.train(train_filename, weights_filename, 15, weights_filename)

    def test_save_rotated_records(self):
        input_filename = "c_1_200_30.dat"
        rotated_filename = "c_1_1600_30.dat"
        recordset = TrainRecordSet.load_from_file(input_filename)
        recordset.do_rotations()
        recordset.save_to_file(rotated_filename)

    def test_merge_records(self):
        merged_filename = "zc_23_2_400_1600.dat"
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
