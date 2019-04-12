import unittest
import numpy as np
import cProfile, pstats, io

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
        print("training e_10_2000_800...")
        train_filename = "e_10_2000_800.dat"
        weights_filename = "e_10_2000_800_noclr.h5"
        prev_weights_filename = "e_09_2000_800.h5"
        trainer_config = TrainerConfig()
        trainer = Trainer(trainer_config)
        trainer_config.colosus_config.lr = 0.00003
        trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        trainer_config.colosus_config.lr = 0.00001
        trainer.train(train_filename, weights_filename, 1, weights_filename)

    def test_train_clr(self):
        print("training e_1617_2000_800...")
        train_filename = "e_1617_2000_800.dat"
        weights_filename = "e_1617_2000_800.h5"
        prev_weights_filename = "e_16_2000_800.h5"
        trainer_config = TrainerConfig()
        trainer = Trainer(trainer_config)
        # trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.0003, 0.0015, 500)
        trainer.train_clr(train_filename, weights_filename, 2, prev_weights_filename, 0.00005, 0.0003, 500)
        prev_weights_filename = weights_filename
        # trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.00005, 0.0003, 500)
        trainer.train_clr(train_filename, weights_filename, 2, prev_weights_filename, 0.00001, 0.00005, 500)

    def test_train_clr2(self):
        print("training e_17_2000_800...")
        train_filename = "e_17_2000_800.dat"
        weights_filename = "e_17_2000_800_.h5"
        prev_weights_filename = "e_17_2000_800.h5"
        trainer_config = TrainerConfig()
        trainer = Trainer(trainer_config)
        # trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.0003, 0.0015, 500)
        trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.000007, 0.00003, 500)


    def test_train_clr_nn(self):
        print("training e_15_2000_800_ppo...")
        train_filename = "e_15_2000_800_ppo.dat"
        weights_filename = "x32.h5"
        prev_weights_filename = None
        trainer_config = TrainerConfig()
        trainer = Trainer(trainer_config)

        trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.0002, 0.001, 500)

        trainer_config.colosus_config.policy_conv_size = 64
        weights_filename = "x64.h5"
        trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.0002, 0.001, 500)

    def test_train_clr_bignn(self):
        print("training e_15_2000_800_ppo...")
        train_filename = "e_15_2000_800_ppo.dat"

        trainer_config = TrainerConfig()
        trainer = Trainer(trainer_config)

        weights_filename = "e_15_2000_800_ppo_64.h5"
        prev_weights_filename = None
        trainer_config.colosus_config.lr = 0.001
        trainer_config.colosus_config.policy_conv_size = 64
        trainer.train(train_filename, weights_filename, 3, prev_weights_filename)
        prev_weights_filename = weights_filename
        print("0.0002 - 0.001")
        trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.0002, 0.001, 500)
        trainer.train_clr(train_filename, weights_filename, 2, prev_weights_filename, 0.0002, 0.001, 1000)
        print("0.00004 - 0.0002")
        trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.00004, 0.0002, 500)
        trainer.train_clr(train_filename, weights_filename, 2, prev_weights_filename, 0.00004, 0.0002, 1000)
        print("0.00001 - 0.00004")
        trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.00001, 0.00004, 500)
        trainer.train_clr(train_filename, weights_filename, 2, prev_weights_filename, 0.00001, 0.00004, 1000)


        # print("training 120-6_GN.h5")
        # weights_filename = "120-6_Bren.h5"
        # prev_weights_filename = None
        # trainer_config.colosus_config.lr = 0.001
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # prev_weights_filename = weights_filename
        # trainer_config.colosus_config.lr = 0.0003
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # trainer_config.colosus_config.lr = 0.0001
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # trainer_config.colosus_config.lr = 0.00003
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # trainer_config.colosus_config.lr = 0.00001
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)


        # print("training 160-6")
        # weights_filename = "160_6.h5"
        # prev_weights_filename = None
        # trainer_config.colosus_config.lr = 0.001
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # prev_weights_filename = weights_filename
        # trainer_config.colosus_config.lr = 0.0003
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # trainer_config.colosus_config.lr = 0.0001
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # trainer_config.colosus_config.lr = 0.00003
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # trainer_config.colosus_config.lr = 0.00001
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)

        # weights_filename = "160_6_2.h5"
        # trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.0002, 0.001, 1000)
        # prev_weights_filename = weights_filename
        # trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.00006, 0.0003, 1000)
        # trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.00002, 0.0001, 1000)
        # trainer.train_clr(train_filename, weights_filename, 1, prev_weights_filename, 0.000006, 0.00003, 1000)

        # print("training 160-4")
        # weights_filename = "160_4.h5"
        # prev_weights_filename = None
        # trainer_config.colosus_config.conv_size = 160
        # trainer_config.colosus_config.residual_blocks = 4
        # trainer_config.colosus_config.lr = 0.001
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # prev_weights_filename = weights_filename
        # trainer_config.colosus_config.lr = 0.0003
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # trainer_config.colosus_config.lr = 0.0001
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # trainer_config.colosus_config.lr = 0.00003
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)
        # trainer_config.colosus_config.lr = 0.00001
        # trainer.train(train_filename, weights_filename, 2, prev_weights_filename)


    def test_train_noclr(self):
        epochs = 3
        train_filename = "e_0608_2000_800.dat"
        weights_filename = "e_0608_2000_800_120-6.h5"
        prev_weights_filename = None
        trainer_config = TrainerConfig()
        trainer = Trainer(trainer_config)

        trainer_config.colosus_config.conv_size = 120
        trainer_config.colosus_config.residual_blocks = 6

        lrs = [0.001, 0.0003, 0.0001, 0.00003, 0.00001]

        for lr in lrs:
            print("training " + str(lr))
            trainer_config.colosus_config.lr = lr
            for ep in range(epochs):
                print("epoch " + str(ep + 1) + "/" + str(epochs))
                trainer.train(train_filename, weights_filename, 1, prev_weights_filename)
                prev_weights_filename = weights_filename

    def test_train2(self):
        epochs = 2
        train_filename = "d_4750_2000_800.dat"
        weights_filename = "d_4750_2000_800_8res.h5"
        prev_weights_filename = "d_4750_2000_800_8res.h5"
        trainer_config = TrainerConfig()
        trainer = Trainer(trainer_config)
        trainer_config.colosus_config.conv_size = 240

        # lrs = [0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        # prev = None
        lrs = [0.00003, 0.00001]
        prev = prev_weights_filename

        for lr in lrs:
            print("training " + str(lr))
            trainer_config.colosus_config.lr = lr
            for ep in range(epochs):
                print("epoch " + str(ep + 1) + "/" + str(epochs))
                trainer.train(train_filename, weights_filename, 1, prev)
                prev = prev_weights_filename

    def test_train3(self):
        print("training d_50_100_800...")
        train_filename = "d_50_100_800.dat"
        weights_filename = "d_50_100_800.h5"
        prev_weights_filename = None
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.0001
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 1, prev_weights_filename)


    def test_train_generator(self):
        print("training d_15_20_800...")
        train_filename = "mp1_0.dat"
        weights_filename = "xg.h5"
        prev_weights_filename = None
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.003
        trainer = Trainer(trainer_config)

        pr = cProfile.Profile()
        pr.enable()

        trainer.train(train_filename, weights_filename, 1, prev_weights_filename)

        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def test_train_multi(self):
        print("training d_9_4000_800.h5...")
        train_filename = "d_9_4000_800.dat"
        weights_filename = "d_9_4000_800.h5"
        prev_weights_filename = "d_8_4000_800.h5"
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.003
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 3, prev_weights_filename)

        print("training d_9_4000_800_sinprev.h5...")
        train_filename = "d_9_4000_800.dat"
        weights_filename = "d_9_4000_800_sinprev.h5"
        prev_weights_filename = None
        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.003
        trainer = Trainer(trainer_config)
        trainer.train(train_filename, weights_filename, 3, prev_weights_filename)

        # EvaluatorTestCase().test_evaluate()

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
