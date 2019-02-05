import unittest
import cProfile, pstats, io

import numpy as np
import time
import random

from colosus.colosus_model import ColosusModel
from colosus.config import SelfPlayConfig, ColosusConfig, SelfPlayMpConfig
from colosus.game.square import Square
from colosus.self_play import SelfPlay
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.self_play_mp import SelfPlayMp
from colosus.tests.test_train_record_set import TrainRecordSetTestCase
from colosus.tests.test_trainer import TrainerTestCase
from colosus.train_record_set import TrainRecordSet


class SelfPlayTestCase(unittest.TestCase):
    def test_play_p(self):
        colosus = ColosusModel(ColosusConfig())
        colosus.build()
        colosus.load_weights("c_1_2000_256.h5")

        pr = cProfile.Profile()
        pr.enable()

        self.test_play(colosus)

        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def test_play(self, colosus=None):
        pos = Position()

        start_time = time.time()

        config = SelfPlayConfig()
        self_play = SelfPlay(config)

        # self_play.play(200, 30, pos, "c_1_200_30.dat", None)

        self_play.play(1, 256, pos, "x.dat", "c_18_1000_1600.h5", colosus=colosus)

        print("fin. time: " + str(time.time() - start_time))

    def test_play_parallel(self):
        pos = Position()

        start_time = time.time()

        config = SelfPlayConfig()
        self_play = SelfPlay(config)
        # self_play.play(1000, 200, pos, colosus, "t2_1_1000_200.dat")
        # self_play.play_parallel(500, 300, pos, "c_7_500_300.dat", 4, "c_6_400_400.h5")
        self_play.play_parallel(200, 400, pos, "c_1_600_1600.dat", 8, "c_1_500_1600.h5")

        print("fin. time: " + str(time.time() - start_time))

    def test_play_mp(self):
        # TrainerTestCase().test_train_multi()

        pos = Position()
        start_time = time.time()
        config = SelfPlayMpConfig()

        config.search_config.move_count_temp0 = 24
        config.search_config.temp0 = 1.18
        config.search_config.tempf = 0
        config.state_config.policy_offset = -0.99 / 800
        self_play = SelfPlayMp(config)

        train_filename = "d_19_2000_800.dat"
        train_filename_a = "d_19a_2000_800.dat"
        train_filename_b = "d_19b_2000_800.dat"
        weights_filename = "d_18_2000_800.h5"

        self_play.play(1000, 800, pos, train_filename_a, 24, weights_filename)
        TrainRecordSet.merge_and_rotate(train_filename_a, 24)

        self_play.play(1000, 800, pos, train_filename_b, 24, weights_filename)
        TrainRecordSet.merge_and_rotate(train_filename_b, 24)

        recordset = TrainRecordSet()

        r = TrainRecordSet.load_from_file(train_filename_a)
        recordset.extend(r.records)
        r = TrainRecordSet.load_from_file(train_filename_b)
        recordset.extend(r.records)
        random.shuffle(recordset.records)

        recordset.save_to_file(train_filename)

        print("fin. time: " + str(time.time() - start_time))

        total, different, duplicated = TrainRecordSet.duplications(train_filename, 0.9)
        print("final positions:")
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        TrainerTestCase().test_train()


        # TrainRecordSetTestCase().test_merge()
        #
        # total, different, duplicated = TrainRecordSet.duplications("d_9_4000_800.dat", 0.9)
        # print("final positions:")
        # print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))
        #
        # TrainerTestCase().test_train_multi()
        # print("Revisar duplicados!!!")


if __name__ == '__main__':
    unittest.main()