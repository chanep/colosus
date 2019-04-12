import unittest
import cProfile, pstats, io

import numpy as np
import time
import random

from colosus.colosus_model import ColosusModel
from colosus.config import SelfPlayConfig, ColosusConfig
from colosus.game.square import Square
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.self_play_mp import SelfPlayMp
from colosus.self_play_mp_mb import SelfPlayMpMb
from colosus.tests.test_train_record_set import TrainRecordSetTestCase
from colosus.tests.test_trainer import TrainerTestCase
from colosus.train_record_set import TrainRecordSet


class SelfPlayTestCase(unittest.TestCase):
    def test_play_mp(self):
        # TrainerTestCase().test_train_multi()

        pos = Position()
        start_time = time.time()
        config = SelfPlayConfig()

        config.search_config.tempf = 0.35
        config.state_config.policy_offset = -0.5
        self_play = SelfPlayMp(config)

        train_filename = "e_01_2000_800.dat"
        train_filename_a = "e_01a_2000_800.dat"
        train_filename_b = "e_01b_2000_800.dat"
        train_filename_c = "e_01c_2000_800.dat"
        train_filename_d = "e_01d_2000_800.dat"
        weights_filename = "d_4953_2000_800_bignn.h5"

        config.search_config.move_count_temp0 = 24
        config.search_config.temp0 = 1.4
        self_play.play(500, 800, pos, train_filename_a, 24, weights_filename)
        time.sleep(5)
        TrainRecordSet.merge_and_rotate(train_filename_a, 24)

        config.search_config.move_count_temp0 = 26
        config.search_config.temp0 = 1.25
        self_play.play(500, 800, pos, train_filename_b, 24, weights_filename)
        time.sleep(5)
        TrainRecordSet.merge_and_rotate(train_filename_b, 24)

        config.search_config.move_count_temp0 = 30
        config.search_config.temp0 = 1.1
        self_play.play(500, 800, pos, train_filename_c, 24, weights_filename)
        time.sleep(5)
        TrainRecordSet.merge_and_rotate(train_filename_c, 24)

        config.search_config.move_count_temp0 = 40
        config.search_config.temp0 = 0.8
        self_play.play(500, 800, pos, train_filename_d, 24, weights_filename)
        time.sleep(5)
        TrainRecordSet.merge_and_rotate(train_filename_d, 24)

        recordset = TrainRecordSet()

        r = TrainRecordSet.load_from_file(train_filename_a)
        recordset.extend(r.records)
        r = TrainRecordSet.load_from_file(train_filename_b)
        recordset.extend(r.records)
        r = TrainRecordSet.load_from_file(train_filename_c)
        recordset.extend(r.records)
        r = TrainRecordSet.load_from_file(train_filename_d)
        recordset.extend(r.records)
        random.shuffle(recordset.records)
        recordset.save_to_file(train_filename)

        print("fin. time: " + str(time.time() - start_time))

        total, different, duplicated = TrainRecordSet.duplications(train_filename_a, 0.9)
        print("final positions: " + train_filename_a)
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        total, different, duplicated = TrainRecordSet.duplications(train_filename_b, 0.9)
        print("final positions: " + train_filename_b)
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        total, different, duplicated = TrainRecordSet.duplications(train_filename_c, 0.9)
        print("final positions: " + train_filename_c)
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        total, different, duplicated = TrainRecordSet.duplications(train_filename_d, 0.9)
        print("final positions: " + train_filename_d)
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        total, different, duplicated = TrainRecordSet.duplications(train_filename, 0.9)
        print("final positions:")
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        TrainerTestCase().test_train_clr()

    def test_play_mp_mb(self):
        pos = Position()
        start_time = time.time()
        config = SelfPlayConfig()

        config.search_config.tempf = 0.3
        config.state_config.policy_offset = -1

        self_play = SelfPlayMpMb(config)

        train_filename = "e_17_2000_800.dat"
        train_filename_a = "e_17a_2000_800.dat"
        train_filename_b = "e_17b_2000_800.dat"
        train_filename_c = "e_17c_2000_800.dat"
        train_filename_d = "e_17d_2000_800.dat"
        weights_filename = "e_16_2000_800.h5"

        config.z_factor = 0

        config.search_config.move_count_temp0 = 24
        config.search_config.temp0 = 1.5
        config.state_config.play_policy_offset = -3
        self_play.play(500, 800, pos, train_filename_a, 11, weights_filename)
        time.sleep(5)
        TrainRecordSet.merge_and_rotate(train_filename_a, 11)

        config.search_config.move_count_temp0 = 26
        config.search_config.temp0 = 1.30
        config.state_config.play_policy_offset = -3
        self_play.play(500, 800, pos, train_filename_b, 11, weights_filename)
        time.sleep(5)
        TrainRecordSet.merge_and_rotate(train_filename_b, 11)

        config.search_config.move_count_temp0 = 30
        config.search_config.temp0 = 1.15
        config.state_config.play_policy_offset = -3
        self_play.play(500, 800, pos, train_filename_c, 11, weights_filename)
        time.sleep(5)
        TrainRecordSet.merge_and_rotate(train_filename_c, 11)

        config.z_factor = 0.25

        config.search_config.move_count_temp0 = 40
        config.search_config.temp0 = 0.9
        config.state_config.play_policy_offset = -3
        self_play.play(500, 800, pos, train_filename_d, 11, weights_filename)
        time.sleep(5)
        TrainRecordSet.merge_and_rotate(train_filename_d, 11)

        recordset = TrainRecordSet()

        r = TrainRecordSet.load_from_file(train_filename_a)
        recordset.extend(r.records)
        r = TrainRecordSet.load_from_file(train_filename_b)
        recordset.extend(r.records)
        r = TrainRecordSet.load_from_file(train_filename_c)
        recordset.extend(r.records)
        r = TrainRecordSet.load_from_file(train_filename_d)
        recordset.extend(r.records)
        random.shuffle(recordset.records)
        recordset.save_to_file(train_filename)

        print("fin. time: " + str(time.time() - start_time))

        total, different, duplicated = TrainRecordSet.duplications(train_filename_a, 0.9)
        print("final positions: " + train_filename_a)
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        total, different, duplicated = TrainRecordSet.duplications(train_filename_b, 0.9)
        print("final positions: " + train_filename_b)
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        total, different, duplicated = TrainRecordSet.duplications(train_filename_c, 0.9)
        print("final positions: " + train_filename_c)
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        total, different, duplicated = TrainRecordSet.duplications(train_filename_d, 0.9)
        print("final positions: " + train_filename_d)
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        total, different, duplicated = TrainRecordSet.duplications(train_filename, 0.9)
        print("final positions:")
        print("total: {}, different: {}, duplicated: {}".format(total, different, duplicated))

        TrainerTestCase().test_train_clr()

    def test_play_mp_1(self):
        # TrainerTestCase().test_train_multi()

        pos = Position()
        start_time = time.time()
        config = SelfPlayConfig()

        config.search_config.tempf = 0.35
        config.state_config.policy_offset = -0.5
        self_play = SelfPlayMp(config)

        train_filename = "mp1.dat"
        weights_filename = "e_01_2000_800.h5"

        config.search_config.move_count_temp0 = 30
        config.search_config.temp0 = 1.15
        self_play.play(192, 64, pos, train_filename, 24, weights_filename)

        print("fin. time: " + str(time.time() - start_time))

    def test_play_mp_mb_1(self):
        pos = Position()
        start_time = time.time()
        config = SelfPlayConfig()

        config.search_config.tempf = 0.35
        config.state_config.policy_offset = -0.5
        config.search_config.mb_size = 16
        config.search_config.max_collisions = 1

        self_play = SelfPlayMpMb(config)

        train_filename = "mpmb1.dat"
        weights_filename = "e_01_2000_800.h5"

        config.search_config.move_count_temp0 = 30
        config.search_config.temp0 = 1.15
        self_play.play(192, 64, pos, train_filename, 11, weights_filename)

        print("fin. time: " + str(time.time() - start_time))


if __name__ == '__main__':
    unittest.main()