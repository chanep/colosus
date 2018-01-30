import unittest
import cProfile, pstats, io

import numpy as np
import time

from colosus.colosus_model import ColosusModel
from colosus.config import SelfPlayConfig, ColosusConfig, SelfPlayMpConfig
from colosus.game.square import Square
from colosus.self_play import SelfPlay
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.self_play_mp import SelfPlayMp


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

        self_play.play(1, 800, pos, "x.dat", "c_4_310_300.h5", colosus=colosus)

        print("fin. time: " + str(time.time() - start_time))

    def test_play_parallel(self):
        pos = Position()

        start_time = time.time()

        config = SelfPlayConfig()
        self_play = SelfPlay(config)
        # self_play.play(1000, 200, pos, colosus, "t2_1_1000_200.dat")
        # self_play.play_parallel(500, 300, pos, "c_7_500_300.dat", 4, "c_6_400_400.h5")
        self_play.play_parallel(500, 400, pos, "c_8_500_400.dat", 8, "c_7_500_300.h5")

        print("fin. time: " + str(time.time() - start_time))

    def test_play_mp(self):
        pos = Position()

        start_time = time.time()

        config = SelfPlayMpConfig()
        self_play = SelfPlayMp(config)
        # self_play.play(1000, 200, pos, colosus, "t2_1_1000_200.dat")
        self_play.play(500, 300, pos, "c_9_500_300.dat", 16, "c_8_500_400.h5")

        print("fin. time: " + str(time.time() - start_time))


if __name__ == '__main__':
    unittest.main()