import unittest
import cProfile, pstats, io

import numpy as np
import time

from colosus.colosus_model import ColosusModel
from colosus.config import SelfPlayConfig, ColosusConfig
from colosus.game.square import Square
from colosus.self_play import SelfPlay
from colosus.game.position import Position
from colosus.game.side import Side


class SelfPlayTestCase(unittest.TestCase):
    def test_play_p(self):
        colosus = ColosusModel(ColosusConfig())
        colosus.build()

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

        self_play.play(10, 30, pos, "x.dat", None, colosus=colosus)

        print("fin. time: " + str(time.time() - start_time))

    def test_play_parallel(self):
        pos = Position()

        start_time = time.time()

        config = SelfPlayConfig()
        self_play = SelfPlay(config)
        # self_play.play(1000, 200, pos, colosus, "t2_1_1000_200.dat")
        self_play.play_parallel(20, 30, pos, "x.dat", 4, None)

        print("fin. time: " + str(time.time() - start_time))

if __name__ == '__main__':
    unittest.main()