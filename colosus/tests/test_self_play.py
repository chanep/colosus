import unittest
import cProfile, pstats, io

import numpy as np
import time

from colosus.colosus_model import ColosusModel
from colosus.config import SelfPlayConfig
from colosus.game.square import Square
from colosus.self_play import SelfPlay
from colosus.game.position import Position
from colosus.game.side import Side


class SelfPlayTestCase(unittest.TestCase):
    def test_play_p(self):
        pr = cProfile.Profile()
        pr.enable()

        self.test_play()

        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def test_play(self):
        pos = Position()

        start_time = time.time()

        config = SelfPlayConfig()
        self_play = SelfPlay(config)

        self_play.play(1, 400, pos, "x.dat", None)

        # self_play.play(200, 800, pos, "c_1_200_800.dat", None)

        print("fin. time: " + str(time.time() - start_time))


if __name__ == '__main__':
    unittest.main()