import unittest
import cProfile, pstats, io
from multiprocessing import Process, Lock, Queue, Manager

import numpy as np
import time

from colosus.colosus_model import ColosusModel
from colosus.config import SelfPlayConfig, ColosusConfig
from colosus.game.square import Square
from colosus.self_play import SelfPlay
from colosus.game.position import Position
from colosus.game.side import Side


def fun(q: Queue, t):
    for i in range(3):
        print("thread " + str(t) + " before put")
        q.put("{} de thread {}".format(i, t))
        print("thread " + str(t) + " after put")
        time.sleep(1)


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

        self_play.play(1, 256, pos, "x.dat", None, colosus=colosus)

        print("fin. time: " + str(time.time() - start_time))

    def test_play_parallel(self):
        pos = Position()

        start_time = time.time()

        config = SelfPlayConfig()
        self_play = SelfPlay(config)
        # self_play.play(1000, 200, pos, colosus, "t2_1_1000_200.dat")
        self_play.play_parallel(200, 400, pos, "c_2_200_400.dat", 4, "c_1_2000_256.h5")

        print("fin. time: " + str(time.time() - start_time))


    def test_process_play_parallel(self):
        pos = Position()

        start_time = time.time()

        config = SelfPlayConfig()
        self_play = SelfPlay(config)
        # self_play.play(1000, 200, pos, colosus, "t2_1_1000_200.dat")
        self_play.process_play_parallel(20, 30, pos, "x.dat", 4, None)

        print("fin. time: " + str(time.time() - start_time))


    def test_processes(self):

        ps = []
        m = Manager()
        q = m.Queue()

        for t in range(3):
            p = Process(target=fun, args=(q, t))
            p.start()
            ps.append(p)

        print("ini get main")

        for i in range(20):
            print("main before get " + str(t))
            print(q.get())
            print("main after get " + str(t))
            time.sleep(1)


        print("fin main")


if __name__ == '__main__':
    unittest.main()