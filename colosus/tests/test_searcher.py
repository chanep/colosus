import unittest
import cProfile, pstats, io

import time

from colosus.config import SearchConfig, StateConfig, ColosusConfig
from colosus.searcher_mp import SearcherMp
from colosus.state import State
from colosus.searcher import Searcher
from colosus.colosus_model import ColosusModel
from colosus.game.position import Position
from colosus.game.side import Side


class SearcherTestCase(unittest.TestCase):
    def test_search_p(self):
        pr = cProfile.Profile()
        pr.enable()

        self.test_search()

        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def test_search(self):
        colosus_config = ColosusConfig()
        colosus = ColosusModel(colosus_config)
        colosus.build()
        colosus.load_weights("cpo99345_47_5000_800.h5")

        pos = Position()
        pos.put_piece(Side.BLACK, 7, 7)
        pos.put_piece(Side.WHITE, 8, 8)
        pos.put_piece(Side.BLACK, 12, 12)
        # pos.put_piece(Side.WHITE, 14, 9)
        # pos.put_piece(Side.WHITE, 15, 10)
        pos.switch_side()

        config = SearchConfig()
        searcher = Searcher(config)
        state = State(pos, None, None, colosus, StateConfig())

        searcher.search(state, 10)

        # pr = cProfile.Profile()
        # pr.enable()

        start = time.time()
        policy, temp_policy, value, move, new_state = searcher.search(state, 256)

        print("time: " + str(time.time() - start))

        # pr.disable()
        # s = io.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        state.print()
        new_state.print()

        print("value: " + str(value))

    def test_get_temperature(self):
        config = SearchConfig()
        searcher = Searcher(config)
        temp = searcher._get_temperature(10)
        self.assertEqual(1.0, temp)

        config.temp0 = [[0.3, 0.7], [1.4, 1]]
        for i in range(10):
            temp = searcher._get_temperature(10)
            print(f'temp: {temp}')


    def test_search_mp(self):
        colosus = ColosusModel(ColosusConfig())
        colosus.build()
        colosus.load_weights("c_12_800_1600.h5")

        pos = Position()
        pos.put_piece(Side.BLACK, 7, 7)
        pos.put_piece(Side.WHITE, 8, 8)
        pos.put_piece(Side.BLACK, 11, 11)
        # pos.put_piece(Side.WHITE, 14, 9)
        # pos.put_piece(Side.WHITE, 15, 10)
        pos.switch_side()

        config = SearchConfig()
        config.workers = 8
        searcher = SearcherMp(config)
        state = State(pos, None, None, colosus, StateConfig())

        start = time.time()
        policy, value, move, new_state = searcher.search(state, 800)
        print("time: " + str(time.time() - start))

        state.print()
        new_state.print()

        print("value: " + str(value))


if __name__ == '__main__':
    unittest.main()
