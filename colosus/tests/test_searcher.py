import unittest
import cProfile, pstats, io

import time

from colosus.config import SearchConfig, StateConfig
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
        colosus = ColosusModel()
        colosus.build()
        # colosus.model.load_weights("c_1_1600_30.h5")

        pos = Position()
        pos.put_piece(Side.WHITE, 11, 6)
        pos.put_piece(Side.WHITE, 12, 7)
        # pos.put_piece(Side.WHITE, 13, 8)
        # pos.put_piece(Side.WHITE, 14, 9)
        # pos.put_piece(Side.WHITE, 15, 10)
        # pos.switch_side()

        config = SearchConfig()
        searcher = Searcher(config)
        state = State(pos, None, None, colosus, StateConfig())

        policy, value, move, new_state = searcher.search(state, 2000)

        # state.print()
        # new_state.print()
        #
        # print("value: " + str(value))
        # time.sleep(1)


if __name__ == '__main__':
    unittest.main()
