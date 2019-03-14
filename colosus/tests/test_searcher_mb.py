import unittest
import cProfile, pstats, io

import time

from colosus.config import SearchConfig, SearchMbConfig, StateConfig, ColosusConfig
from colosus.game.square import Square
from colosus.state import State
from colosus.state_mb import StateMb
from colosus.searcher import Searcher
from colosus.searcher_mb import SearcherMb
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
        colosus.load_weights("d_53_2000_800.h5")

        pos = Position()
        pos.put_piece(Side.BLACK, 7, 7)
        pos.put_piece(Side.WHITE, 8, 8)
        pos.put_piece(Side.BLACK, 12, 12)

        # pos.put_piece(Side.WHITE, 7, 8)
        # pos.put_piece(Side.BLACK, 1, 2)
        # pos.put_piece(Side.WHITE, 6, 8)
        # pos.put_piece(Side.BLACK, 4, 9)

        pos.switch_side()

        config = SearchConfig()
        config.temp0 = 0
        searcher = Searcher(config)

        state_config = StateConfig()
        state_config.noise_factor = 0.0
        state = State(pos, None, None, colosus, state_config)

        searcher.search(state, 2)

        pr = cProfile.Profile()
        pr.enable()

        start = time.time()
        policy, temp_policy, value, move, new_state = searcher.search(state, 512)

        print("time: " + str(time.time() - start))

        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        state.print()
        new_state.print()

        print("value: " + str(value))

    def test_search_mb(self):
        colosus_config = ColosusConfig()
        colosus = ColosusModel(colosus_config)
        colosus.build()

        colosus.load_weights("d_53_2000_800.h5")

        pos = Position()
        pos.put_piece(Side.BLACK, 7, 7)
        pos.put_piece(Side.WHITE, 8, 8)
        pos.put_piece(Side.BLACK, 12, 12)

        # pos.put_piece(Side.WHITE, 7, 8)
        # pos.put_piece(Side.BLACK, 1, 2)
        # pos.put_piece(Side.WHITE, 6, 8)
        # pos.put_piece(Side.BLACK, 4, 9)


        pos.switch_side()

        config = SearchMbConfig()
        config.temp0 = 0
        config.mb_size = 64
        config.max_collisions = 8

        searcher = SearcherMb(config, colosus)

        state_config = StateConfig()
        state_config.noise_factor = 0.0
        state = StateMb(pos, None, None, state_config)

        searcher.search(state, 2)

        pr = cProfile.Profile()
        pr.enable()

        start = time.time()
        policy, temp_policy, value, move, new_state = searcher.search(state, 512)

        print("time: " + str(time.time() - start))

        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        state.print()
        new_state.print()

        searcher.stats.print()
        print([Square.to_string(m) for m in new_state.principal_variation()])

        print("value: " + str(value))




if __name__ == '__main__':
    unittest.main()
