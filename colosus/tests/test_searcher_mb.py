import unittest
import cProfile, pstats, io

import time

from colosus.config import SearchConfig, StateConfig, ColosusConfig
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
        colosus_config.residual_blocks = 6
        colosus_config.conv_size = 160
        colosus = ColosusModel(colosus_config)
        colosus.build()

        colosus.load_weights("e_07_2000_800.h5")

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
        config.mb_size = 64
        config.max_collisions = 16
        config.noise_factor = 0.0
        config.smart_pruning_factor = 1

        searcher = SearcherMb(config, colosus)

        state_config = StateConfig()

        state = StateMb(pos, None, None, state_config)

        searcher.search(state, 2)

        # pr = cProfile.Profile()
        # pr.enable()

        start = time.time()
        policy, temp_policy, value, move, new_state = searcher.search(state, 0, 5.0)

        print("time: " + str(time.time() - start))

        # pr.disable()
        # s = io.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        state.print_children_stats(20)
        print()
        state.print()
        new_state.print()

        searcher.stats.print()
        print([Square.to_string(m) for m in new_state.principal_variation()])

        print("value: " + str(value))

    def test_search_time_per_move(self):
        colosus_config = ColosusConfig()
        colosus = ColosusModel(colosus_config)
        colosus.build()
        colosus.load_weights("e_0806_2000_800.h5")

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

        start = time.time()
        policy, temp_policy, value, move, new_state = searcher.search(state, 0, time_per_move=1)

        print("nodes " + str(state.N))
        print("time: " + str(time.time() - start))

        state.print()
        new_state.print()

        print("value: " + str(value))

    def test_search_mb_time_per_move(self):
        colosus_config = ColosusConfig()
        # colosus_config.conv_size = 120
        # colosus_config.residual_blocks = 4

        colosus = ColosusModel(colosus_config)
        colosus.build()

        colosus.load_weights("e_0608_2000_800.h5")
        # colosus.load_weights("cpo99345_47_5000_800.h5")

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
        config.mb_size = 64
        config.max_collisions = 16

        searcher = SearcherMb(config, colosus)

        state_config = StateConfig()
        state_config.noise_factor = 0.0
        state = StateMb(pos, None, None, state_config)

        searcher.search(state, 2)

        start = time.time()
        policy, temp_policy, value, move, new_state = searcher.search(state, 0, time_per_move=1)

        searcher.stats.print()
        print("time: " + str(time.time() - start))

        state.print()
        new_state.print()

        print("value: " + str(value))

    def test_time(self):
        start = time.time()

        for i in range(100000):
            time.time()

        print("time: " + str(time.time() - start))


if __name__ == '__main__':
    unittest.main()
