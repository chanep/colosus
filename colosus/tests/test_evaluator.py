import unittest
import numpy as np
import time

from colosus.config import EvaluatorConfig
from colosus.evaluator import Evaluator
from colosus.game.position import Position
from colosus.game.side import Side


class EvaluatorTestCase(unittest.TestCase):

    def test_evaluate(self):
        pos = Position()
        config = EvaluatorConfig()
        config.player_config.search_config.move_count_temp0 = 22
        config.player2_config.search_config.move_count_temp0 = 22

        config.player_config.search_config.temp0 = 0.5
        config.player2_config.search_config.temp0 = 0.5
        config.player_config.state_config.policy_offset = 0
        config.player2_config.state_config.policy_offset = 0
        evaluator = Evaluator(config)
        evaluator.evaluate(200, 256, pos, "xxc_27_1100_1600.h5", "cpo99345_47_5000_800.h5")
        # evaluator.evaluate(200, 256, pos, "cpo99_46_5000_800.h5", "cpo99345_47_5000_800.h5")

    def test_evaluate2(self):
        pos = Position()

        config = EvaluatorConfig()
        evaluator = Evaluator(config)

        # evaluator.evaluate(200, 512, pos, "xres.h5", "c_17_400_3200.h5")
        evaluator.evaluate(100, 1600, pos, "c_17_400_3200.h5", "c_17_400_3200.h5", 1)

    def test_evaluate_mp(self):
        pos = Position()

        config = EvaluatorConfig()
        config.player2_is_mp = True
        evaluator = Evaluator(config)

        evaluator.evaluate(100, 256, pos, "c_17_400_3200.h5", "c_17_400_3200.h5")


if __name__ == '__main__':
    unittest.main()