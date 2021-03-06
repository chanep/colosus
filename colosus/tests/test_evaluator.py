import unittest
import numpy as np
import time

from colosus.config import EvaluatorConfig, PlayerType
from colosus.evaluator import Evaluator
from colosus.game.position import Position
from colosus.game.side import Side


class EvaluatorTestCase(unittest.TestCase):

    def test_evaluate(self):
        pos = Position()
        config = EvaluatorConfig()
        config.player_config.search_config.move_count_temp0 = 30
        config.player2_config.search_config.move_count_temp0 = 30

        config.player_config.search_config.tempf = 0.3
        config.player2_config.search_config.tempf = 0.3
        config.player_config.search_config.temp0 = 0.8
        config.player2_config.search_config.temp0 = 0.8
        config.player_config.state_config.policy_offset = -1
        config.player2_config.state_config.policy_offset = -1

        # config.colosus2_config.residual_blocks = 4
        # config.colosus2_config.conv_size = 120
        config.colosus2_config.policy_conv_size = 32

        config.player_type = PlayerType.player_mb
        config.player2_type = PlayerType.player_mb

        evaluator = Evaluator(config)

        evaluator.evaluate(400, 1, pos, "e_16_2000_800.h5", "e_14_2000_800.h5", times_per_move=1)
        # evaluator.evaluate(400, 1, pos, "e_01_2000_800.h5", "cpo99345_47_5000_800.h5")

    def test_evaluate2(self):
        pos = Position()
        config = EvaluatorConfig()
        config.player_config.search_config.move_count_temp0 = 30
        config.player2_config.search_config.move_count_temp0 = 30

        config.player_config.search_config.tempf = 0.3
        config.player2_config.search_config.tempf = 0.3
        config.player_config.search_config.temp0 = 0.8
        config.player2_config.search_config.temp0 = 0.8
        config.player_config.state_config.policy_offset = -1
        config.player2_config.state_config.policy_offset = -1

        config.colosus2_config.residual_blocks = 4
        config.colosus2_config.conv_size = 120
        config.colosus2_config.policy_conv_size = 32

        config.player_type = PlayerType.player_mb
        config.player2_type = PlayerType.player

        evaluator = Evaluator(config)

        evaluator.evaluate(200, 0, pos, "e_16_2000_800.h5", "cpo99345_47_5000_800.h5", times_per_move=1)
        # evaluator.evaluate(400, 1, pos, "e_01_2000_800.h5", "cpo99345_47_5000_800.h5")

    def test_evaluate_mp(self):
        pos = Position()

        config = EvaluatorConfig()
        config.player2_is_mp = True
        evaluator = Evaluator(config)

        evaluator.evaluate(100, 256, pos, "c_17_400_3200.h5", "c_17_400_3200.h5")


if __name__ == '__main__':
    unittest.main()