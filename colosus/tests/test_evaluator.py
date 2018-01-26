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
        # config.state_config.noise_factor = 0.25
        evaluator = Evaluator(config)

        evaluator.evaluate(20, 512, pos, "c_2_1600_400.h5", "c_2_2480_256.h5")


if __name__ == '__main__':
    unittest.main()