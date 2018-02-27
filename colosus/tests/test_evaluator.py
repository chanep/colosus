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
        evaluator = Evaluator(config)

        # evaluator.evaluate(200, 512, pos, "c_18_1000_1600.h5", "c_17_400_3200.h5")
        evaluator.evaluate(500, 1, pos, "c_18_1000_1600.h5", "c_18_1000_1600.h5")

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