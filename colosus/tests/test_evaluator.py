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

        evaluator.evaluate(1000, 1, pos, "c_16_800_1600.h5", "c_16_400_1600.h5")


if __name__ == '__main__':
    unittest.main()