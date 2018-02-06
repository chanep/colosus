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

        evaluator.evaluate(20, 256, pos, "c_8_600_2000.h5", "c_9_800_1600.h5")


if __name__ == '__main__':
    unittest.main()