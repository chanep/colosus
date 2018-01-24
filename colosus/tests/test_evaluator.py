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

        evaluator.evalueate(10, 256, pos, None, "c_1_1600_30.h5")


if __name__ == '__main__':
    unittest.main()