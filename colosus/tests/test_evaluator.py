import unittest
import numpy as np
import time

from colosus.config import EvaluatorConfig
from colosus.evaluator import Evaluator
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.game.piece import Piece


class EvaluatorTestCase(unittest.TestCase):

    def test_evaluate(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 3)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 4)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 5)

        config = EvaluatorConfig()
        evaluator = Evaluator(config)

        evaluator.evalueate(100, 100, pos, "wpp_2_1200_800.h5", "wz_2_1200_800.h5")


if __name__ == '__main__':
    unittest.main()