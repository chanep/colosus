import unittest

import datetime
import tensorflow as tf
import numpy as np

from colosus.self_play import SelfPlay
from ..state import State
from ..searcher import Searcher
from ..colosus_model import ColosusModel
from ..game.position import Position
from ..game.move import Move
from ..game.square import Square
from ..game.side import Side
from ..game.piece import Piece


class SelfPlayTestCase(unittest.TestCase):
    def test_play(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 3)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 4)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 5)

        self_play = SelfPlay()
        self_play.play(10, 200, pos)

        print("fin")


if __name__ == '__main__':
    unittest.main()