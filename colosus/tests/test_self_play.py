import unittest

import numpy as np
import time

from colosus.colosus_model import ColosusModel
from colosus.game.move import Move
from colosus.game.square import Square
from colosus.self_play import SelfPlay
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.game.piece import Piece


class SelfPlayTestCase(unittest.TestCase):
    def test_play(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 3)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 4)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 5)

        colosus = ColosusModel()
        colosus.build()
        # colosus.model.load_weights("weights1.h5")

        self_play = SelfPlay()
        self_play.play(100, 200, pos, colosus, "x.dat")

        print("fin")

    def test_play2(self):
        pos = Position()
        # pos.put_piece(Side.WHITE, Piece.KING, 4, 3)
        # pos.put_piece(Side.WHITE, Piece.ROOK, 4, 4)
        # pos.put_piece(Side.BLACK, Piece.KING, 5, 5)
        pos.put_piece(Side.WHITE, Piece.KING, 5, 5)
        pos.put_piece(Side.WHITE, Piece.ROOK, 2, 0)
        pos.put_piece(Side.BLACK, Piece.KING, 7, 5)

        colosus = ColosusModel()
        colosus.build()
        colosus.model.load_weights("weights1.h5")

        pos.print()
        policy, value = colosus.predict(pos)
        print(policy)
        move = np.random.choice(len(policy), 1, p=policy)[0]
        print(Move.to_string(move))
        print(value)


if __name__ == '__main__':
    unittest.main()