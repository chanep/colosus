import unittest
import cProfile, pstats, io

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
    def test_play_p(self):
        pr = cProfile.Profile()
        pr.enable()

        self.test_play()

        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def test_play(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 3)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 4)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 5)

        start_time = time.time()
        self_play = SelfPlay()
        # self_play.play(150, 800, pos, "t999_2_150_800.dat", "w999_1_1200_800.h5")
        # self_play.play(10, 800, pos, "x.dat", "w_1_1000_800.h5")
        self_play.play(100, 2, pos, "x.dat", None)

        print("fin. time: " + str(time.time() - start_time))

    def test_play_parallel(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 3)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 4)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 5)

        start_time = time.time()
        self_play = SelfPlay()
        # self_play.play(1000, 200, pos, colosus, "t2_1_1000_200.dat")
        self_play.play_parallel(6, 200, pos, "x.dat", 4, None)

        print("fin. time: " + str(time.time() - start_time))

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

    def test_named_params(self):
        print("test named params")
        def method(x, o1=None, o2=None):
            print("x: {}, o1: {}, o2:{}".format(x, o1, o2))

        method(10, 1)
        method(10, o1=1)
        method(10, o2=2)

if __name__ == '__main__':
    unittest.main()