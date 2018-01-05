import unittest

import datetime
import tensorflow as tf
import numpy as np
from ..colosus_model import ColosusModel
from ..game.position import Position
from ..game.move import Move
from ..game.square import Square
from ..game.side import Side
from ..game.piece import Piece
from tensorflow.python.keras import backend as K


class PositionTestCase(unittest.TestCase):
    def test_evaluate(self):
        colosus = ColosusModel()
        colosus.build()

        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 0, 5)
        pos.put_piece(Side.WHITE, Piece.ROOK, 0, 0)
        pos.put_piece(Side.BLACK, Piece.KING, 7, 5)

        t1 = datetime.datetime.now()
        board = tf.cast(pos.board, tf.float32)
        t_board = tf.transpose(board, perm=[1, 2, 0])
        print("board shape: {}".format(board.shape))
        input = tf.expand_dims(t_board, axis=0)
        print("input shape: {}".format(input.shape))
        output = colosus.model(input)
        policy = output[0]
        value = output[1]

        t2 = datetime.datetime.now()
        d1 = t2 - t1
        print("milisec:" + str(d1.microseconds // 1000))

        print("value")
        print(K.eval(value))

        t3 = datetime.datetime.now()
        d2 = t3 - t2
        print("milisec:" + str(d2.microseconds // 1000))

        # print("policy")
        # print(K.eval(policy))

if __name__ == '__main__':
    unittest.main()