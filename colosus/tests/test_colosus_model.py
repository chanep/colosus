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
        board = tf.cast(pos.board, tf.float32)
        t_board = tf.transpose(board, perm=[1, 2, 0])
        input = tf.expand_dims(t_board, axis=0)

        pos2 = Position()
        pos2.put_piece(Side.WHITE, Piece.KING, 0, 5)
        pos2.put_piece(Side.WHITE, Piece.ROOK, 6, 7)
        pos2.put_piece(Side.BLACK, Piece.KING, 7, 5)
        board2 = tf.cast(pos2.board, tf.float32)
        t_board2 = tf.transpose(board2, perm=[1, 2, 0])
        input2 = tf.expand_dims(t_board2, axis=0)


        output = colosus.model(input)
        value = output[1]

        output2 = colosus.model(input)
        value2 = output2[1]

        output3 = colosus.model(input2)
        value3 = output3[1]

        board4 = pos.board
        t_board4 = np.transpose(board4, [1, 2, 0])
        board5 = pos2.board
        t_board5 = np.transpose(board5, [1, 2, 0])

        input4 = np.expand_dims(t_board4, axis=0)
        print("shape")
        print(input4.shape)
        input5 = np.append(input4, np.expand_dims(t_board5, axis=0), axis=0)
        print(input5.shape)

        t3 = datetime.datetime.now()
        x = colosus.model.predict_on_batch(input4)
        t4 = datetime.datetime.now()
        p = x[0]
        v = x[1]
        print(v)

        t1 = datetime.datetime.now()
        print(K.eval(value))
        t2 = datetime.datetime.now()

        d1 = t2 - t1
        d2 = t4 - t3

        print(d1.microseconds)
        print(d2.microseconds)

    def test_predict(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 0, 5)
        pos.put_piece(Side.WHITE, Piece.ROOK, 0, 0)
        pos.put_piece(Side.BLACK, Piece.KING, 7, 5)

        colosus = ColosusModel()
        colosus.build()
        policy, value = colosus.predict(pos.board)
        print(value)
        print(policy)
        print(policy.sum())

    def test_train(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 0, 5)
        pos.put_piece(Side.WHITE, Piece.ROOK, 0, 0)
        pos.put_piece(Side.BLACK, Piece.KING, 7, 5)

        colosus = ColosusModel()
        colosus.build()

        policy, value = colosus.predict(pos.board)
        print(value)
        print(policy)

        boards = [pos.board]
        boards = map(lambda b: np.transpose(b, [1, 2, 0]), boards)
        boards = np.stack(boards)

        values = np.array([1.0])

        policies = np.zeros(4096, np.float32)
        policies[0] = 1.0
        policies = [policies]
        policies = np.stack(policies)

        colosus.train(boards, policies, values)

        policy, value = colosus.predict(pos.board)
        print(value)
        print(policy)

    def test_legal_policy(self):
        policy = np.array([0.25, 0.25, 0.25, 0.25])
        moves = [1,3]
        legal_policy = ColosusModel.legal_policy(policy, moves)
        self.assertEqual(0.0, legal_policy[0])
        self.assertEqual(0.5, legal_policy[1])


if __name__ == '__main__':
    unittest.main()