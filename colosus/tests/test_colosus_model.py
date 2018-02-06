import unittest

import datetime
import time
import tensorflow as tf
import numpy as np
from colosus.colosus_model import ColosusModel
from colosus.config import SearchConfig, StateConfig, ColosusConfig
from colosus.game.position import Position
from colosus.game.side import Side
from tensorflow.python.keras import backend as K

from colosus.game.square import Square
from colosus.searcher import Searcher
from colosus.state import State



class ColosusModelTestCase(unittest.TestCase):
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

        # tapar 4
        # pos.put_piece(Side.BLACK, 1, 1)
        # pos.put_piece(Side.BLACK, 4, 2)
        # pos.put_piece(Side.BLACK, 13, 0)
        # pos.put_piece(Side.BLACK, 2, 9)
        #
        # pos.put_piece(Side.WHITE, 11, 6)
        # pos.put_piece(Side.WHITE, 12, 7)
        # pos.put_piece(Side.WHITE, 14, 9)
        # pos.put_piece(Side.WHITE, 15, 10)

        # poner la 4ta en 3
        pos.put_piece(Side.BLACK, 7, 7)
        pos.put_piece(Side.WHITE, 7, 8)
        pos.put_piece(Side.BLACK, 11, 7)
        pos.put_piece(Side.WHITE, 8, 8)
        pos.put_piece(Side.BLACK, 6, 7)
        pos.put_piece(Side.WHITE, 9, 8)

        colosus = ColosusModel(ColosusConfig())
        colosus.build()
        # colosus.load_weights("c_1_500_1600.h5")
        # colosus.load_weights("c_4_600_1600.h5")
        colosus.load_weights("c_9_800_1600.h5")

        pos.print()

        policy, value = colosus.predict(pos.to_model_position())
        print("value: " + str(value))
        print("moves prob")
        sorted_policy = self.sort_policy(policy)
        for i in range(20):
            m = sorted_policy[i]
            print("{} - {}".format(m[0], m[1]))

        print('')

        searcher = Searcher(SearchConfig());
        state = State(pos, None, None, colosus, StateConfig())
        policy, value, move, new_state = searcher.search(state, 2048)
        print("value: " + str(value))
        print("moves prob")
        sorted_policy = self.sort_policy(policy)
        for i in range(10):
            m = sorted_policy[i]
            print("{} - {}".format(m[0], m[1]))

        for m in range(len(state.children())):
            c = state.children()[m]
            if c is not None:
                print("{} N: {}, W: {:.3g}, Q: {:.3g}, p:{:.3g}".format(Square.to_string(m), c.N, c.W, c.Q, c.P))

    def test_predict_on_batch(self):
        pos = Position()

        # poner la 4ta en 3
        pos.put_piece(Side.BLACK, 1, 1)
        pos.put_piece(Side.BLACK, 4, 2)
        pos.put_piece(Side.BLACK, 13, 0)

        pos.put_piece(Side.WHITE, 11, 6)
        pos.put_piece(Side.WHITE, 12, 7)
        pos.put_piece(Side.WHITE, 13, 8)

        colosus = ColosusModel(ColosusConfig())
        colosus.build()

        positions = []
        position = pos
        for m in range(8):
            positions.append(position.to_model_position())
            position = position.move(m)

        policies, values = colosus.predict_on_batch(positions)

        cant = 1000
        start = time.time()
        for i in range(cant):
            policies, values = colosus.predict_on_batch(positions)
        print("predict_on_batch time: " + str(time.time() - start))

        start = time.time()
        for i in range(cant):
            for p in positions:
                policy, value = colosus.predict(p)
        print("predict time: " + str(time.time() - start))

    def sort_policy(self, policy):
        move_policy = []
        for m in range(len(policy)):
            m_str = Square.to_string(m)
            move_policy.append((m_str, policy[m]))
        return sorted(move_policy, key=lambda t: t[1], reverse=True)

    def test_train(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 0, 5)
        pos.put_piece(Side.WHITE, Piece.ROOK, 0, 0)
        pos.put_piece(Side.BLACK, Piece.KING, 7, 5)
        pos.move_count = 1

        pos2 = pos.clone()
        pos2.move_count = 100

        colosus = ColosusModel()
        colosus.build()

        policy, value = colosus.predict(pos.to_model_position())
        print(value)
        print(policy)

        positions = [pos, pos2]

        values = np.array([-1.0, 0.0])

        policy = np.zeros(4096, np.float32)
        policy[0] = 1.0
        policies = [policy, policy]

        colosus.train(positions, policies, values)

        policy, value = colosus.predict(pos.to_model_position())
        print(value)
        print(policy)

        policy, value = colosus.predict(pos2.to_model_position())
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