import unittest
import numpy as np

from colosus.game.model_position import ModelPosition
from colosus.game.move import Move
from colosus.game.piece import Piece
from colosus.game.rotator import Rotator
from colosus.game.side import Side
from colosus.game.square import Square
from colosus.train_record import TrainRecord


class PositionTestCase(unittest.TestCase):
    def test_rotations(self):
        assertEqual = self.assertEqual
        assertTrue = self.assertTrue

        board = np.zeros((Side.COUNT * Piece.COUNT, 8, 8), np.uint8)
        board[0, 1, 2] = 1
        board[1, 7, 5] = 1
        board[2, 1, 3] = 1

        orig = Square.square(1, 2)
        dest = Square.square(1, 3)
        move = Move.from_squares(orig, dest)
        policy = np.array([0.0] * (64 * 64))
        policy[move] = 1.0
        move_count = 10
        value = 0.8

        position = ModelPosition(board, move_count)

        record = TrainRecord(position, policy, value)

        rotator = Rotator()
        rotations = rotator.rotations(record)

        # print("original board")
        # print(board)

        # original
        rec = rotations[0]
        b = rec.position.board
        p = rec.policy
        assertEqual(1, b[0, 1, 2])
        assertEqual(1, b[2, 1, 3])
        assertEqual(1.0, p[move])
        assertEqual(value, rec.value)

        # flip original
        rec = rotations[1]
        b = rec.position.board
        p = rec.policy

        m = Move.from_rank_files(1, 5, 1, 4)

        assertEqual(0, b[0, 1, 2])
        assertEqual(1, b[0, 1, 5])

        assertEqual(0, b[1, 7, 5])
        assertEqual(1, b[1, 7, 2])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # rot90
        rec = rotations[2]
        b = rec.position.board
        p = rec.policy

        m = Move.from_rank_files(5, 1, 4, 1)

        # print("rot90 board")
        # print(b)

        assertEqual(0, b[0, 1, 2])
        assertEqual(1, b[0, 5, 1])

        assertEqual(0, b[1, 7, 5])
        assertEqual(1, b[1, 2, 7])

        assertEqual(0, b[2, 1, 3])
        assertEqual(1, b[2, 4, 1])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # flip rot90
        rec = rotations[3]
        b = rec.position.board
        p = rec.policy

        m = Move.from_rank_files(5, 6, 4, 6)

        # print("flip rot90 board")
        # print(b)

        assertEqual(0, b[0, 1, 2])
        assertEqual(1, b[0, 5, 6])

        assertEqual(0, b[1, 7, 5])
        assertEqual(1, b[1, 2, 0])

        assertEqual(0, b[2, 1, 3])
        assertEqual(1, b[2, 4, 6])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # rot180
        rec = rotations[4]
        b = rec.position.board
        p = rec.policy

        m = Move.from_rank_files(6, 5, 6, 4)

        # print("rot180 board")
        # print(b)

        assertEqual(0, b[0, 1, 2])
        assertEqual(1, b[0, 6, 5])

        assertEqual(0, b[1, 7, 5])
        assertEqual(1, b[1, 0, 2])

        assertEqual(0, b[2, 1, 3])
        assertEqual(1, b[2, 6, 4])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # flip rot180
        rec = rotations[5]
        b = rec.position.board
        p = rec.policy

        m = Move.from_rank_files(6, 2, 6, 3)

        # print("flip rot180 board")
        # print(b)

        assertEqual(0, b[0, 1, 2])
        assertEqual(1, b[0, 6, 2])

        assertEqual(0, b[1, 7, 5])
        assertEqual(1, b[1, 0, 5])

        assertEqual(0, b[2, 1, 3])
        assertEqual(1, b[2, 6, 3])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # rot270
        rec = rotations[6]
        b = rec.position.board
        p = rec.policy

        m = Move.from_rank_files(2, 6, 3, 6)

        # print("rot270 board")
        # print(b)

        assertEqual(0, b[0, 1, 2])
        assertEqual(1, b[0, 2, 6])

        assertEqual(0, b[1, 7, 5])
        assertEqual(1, b[1, 5, 0])

        assertEqual(0, b[2, 1, 3])
        assertEqual(1, b[2, 3, 6])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # flip rot270
        rec = rotations[7]
        b = rec.position.board
        p = rec.policy

        m = Move.from_rank_files(2, 1, 3, 1)

        # print("flip rot270 board")
        # print(b)

        assertEqual(0, b[0, 1, 2])
        assertEqual(1, b[0, 2, 1])

        assertEqual(0, b[1, 7, 5])
        assertEqual(1, b[1, 5, 7])

        assertEqual(0, b[2, 1, 3])
        assertEqual(1, b[2, 3, 1])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])







if __name__ == '__main__':
    unittest.main()