import unittest
import numpy as np

from colosus.game.model_position import ModelPosition

from colosus.game.position import Position
from colosus.game.rotator import Rotator
from colosus.game.side import Side
from colosus.game.square import Square
from colosus.train_record import TrainRecord


class PositionTestCase(unittest.TestCase):
    def test_rotations(self):
        size = Position.B_SIZE
        assertEqual = self.assertEqual

        board = np.zeros((Side.COUNT, size, size), np.uint8)
        board[0, 13, 4] = 1
        board[1, 11, 5] = 1

        move = Square.square(9, 14)
        policy = np.array([0.0] * (size * size))
        policy[move] = 1.0
        value = 0.8

        position = ModelPosition(board)

        record = TrainRecord(position, policy, value)

        rotator = Rotator()
        rotations = rotator.rotations(record)

        # print("original board")
        # print(board)

        # original
        rec = rotations[0]
        b = rec.position.board
        p = rec.policy
        assertEqual(1, b[0, 13, 4])
        assertEqual(1, b[1, 11, 5])
        assertEqual(1.0, p[move])
        assertEqual(value, rec.value)

        # flip original
        rec = rotations[1]
        b = rec.position.board
        p = rec.policy

        m = Square.square(9, 0)

        assertEqual(0, b[0, 13, 4])
        assertEqual(1, b[0, 13, 10])

        assertEqual(0, b[1, 11, 5])
        assertEqual(1, b[1, 11, 9])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # rot90
        rec = rotations[2]
        b = rec.position.board
        p = rec.policy

        m = Square.square(0, 9)

        assertEqual(0, b[0, 13, 4])
        assertEqual(1, b[0, 10, 13])

        assertEqual(0, b[1, 11, 5])
        assertEqual(1, b[1, 9, 11])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # flip rot90
        rec = rotations[3]
        b = rec.position.board
        p = rec.policy

        m = Square.square(0, 5)

        assertEqual(0, b[0, 13, 4])
        assertEqual(1, b[0, 10, 1])

        assertEqual(0, b[1, 11, 5])
        assertEqual(1, b[1, 9, 3])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # rot180
        rec = rotations[4]
        b = rec.position.board
        p = rec.policy

        m = Square.square(5, 0)

        assertEqual(0, b[0, 13, 4])
        assertEqual(1, b[0, 1, 10])

        assertEqual(0, b[1, 11, 5])
        assertEqual(1, b[1, 3, 9])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # flip rot180
        rec = rotations[5]
        b = rec.position.board
        p = rec.policy

        m = Square.square(5, 14)

        assertEqual(0, b[0, 13, 4])
        assertEqual(1, b[0, 1, 4])

        assertEqual(0, b[1, 11, 5])
        assertEqual(1, b[1, 3, 5])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # rot270
        rec = rotations[6]
        b = rec.position.board
        p = rec.policy

        m = Square.square(14, 5)

        assertEqual(0, b[0, 13, 4])
        assertEqual(1, b[0, 4, 1])

        assertEqual(0, b[1, 11, 5])
        assertEqual(1, b[1, 5, 3])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])

        # flip rot270
        rec = rotations[7]
        b = rec.position.board
        p = rec.policy

        m = Square.square(14, 9)

        assertEqual(0, b[0, 13, 4])
        assertEqual(1, b[0, 4, 13])

        assertEqual(0, b[1, 11, 5])
        assertEqual(1, b[1, 5, 11])

        assertEqual(0, p[move])
        assertEqual(1.0, p[m])


if __name__ == '__main__':
    unittest.main()