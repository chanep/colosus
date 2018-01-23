import unittest
import cProfile, pstats, io

import numpy as np
from ..position import Position
from ..square import Square
from ..side import Side


class PositionTestCase(unittest.TestCase):
    def test_put_piece_profile(self):
        pr = cProfile.Profile()
        pos = Position()

        pr.enable()

        for i in range(100000):
            pos.put_piece(Side.WHITE, 10, 4)

        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def test_print(self):
        pos = Position()
        pos.put_piece(Side.WHITE, 3, 7)
        pos.put_piece(Side.BLACK, 4, 4)
        pos.print()

    def test_move(self):
        pos = Position()
        pos.put_piece(Side.WHITE, 12, 7)
        pos.put_piece(Side.BLACK, 13, 8)

        move = Square.square(14, 9)
        new_pos = pos.move(move)

        self.assertTrue(new_pos.piece_at(Side.WHITE, 14, 9))
        self.assertEqual(3, new_pos.move_count)

    def test_is_legal_move(self):
        pos = Position()
        pos.put_piece(Side.WHITE, 12, 7)
        pos.put_piece(Side.BLACK, 13, 8)

        move = Square.square(14, 9)
        move2 = Square.square(12, 7)

        self.assertTrue(pos.is_legal(move))
        self.assertFalse(pos.is_legal(move2))
        pos.switch_side()
        self.assertFalse(pos.is_legal(move2))

    def test_is_end(self):
        pos = Position()
        pos.put_piece(Side.WHITE, 12, 7)
        pos.put_piece(Side.WHITE, 13, 8)
        pos.put_piece(Side.WHITE, 14, 9)
        pos.put_piece(Side.WHITE, 15, 10)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.WHITE, 11, 6)
        self.assertTrue(pos.is_end)
        self.assertEqual(1, pos.score)

    def test_is_end2(self):
        pos = Position()
        pos.switch_side()
        pos.put_piece(Side.BLACK, 15, 2)
        pos.put_piece(Side.BLACK, 14, 3)
        pos.put_piece(Side.BLACK, 13, 4)
        pos.put_piece(Side.BLACK, 12, 5)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.BLACK, 11, 6)
        self.assertTrue(pos.is_end)

    def test_is_end3(self):
        pos = Position()
        pos.switch_side()
        pos.put_piece(Side.BLACK, 5, 0)
        pos.put_piece(Side.BLACK, 4, 1)
        pos.put_piece(Side.BLACK, 3, 2)
        pos.put_piece(Side.BLACK, 2, 3)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.BLACK, 1, 4)
        self.assertTrue(pos.is_end)

    def test_is_end4(self):
        pos = Position()
        pos.put_piece(Side.WHITE, 0, 4)
        pos.put_piece(Side.WHITE, 1, 5)
        pos.put_piece(Side.WHITE, 2, 6)
        pos.put_piece(Side.WHITE, 3, 7)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.WHITE, 4, 8)
        self.assertTrue(pos.is_end)

    def test_is_end5(self):
        pos = Position()
        pos.put_piece(Side.WHITE, 5, 4)
        pos.put_piece(Side.WHITE, 5, 5)
        pos.put_piece(Side.WHITE, 5, 6)
        pos.put_piece(Side.WHITE, 5, 7)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.WHITE, 5, 8)
        self.assertTrue(pos.is_end)

    def test_is_end6(self):
        pos = Position()
        pos.put_piece(Side.WHITE, 4, 4)
        pos.put_piece(Side.WHITE, 5, 4)
        pos.put_piece(Side.WHITE, 6, 4)
        pos.put_piece(Side.WHITE, 7, 4)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.WHITE, 8, 4)
        self.assertTrue(pos.is_end)

    def test_is_end7(self):
        pos = Position()
        pos.put_piece(Side.WHITE, 12, 7)
        pos.put_piece(Side.WHITE, 13, 8)
        pos.put_piece(Side.WHITE, 14, 9)
        pos.put_piece(Side.WHITE, 15, 10)

        self.assertFalse(pos.is_end)
        move = Square.square(11, 6)
        new_pos = pos.move(move)
        self.assertTrue(new_pos.is_end)
        self.assertEqual(-1, new_pos.score)

    def test_to_model_position(self):
        pos = Position()
        pos.put_piece(Side.WHITE, 4, 4)
        pos.put_piece(Side.WHITE, 5, 4)
        pos.put_piece(Side.WHITE, 6, 4)
        pos.put_piece(Side.WHITE, 7, 4)
        pos.put_piece(Side.WHITE, 8, 4)

        model_position = pos.to_model_position()

        self.assertTrue(np.array_equal(model_position.board[0, 4:9, 4], np.array([1, 1, 1, 1, 1])))

        pos.switch_side()
        model_position = pos.to_model_position()
        self.assertFalse(np.array_equal(model_position.board[0, 4:9, 4], np.array([1, 1, 1, 1, 1])))
        self.assertTrue(np.array_equal(model_position.board[1, 4:9, 4], np.array([1, 1, 1, 1, 1])))


if __name__ == '__main__':
    unittest.main()
