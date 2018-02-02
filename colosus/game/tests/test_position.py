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
        move = Square.square(8, 8)
        self.assertFalse(pos.is_legal_colosus(move))
        move = Square.square(7, 7)
        self.assertTrue(pos.is_legal_colosus(move))
        pos = pos.move(move)

        move = Square.square(10, 10)
        self.assertFalse(pos.is_legal_colosus(move))
        move = Square.square(5, 9)
        self.assertTrue(pos.is_legal_colosus(move))
        pos = pos.move(move)

        move = Square.square(10, 10)
        self.assertFalse(pos.is_legal_colosus(move))
        move = Square.square(6, 6)
        self.assertFalse(pos.is_legal_colosus(move))
        move = Square.square(3, 11)
        self.assertTrue(pos.is_legal_colosus(move))


    def test_is_legal_colosus(self):
        pos = Position()
        pos.put_piece(Side.BLACK, 0, 0)
        pos.put_piece(Side.WHITE, 12, 7)
        pos.put_piece(Side.BLACK, 13, 8)

        move = Square.square(14, 9)
        move2 = Square.square(12, 7)
        m3 = Square.square(11, 7)
        m4 = Square.square(13, 7)
        m5 = Square.square(13, 6)
        m6 = Square.square(13, 9)
        m7 = Square.square(12, 8)
        m8 = Square.square(13, 7)
        m9 = Square.square(14, 9)
        m10 = Square.square(14, 7)
        m11 = Square.square(10, 7)
        m12 = Square.square(10, 8)
        m14 = Square.square(10, 5)
        m15 = Square.square(11, 4)

        self.assertTrue(pos.is_legal_colosus(move))
        self.assertFalse(pos.is_legal_colosus(move2))
        pos.switch_side()
        self.assertFalse(pos.is_legal_colosus(move2))
        self.assertTrue(pos.is_legal_colosus(m3))
        self.assertTrue(pos.is_legal_colosus(m4))
        self.assertTrue(pos.is_legal_colosus(m5))
        self.assertTrue(pos.is_legal_colosus(m6))
        self.assertTrue(pos.is_legal_colosus(m7))
        self.assertTrue(pos.is_legal_colosus(m8))
        self.assertTrue(pos.is_legal_colosus(m9))
        self.assertTrue(pos.is_legal_colosus(m10))
        self.assertTrue(pos.is_legal_colosus(m11))
        self.assertTrue(pos.is_legal_colosus(m12))
        self.assertTrue(pos.is_legal_colosus(m14))
        self.assertFalse(pos.is_legal_colosus(m15))

    def test_is_legal_colosus2(self):
        pos = Position()
        pos.put_piece(Side.BLACK, 7, 7)
        pos.put_piece(Side.WHITE, 7, 8)

        m1 = Square.square(11, 7)
        m2 = Square.square(7, 11)
        m3 = Square.square(8, 3)
        m4 = Square.square(11, 11)
        m5 = Square.square(3, 3)
        m6 = Square.square(3, 7)
        m7 = Square.square(3, 6)
        m8 = Square.square(11, 5)
        m9 = Square.square(11, 10)
        m10 = Square.square(4, 3)
        m11 = Square.square(10, 3)
        m12 = Square.square(6, 3)

        m14 = Square.square(10, 10)
        m15 = Square.square(14, 14)
        m16 = Square.square(10, 7)
        m17 = Square.square(9, 4)

        self.assertTrue(pos.is_legal_colosus(m1))
        self.assertTrue(pos.is_legal_colosus(m2))
        self.assertTrue(pos.is_legal_colosus(m3))
        self.assertTrue(pos.is_legal_colosus(m4))
        self.assertTrue(pos.is_legal_colosus(m5))
        self.assertTrue(pos.is_legal_colosus(m6))
        self.assertTrue(pos.is_legal_colosus(m7))
        self.assertTrue(pos.is_legal_colosus(m8))
        self.assertTrue(pos.is_legal_colosus(m9))
        self.assertTrue(pos.is_legal_colosus(m10))
        self.assertTrue(pos.is_legal_colosus(m11))
        self.assertTrue(pos.is_legal_colosus(m12))

        self.assertFalse(pos.is_legal_colosus(m14))
        self.assertFalse(pos.is_legal_colosus(m15))
        self.assertFalse(pos.is_legal_colosus(m16))
        self.assertFalse(pos.is_legal_colosus(m17))

    def test_is_end(self):
        pos = Position()
        pos.switch_side()
        pos.put_piece(Side.WHITE, 12, 7)
        pos.put_piece(Side.WHITE, 13, 8)
        pos.put_piece(Side.WHITE, 14, 9)
        pos.put_piece(Side.WHITE, 10, 5)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.WHITE, 11, 6)
        self.assertTrue(pos.is_end)
        self.assertEqual(1, pos.score)

    def test_is_end2(self):
        pos = Position()
        pos.put_piece(Side.BLACK, 10, 7)
        pos.put_piece(Side.BLACK, 14, 3)
        pos.put_piece(Side.BLACK, 13, 4)
        pos.put_piece(Side.BLACK, 12, 5)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.BLACK, 11, 6)
        self.assertTrue(pos.is_end)

    def test_is_end3(self):
        pos = Position()
        pos.put_piece(Side.BLACK, 5, 0)
        pos.put_piece(Side.BLACK, 4, 1)
        pos.put_piece(Side.BLACK, 3, 2)
        pos.put_piece(Side.BLACK, 2, 3)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.BLACK, 1, 4)
        self.assertTrue(pos.is_end)

    def test_is_end4(self):
        pos = Position()
        pos.switch_side()
        pos.put_piece(Side.WHITE, 0, 4)
        pos.put_piece(Side.WHITE, 1, 5)
        pos.put_piece(Side.WHITE, 2, 6)
        pos.put_piece(Side.WHITE, 3, 7)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.WHITE, 4, 8)
        self.assertTrue(pos.is_end)

    def test_is_end5(self):
        pos = Position()
        pos.switch_side()
        pos.put_piece(Side.WHITE, 5, 4)
        pos.put_piece(Side.WHITE, 5, 5)
        pos.put_piece(Side.WHITE, 5, 6)
        pos.put_piece(Side.WHITE, 5, 7)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.WHITE, 5, 8)
        self.assertTrue(pos.is_end)

    def test_is_end6(self):
        pos = Position()
        pos.switch_side()
        pos.put_piece(Side.WHITE, 4, 4)
        pos.put_piece(Side.WHITE, 5, 4)
        pos.put_piece(Side.WHITE, 6, 4)
        pos.put_piece(Side.WHITE, 7, 4)

        self.assertFalse(pos.is_end)
        pos.put_piece(Side.WHITE, 8, 4)
        self.assertTrue(pos.is_end)

    def test_is_end7(self):
        pos = Position()
        pos.switch_side()
        pos.put_piece(Side.WHITE, 12, 7)
        pos.put_piece(Side.WHITE, 13, 8)
        pos.put_piece(Side.WHITE, 14, 9)
        pos.put_piece(Side.WHITE, 10, 5)

        self.assertFalse(pos.is_end)
        move = Square.square(11, 6)
        new_pos = pos.move(move)
        self.assertTrue(new_pos.is_end)
        self.assertEqual(-1, new_pos.score)

    def test_is_end8(self):
        pos = Position()
        pos.switch_side()
        pos.put_piece(Side.WHITE, 4, 4)
        pos.put_piece(Side.WHITE, 5, 5)
        pos.put_piece(Side.WHITE, 6, 6)
        pos.put_piece(Side.WHITE, 8, 8)
        pos.put_piece(Side.WHITE, 9, 9)

        self.assertFalse(pos.is_end)
        move = Square.square(7, 7)
        new_pos = pos.move(move)
        self.assertFalse(new_pos.is_end)

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
