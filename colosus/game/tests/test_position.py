import unittest
from ..position import Position
from ..move import Move
from ..square import Square
from ..side import Side
from ..piece import Piece


class PositionTestCase(unittest.TestCase):
    def test_print(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 0, 5)
        pos.put_piece(Side.WHITE, Piece.ROOK, 0, 0)
        pos.put_piece(Side.BLACK, Piece.KING, 7, 5)
        pos.print()

    def test_move(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 4)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 5)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 6)

        pos.print()

        move = Move.from_squares(Square.square(4,5), Square.square(4,6))
        new_pos = pos.move(move)
        new_pos.print()
        self.assertEqual(Piece.ROOK, new_pos.piece_at(Side.WHITE, 4, 6))
        self.assertIsNone(new_pos.piece_at(Side.WHITE, 4, 5))

    def test_checkmate(self):
        pos = Position()
        pos.side_to_move = Side.BLACK
        pos.put_piece(Side.WHITE, Piece.KING, 5, 4)
        pos.put_piece(Side.WHITE, Piece.ROOK, 7, 0)
        pos.put_piece(Side.BLACK, Piece.KING, 7, 4)
        pos._chech_end()

        pos.print()

        self.assertTrue(pos.is_end)
        self.assertEqual(1, pos.score)

        pos = Position()
        pos.side_to_move = Side.BLACK
        pos.put_piece(Side.WHITE, Piece.KING, 5, 4)
        pos.put_piece(Side.WHITE, Piece.ROOK, 7, 0)
        pos.put_piece(Side.BLACK, Piece.KING, 7, 5)

        self.assertFalse(pos.is_end)
        self.assertIsNone(pos.score)


if __name__ == '__main__':
    unittest.main()
