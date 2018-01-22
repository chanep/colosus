import unittest
import pickle
from ..position import Position
from ..square import Square
from ..side import Side


class PositionTestCase(unittest.TestCase):
    def test_print(self):
        pos = Position()
        pos.put_piece(Side.WHITE, 3, 7)
        pos.put_piece(Side.BLACK, 4, 4)
        pos.print()

    def test_move(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 4)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 5)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 6)

        pos.print()

        move = Move.from_squares(Square.square(4, 5), Square.square(4, 6))
        new_pos = pos.move(move)
        new_pos.print()
        self.assertEqual(Piece.ROOK, new_pos.piece_at(Side.WHITE, 4, 6))
        self.assertIsNone(new_pos.piece_at(Side.WHITE, 4, 5))

    def test_to_model_position(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 4)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 5)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 6)

        pos.print()

        model_position = pos.to_model_position()
        print(model_position.board)

    def test_is_legal_move(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 4)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 5)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 6)

        pos.print()

        move = Move.from_squares(Square.square(4, 4), Square.square(5, 5))
        self.assertFalse(pos.is_legal(move))

        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 2, 4)
        pos.put_piece(Side.WHITE, Piece.ROOK, 2, 1)
        pos.put_piece(Side.BLACK, Piece.KING, 3, 6)
        pos.switch_side()

        pos.print()

        move = Move.from_squares(Square.square(3, 6), Square.square(2, 6))
        self.assertTrue(pos.is_legal(move))

        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 2, 6)
        pos.put_piece(Side.WHITE, Piece.ROOK, 2, 1)
        pos.put_piece(Side.BLACK, Piece.KING, 3, 4)
        pos.switch_side()

        pos.print()

        move = Move.from_squares(Square.square(3, 4), Square.square(2, 4))
        self.assertFalse(pos.is_legal(move))

    def test_checkmate(self):
        pos = Position()
        pos.side_to_move = Side.BLACK
        pos.put_piece(Side.WHITE, Piece.KING, 5, 4)
        pos.put_piece(Side.WHITE, Piece.ROOK, 7, 0)
        pos.put_piece(Side.BLACK, Piece.KING, 7, 4)
        pos._check_end()

        pos.print()

        self.assertTrue(pos.is_end)
        self.assertEqual(-1, pos.score)

        pos = Position()
        pos.side_to_move = Side.BLACK
        pos.put_piece(Side.WHITE, Piece.KING, 5, 4)
        pos.put_piece(Side.WHITE, Piece.ROOK, 7, 0)
        pos.put_piece(Side.BLACK, Piece.KING, 7, 5)

        self.assertFalse(pos.is_end)
        self.assertIsNone(pos.score)

    def test_in_check(self):
        pos = Position()
        pos.side_to_move = Side.BLACK
        pos.put_piece(Side.WHITE, Piece.KING, 6, 3)
        pos.put_piece(Side.WHITE, Piece.ROOK, 5, 3)
        pos.put_piece(Side.BLACK, Piece.KING, 3, 3)
        self.assertTrue(pos.in_check())


    def test_save(self):
        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 4)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 5)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 6)
        records = [(pos, 18)]
        with open("bin.dat", "wb") as f:
            pickle.dump(records, f)

        with open("bin.dat", "rb") as f:
            records = pickle.load(f)
            print(records)


if __name__ == '__main__':
    unittest.main()
