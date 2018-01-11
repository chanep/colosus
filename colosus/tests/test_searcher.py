import unittest

from colosus.state import State
from colosus.searcher import Searcher
from colosus.colosus_model import ColosusModel
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.game.piece import Piece


class SearcherTestCase(unittest.TestCase):
    def test_search(self):
        colosus = ColosusModel()
        colosus.build()

        pos = Position()
        pos.put_piece(Side.WHITE, Piece.KING, 4, 3)
        pos.put_piece(Side.WHITE, Piece.ROOK, 4, 4)
        pos.put_piece(Side.BLACK, Piece.KING, 5, 5)

        state = State(pos, None, None, colosus)

        searcher = Searcher()
        policy, value, move, new_state = searcher.search(state, 1000)

        state.print()
        new_state.print()

        print("value: " + str(value))


if __name__ == '__main__':
    unittest.main()
