import unittest

import datetime
import time
import tensorflow as tf
import numpy as np
from colosus.colosus_model import ColosusModel
from colosus.config import SearchConfig, StateConfig, ColosusConfig, PlayerConfig
from colosus.game.position import Position
from colosus.game.side import Side
from tensorflow.python.keras import backend as K

from colosus.game.square import Square
from colosus.player import Player
from colosus.searcher import Searcher
from colosus.state import State



class ColosusModelTestCase(unittest.TestCase):
    def test_player(self):
        pos = Position()

        pos.put_piece(Side.BLACK, 7, 7)
        pos.put_piece(Side.WHITE, 8, 6)

        pos.put_piece(Side.BLACK, 7, 2)
        pos.put_piece(Side.WHITE, 8, 8)

        # pos.put_piece(Side.BLACK, 9, 7)
        # pos.put_piece(Side.WHITE, 8, 7)

        # pos.put_piece(Side.BLACK, 7, 7)
        # pos.put_piece(Side.WHITE, 7, 8)

        pos.print()

        colosus = ColosusModel(ColosusConfig())
        colosus.build()
        # colosus.load_weights("c_1_500_1600.h5")
        # colosus.load_weights("c_4_600_1600.h5")
        colosus.load_weights("c_22_1200_1600.h5")

        player_config = PlayerConfig()
        player = Player(player_config, colosus)

        player.new_game(pos, 256)

        policy, value, move, old_state, new_state = player.move()

        self.print_children(old_state)

        print('\n\n')

        self.print_children(new_state)

        child = new_state.children()[Square.square(8, 7)]

        print('\n\n')

        self.print_children(child)

        print('\n\n')

        player.opponent_move(Square.square(8, 7))

        policy, value, move, old_state, new_state = player.move()

        self.print_children(old_state)


    def test_player(self):
        pos = Position()

        # pos.put_piece(Side.BLACK, 7, 7)
        # pos.put_piece(Side.WHITE, 8, 6)
        #
        # pos.put_piece(Side.BLACK, 7, 2)
        # pos.put_piece(Side.WHITE, 8, 8)

        # pos.put_piece(Side.BLACK, 9, 7)
        # pos.put_piece(Side.WHITE, 8, 7)

        # pos.put_piece(Side.BLACK, 7, 7)
        # pos.put_piece(Side.WHITE, 7, 8)

        pos.print()

        colosus = ColosusModel(ColosusConfig())
        colosus.build()
        # colosus.load_weights("c_1_500_1600.h5")
        # colosus.load_weights("c_4_600_1600.h5")
        colosus.load_weights("c_22_1200_1600.h5")

        player_config = PlayerConfig()
        player = Player(player_config, colosus)

        player.new_game(pos, 256)

        policy, value, move, old_state, new_state = player.move()

        player.opponent_move(Square.square(8, 6))

        policy, value, move, old_state, new_state = player.move()

        child = old_state.children()[Square.square(7, 2)]

        player.state = child

        player.state.position().print()




        player.opponent_move(Square.square(8, 8))

        policy, value, move, old_state, new_state = player.move()

        self.print_children(old_state)

        print(new_state.position().print())



        player.opponent_move(Square.square(8, 7))

        policy, value, move, old_state, new_state = player.move()

        self.print_children(old_state)

        print(new_state.position().print())



    def print_children(self, state: State):
        for m in range(len(state.children())):
            c = state.children()[m]
            if c is not None:
                print("{} N: {}, W: {:.3g}, Q: {:.3g}, p:{:.3g}".format(Square.to_string(m), c.N, c.W, c.Q, c.P))

if __name__ == '__main__':
    unittest.main()