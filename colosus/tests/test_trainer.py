import unittest
import numpy as np

from colosus.trainer import Trainer
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.game.piece import Piece


class SelfPlayTestCase(unittest.TestCase):
    def test_train(self):
        train_filename = "t2_1_1000_200.dat"
        # weights_filename = "w2_1_1000_200_3000.h5"
        weights_filename = "x.h5"

        trainer = Trainer()
        trainer.train(train_filename, weights_filename, 30)

    def test_generator(self):

        def generator():
            yield 3
            yield 4
            yield 5

        for x in generator():
            print(x)

        def p(x, y):
            print("x: {}, y: {}".format(x, y))

        def get_input():
            return 3, 4

        p(*get_input())



if __name__ == '__main__':
    unittest.main()