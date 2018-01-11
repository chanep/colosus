import unittest

from colosus.trainer import Trainer
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.game.piece import Piece


class SelfPlayTestCase(unittest.TestCase):
    def test_train(self):
        train_filename = "train1.dat"
        weights_filename = "weights1.h5"

        trainer = Trainer()
        trainer.train(train_filename, weights_filename, 100)


if __name__ == '__main__':
    unittest.main()