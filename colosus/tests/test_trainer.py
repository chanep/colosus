import unittest
import numpy as np

from colosus.trainer import Trainer
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.game.piece import Piece


class SelfPlayTestCase(unittest.TestCase):
    def test_train(self):
        train_filename = "t2_1_1000_200.dat"
        weights_filename = "w2_1_1000_200_3000.h5"
        # weights_filename = "x.h5"

        trainer = Trainer()
        trainer.train(train_filename, weights_filename, 3000)

    def test_rotate(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        todos = []
        aux = x
        for i in range(4):
            # a = np.flipud(aux)
            b = np.fliplr(aux)
            todos.append(aux)
            # todos.append(a)
            todos.append(b)
            aux = np.rot90(aux, 1)

        unicos = []
        len_todos = len(todos)
        print("todos: " + str(len_todos))
        for i in range(len_todos):
            e = todos[i]
            ya_esta = False
            for j in range(len(unicos)):
                if np.array_equal(e, unicos[j]):
                    ya_esta = True
                    break
            if not ya_esta:
                unicos.append(e)

        print("unicos: " + str(len(unicos)))

    def test_rotate2(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        i = np.identity(3)
        rx = np.rot90(x)
        ri = np.rot90(i)
        xi = np.dot(x, ri)
        print(rx)
        print(ri)
        print(xi)

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