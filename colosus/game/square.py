import numpy as np

files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


class Square:
    @staticmethod
    def square(rank, file):
        return rank * 8 + file

    @staticmethod
    def board(rank, file):
        return np.uint64(1) << Square.square(rank, file)

    @staticmethod
    def board(square):
        return np.uint64(1) << square

    @staticmethod
    def to_rank_file(square):
        return square / 8, square % 8

    @staticmethod
    def board_to_rank_file(board):
        square = np.math.log(board, 2)
        return Square.to_rank_file(square)

    @staticmethod
    def to_string(square):
        rank, file = Square.to_rank_file(square)
        r = rank + 1
        f = files[file]
        return f + str(r)
