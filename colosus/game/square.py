import numpy as np


class Square:
    @staticmethod
    def square(rank, file):
        return rank * 8 + file

    @staticmethod
    def board(rank, file):
        return np.uint64(1) << Square.index(rank, file)

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
