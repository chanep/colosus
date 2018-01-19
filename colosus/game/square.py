import numpy as np

files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


class Square:
    @staticmethod
    def square(rank, file):
        return rank * 8 + file

    @staticmethod
    def to_rank_file(square):
        return int(square // 8), square % 8

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

    @staticmethod
    def piece_rank_file(board):
        if np.sum(board) == 0:
            return None, None
        rank = np.sum(np.sum(board, axis=1) * np.arange(8)).astype(np.uint8)
        file = np.sum(np.sum(board, axis=0) * np.arange(8)).astype(np.uint8)
        return rank, file
