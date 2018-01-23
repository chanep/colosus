import numpy as np

files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
SIZE = 16


class Square:

    @staticmethod
    def square(rank, file):
        return rank * SIZE + file

    @staticmethod
    def to_rank_file(square):
        return int(square // 16), square % 16

    @staticmethod
    def board_to_rank_file(board):
        square = np.math.log(board, 2)
        return Square.to_rank_file(square)

    @staticmethod
    def piece_rank_file(board):
        if np.sum(board) == 0:
            return None, None
        rank = np.sum(np.sum(board, axis=1) * np.arange(SIZE)).astype(np.uint8)
        file = np.sum(np.sum(board, axis=0) * np.arange(SIZE)).astype(np.uint8)
        return rank, file

    @staticmethod
    def to_string(square):
        rank, file = Square.to_rank_file(square)
        r = rank + 1
        f = files[file]
        return f + str(r)
