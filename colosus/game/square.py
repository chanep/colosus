import numpy as np

SIZE = 15


class Square:

    @staticmethod
    def square(rank, file):
        return rank * SIZE + file

    @staticmethod
    def to_rank_file(square):
        return int(square // SIZE), square % SIZE

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
        return "({},{})".format(rank, file)
