import numpy as np

files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']


class Square:
    @staticmethod
    def square(rank, file):
        return rank * 16 + file

    @staticmethod
    def to_rank_file(square):
        return int(square // 16), square % 16

    @staticmethod
    def to_string(square):
        rank, file = Square.to_rank_file(square)
        r = rank + 1
        f = files[file]
        return f + str(r)
