class Move:
    @staticmethod
    def from_squares(orig, dest):
        return orig + dest << 6

    @staticmethod
    def to_squares(move):
        return move & 63, move >> 6
