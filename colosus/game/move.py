from .square import Square


class Move:
    @staticmethod
    def from_squares(orig, dest):
        return orig + (dest << 6)

    @staticmethod
    def to_squares(move):
        return move & 63, move >> 6

    @staticmethod
    def to_string(move):
        orig, dest = Move.to_squares(move)
        return Square.to_string(orig) + Square.to_string(dest)