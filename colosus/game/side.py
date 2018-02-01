from enum import IntEnum


class Side(IntEnum):
    BLACK = 0
    WHITE = 1
    COUNT = 2

    def change(self):
        if self == Side.WHITE:
            return Side.BLACK
        return Side.WHITE
