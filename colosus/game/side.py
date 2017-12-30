from enum import IntEnum


class Side(IntEnum):
    WHITE = 0
    BLACK = 1
    COUNT = 2

    def change(self):
        if self == Side.WHITE:
            return Side.BLACK
        return Side.WHITE
