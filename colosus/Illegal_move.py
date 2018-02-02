class IllegalMove(Exception):
    def __init__(self, move):
        self.move = move
        self.value = f"{move} is illegal"

    def __str__(self):
        return repr(self.value)
