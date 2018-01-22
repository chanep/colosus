from typing import overload

import numpy as np

from colosus.game.model_position import ModelPosition
from .square import Square
from .side import Side



class Position:
    RANKS_I = 0
    FILES_I = 1
    DIAG_DOWN_I = 2
    DIAG_UP_I = 3
    BOARDS_COUNT = 4
    B_SIZE = 16
    DIAGS = (B_SIZE - 5 + 1) * 2 - 1  # 23 diagonals
    DIAG_LEN = [5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5]

    def __init__(self, initialize_boards=True):
        self.side_to_move = Side.WHITE
        self.is_end = False
        self.score = None
        self.move_count = 0
        self.boards = [None] * self.BOARDS_COUNT
        # 23
        if initialize_boards:
            self.boards[self.RANKS_I] = np.zeros((2, self.B_SIZE), np.uint16)
            self.boards[self.FILES_I] = np.zeros((2, self.B_SIZE), np.uint16)
            self.boards[self.DIAG_DOWN_I] = np.zeros((2, self.DIAGS), np.uint16)
            self.boards[self.DIAG_UP_I] = np.zeros((2, self.DIAGS), np.uint16)

    def clone(self):
        new_pos = Position(False)
        new_pos.side_to_move = self.side_to_move
        new_pos.is_end = self.is_end
        new_pos.score = self.score
        new_pos.move_count = self.move_count
        for i in range(self.BOARDS_COUNT):
            new_pos.boards[i] = np.copy(self.boards[i])
        return new_pos

    def to_model_position(self):
        model_board = np.zeros((Side.COUNT, self.B_SIZE, self.B_SIZE), np.uint8)
        for side in range(Side.COUNT):
            b_index = self._get_board_index(side)
            for r in range(self.B_SIZE):
                bit_rank = self.boards[self.RANKS_I][side, r]
                aux = np.array([bit_rank], dtype=">u2")
                aux2 = aux.view(np.uint8)
                model_board[b_index, r, :] = np.flip(np.unpackbits(aux2), axis=0)

        return ModelPosition(model_board)

    def _get_board_index(self, side):
        return self.side_to_move ^ side

    def switch_side(self):
        self.side_to_move = self.side_to_move.change()

    def _get_coords(self, rank, file):
        rank_coords = (rank, file)

        file_coords = (file, rank)

        diag_down_index = self.B_SIZE - 5 - rank + file
        diag_down_bit = min(rank, file)
        diag_down_coords = None
        if 0 <= diag_down_index < self.DIAGS:
            diag_down_coords = (diag_down_index, diag_down_bit)

        diag_up_index = rank + file - 5
        diag_up_bit = file - min(rank + file - 15, 0)
        diag_up_coords = None
        if 0 <= diag_up_index < self.DIAGS:
            diag_up_coords = (diag_up_index, diag_up_bit)

        return [rank_coords, file_coords, diag_down_coords, diag_up_coords]

    def piece_at(self, side, rank, file):
        return self.boards[self.RANKS_I][side, rank] & (1 << file) != 0

    @overload
    def put_piece(self, side, square):
        ...

    @overload
    def put_piece(self, side, rank, file):
        ...

    def put_piece(self, side, rank, file=None):
        if file is None:
            rank, file = Square.to_rank_file(rank)
        coords = self._get_coords(rank, file)
        for i in range(self.BOARDS_COUNT):
            coord = coords[i]
            if coord is not None:
                r, f = coord
                self.boards[i][side, r] |= 1 << f
        self.move_count += 1

    def is_legal(self, move):
        r, f = Square.to_rank_file(move)
        rank = self.boards[self.RANKS_I][Side.WHITE,  r] | self.boards[self.RANKS_I][Side.BLACK,  r]
        if rank & (1 << f) != 0:
            return False
        else:
            return True

    def legal_moves(self):
        moves = []
        for m in range(self.B_SIZE * self.B_SIZE):
            if self.is_legal(m):
                moves.append(m)
        return moves

    def move(self, move):
        side = self.side_to_move
        new_pos = self.clone()
        new_pos.put_piece(side, move)
        new_pos.switch_side()
        new_pos._check_end(move)
        return new_pos

    def _check_end(self, last_move):
        if self.move_count >= (self.B_SIZE * self.B_SIZE):
            self.is_end = True
            self.score = 0
        elif self.check_win(last_move):
            self.is_end = True
            self.score = 1

    def check_win(self, last_move):
        side = self.side_to_move.change()
        r, f = Square.to_rank_file(last_move)
        coords = self._get_coords(r, f)
        mask = 0b11111
        shifts = self.B_SIZE - 5
        for i in range(self.BOARDS_COUNT):
            coord = coords[i]
            if coord is not None:
                index, bit = coord
                if i > 1:
                    shifts = self.DIAG_LEN[index]
                for s in range(shifts):
                    shifted_mask = mask << s
                    if self.board[i][side, index] & shifted_mask == shifted_mask:
                        return True

        return False

    def print(self):
        p_str = ['O', 'X']
        for r in reversed(range(self.B_SIZE)):
            rank_str = ''
            for f in range(self.B_SIZE):
                if self.piece_at(Side.WHITE, r, f):
                    rank_str += p_str[Side.WHITE] + ' '
                elif self.piece_at(Side.BLACK, r, f):
                    rank_str += p_str[Side.BLACK] + ' '
                else:
                    rank_str += '- '
            rank_str += str(r)
            print(rank_str)
        print('0 1 2 3 4 5 6 7 8 9101112131415')









