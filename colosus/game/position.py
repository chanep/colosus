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
    B_SIZE = 15
    DIAGS = (B_SIZE - 5 + 1) * 2 - 1  # 21 diagonals
    DIAG_LEN = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
    DISTANCE_MOVE2 = 5

    def __init__(self, initialize_boards=True):
        self.side_to_move = Side.BLACK
        self.is_end = False
        self.score = None
        self.move_count = 0
        self.boards = [None] * self.BOARDS_COUNT
        if initialize_boards:
            self.boards[self.RANKS_I] = np.zeros((2, self.B_SIZE), np.int)  # better bitwise performance than uint16
            self.boards[self.FILES_I] = np.zeros((2, self.B_SIZE), np.int)
            self.boards[self.DIAG_DOWN_I] = np.zeros((2, self.DIAGS), np.int)
            self.boards[self.DIAG_UP_I] = np.zeros((2, self.DIAGS), np.int)

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
                model_board[b_index, r, :] = np.flip(np.unpackbits(aux2), axis=0)[0:self.B_SIZE]

        return ModelPosition(model_board)

    def _get_board_index(self, side):
        return self.side_to_move ^ side

    def switch_side(self):
        self.side_to_move = self.side_to_move.change()
        if self.score is not None:
            self.score = - self.score

    def _get_coords(self, rank, file):
        rank_coords = (rank, file)

        file_coords = (file, rank)

        diag_down_index = self.B_SIZE - 5 - rank + file
        diag_down_bit = min(rank, file)
        diag_down_coords = None
        if 0 <= diag_down_index < self.DIAGS:
            diag_down_coords = (diag_down_index, diag_down_bit)

        diag_up_index = rank + file - 4
        diag_up_bit = file - max(rank + file - self.B_SIZE + 1, 0)
        diag_up_coords = None
        if 0 <= diag_up_index < self.DIAGS:
            diag_up_coords = (diag_up_index, diag_up_bit)

        return [rank_coords, file_coords, diag_down_coords, diag_up_coords]

    def _coord_to_rank_file(self, board, line, bit):
        if board == 0:
            return line, bit
        elif board == 1:
            return bit, line
        elif board == 2:
            rank = max(0, self.B_SIZE - 5 - line) + bit
            file = line + 5 + rank - self.B_SIZE
            return rank, file
        else:
            rank = min(self.B_SIZE - 1, line + 4) - bit
            file = line - rank + 4
            return rank, file

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
            square = rank
            rank, file = Square.to_rank_file(rank)
        else:
            square = Square.square(rank, file)
        coords = self._get_coords(rank, file)
        for i in range(self.BOARDS_COUNT):
            coord = coords[i]
            if coord is not None:
                r, f = coord
                self.boards[i][side, r] |= (1 << f)
        self.move_count += 1
        self._check_end(square)

    def is_legal_colosus3(self, move):
        r, f = Square.to_rank_file(move)
        mid = int(self.B_SIZE / 2)

        if self.move_count == 0:
            return r == mid and f == mid

        rank = self.boards[self.RANKS_I][Side.WHITE,  r] | self.boards[self.RANKS_I][Side.BLACK,  r]  # 8%
        if rank & (1 << f) != 0:
            return False

        if self.move_count == 2:
            return (abs(r - mid) == self.DISTANCE_MOVE2 and abs(f - mid) <= self.DISTANCE_MOVE2) or \
                   (abs(r - mid) <= self.DISTANCE_MOVE2 and abs(f - mid) == self.DISTANCE_MOVE2)

        # 80%
        for i in range(max(0, r - 2), min(self.B_SIZE, r + 3)):  # 8%
            rank = self.boards[self.RANKS_I][Side.WHITE, i] | self.boards[self.RANKS_I][Side.BLACK, i]  # 33%
            mask = (1 << f) | (1 << min(self.B_SIZE - 1, f + 1)) | (1 << max(0, f - 1)) | \
                   (1 << min(self.B_SIZE - 1, f + 2)) | (1 << max(0, f - 2)) # 37%
            if rank & mask != 0:
                return True
        return False

    def is_legal_colosus(self, move):
        r, f = Square.to_rank_file(move)
        mid = int(self.B_SIZE / 2)

        if self.move_count == 0:
            return r == mid and f == mid

        ranks = self.boards[self.RANKS_I]

        rank = ranks[Side.WHITE,  r] | ranks[Side.BLACK,  r]

        if rank & (1 << f) != 0:
            return False

        if self.move_count == 2:
            return (abs(r - mid) == self.DISTANCE_MOVE2 and abs(f - mid) <= self.DISTANCE_MOVE2) or \
                   (abs(r - mid) <= self.DISTANCE_MOVE2 and abs(f - mid) == self.DISTANCE_MOVE2)

        all = 65535
        i_min = f - 2
        if i_min < 0:
            i_min = 0
        i_max = f + 2
        if i_max > self.B_SIZE - 1:
            i_max = self.B_SIZE - 1
        mask = (all << i_min) & (all >> (self.B_SIZE - i_max))

        i_min = r - 2
        if i_min < 0:
            i_min = 0
        i_max = r + 3
        if i_max > self.B_SIZE:
            i_max = self.B_SIZE
        for i in range(i_min, i_max):
            if mask & (ranks[Side.WHITE, i] | ranks[Side.BLACK, i]) != 0:  # 43%
                return True
        return False

    def is_legal(self, move):
        r, f = Square.to_rank_file(move)
        mid = int(self.B_SIZE / 2)

        if self.move_count == 0:
            return r == mid and f == mid

        rank = self.boards[self.RANKS_I][Side.WHITE,  r] | self.boards[self.RANKS_I][Side.BLACK,  r]
        if rank & (1 << f) != 0:
            return False

        if self.move_count == 2:
            return abs(r - mid) >= self.DISTANCE_MOVE2 or abs(f - mid) >= self.DISTANCE_MOVE2

        return True

    def esta_pegada(self, move):
        r, f = Square.to_rank_file(move)
        for i in range(max(0, r - 1), min(self.B_SIZE, r + 2)):
            rank = self.boards[self.RANKS_I][Side.WHITE, i] | self.boards[self.RANKS_I][Side.BLACK, i]
            mask = (1 << f) | (1 << min(self.B_SIZE - 1, f + 1)) | (1 << max(0, f - 1))
            if rank & mask != 0:
                return True
        return False

    def legal_moves(self):
        moves = []
        for m in range(self.B_SIZE * self.B_SIZE):
            # if self.is_legal_colosus(m) != self.is_legal_colosus2(m):
            #     raise Exception
            if self.is_legal_colosus(m):
                moves.append(m)
        return moves

    def move(self, move):
        side = self.side_to_move
        new_pos = self.clone()
        new_pos.put_piece(side, move)
        new_pos.switch_side()
        return new_pos

    def _check_end(self, last_move):
        if self.check_win(last_move):
            self.is_end = True
            self.score = 1
        elif self.move_count >= (self.B_SIZE * self.B_SIZE):
            self.is_end = True
            self.score = 0

    def win_line(self):
        win_line = []
        mask = 0b11111
        overline_mask = 0b111111
        shifts = self.B_SIZE - 5 + 1
        for i in range(self.BOARDS_COUNT):
            board = self.boards[i]
            for s in range(Side.COUNT):
                lines = board[s, :]
                for l in range(len(lines)):
                    line = lines[l]
                    if i > 1:
                        shifts = self.DIAG_LEN[l] - 5 + 1
                    for s in range(shifts):
                        shifted_mask = mask << s
                        shifted_overline_mask = shifted_overline_mask2 = overline_mask << s
                        if s > 0:
                            shifted_overline_mask2 = overline_mask << (s - 1)
                        if line & shifted_mask == shifted_mask and \
                                line & shifted_overline_mask != shifted_overline_mask and \
                                line & shifted_overline_mask2 != shifted_overline_mask2:
                            for j in range(5):
                                rank, file = self._coord_to_rank_file(i, l, s + j)
                                win_line.append((rank, file))
                            return win_line
        return win_line

    def check_win(self, last_move):
        side = self.side_to_move
        r, f = Square.to_rank_file(last_move)
        coords = self._get_coords(r, f)
        # mask = 0b11111
        # shifts = self.B_SIZE - 5 + 1
        for i in range(self.BOARDS_COUNT):
            coord = coords[i]
            if coord is not None:
                index, bit = coord
                line = self.boards[i][side, index]
                bin_line = bin(line)
                if "11111" in bin_line and "111111" not in bin_line:
                    return True
                # if i > 1:
                #     shifts = self.DIAG_LEN[index] - 5 + 1
                # for s in range(shifts):
                #     shifted_mask = mask << s
                #     if line & shifted_mask == shifted_mask:
                #         return True
        return False

    def hash(self):
        return str(hash(str(self.to_model_position().board)))

    def hash_rotations(self):
        def hash_board(board):
            return str(hash(str(board)))

        def flip_board(board):
            board_f = []
            for s in range(2):
                board_f.append(np.fliplr(board[s]))
            return np.stack(board_f)

        def rot90_board(board):
            board_f = []
            for s in range(2):
                board_f.append(np.rot90(board[s]))
            return np.stack(board_f)

        board = self.to_model_position().board
        hashes = []
        for i in range(4):
            hashes.append(hash_board(board))
            board_f = flip_board(board)
            hashes.append(hash_board(board_f))
            if i < 3:
                board = rot90_board(board)

        return hashes

    def print(self):
        p_str = ['X', 'O']
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
        print('0 1 2 3 4 5 6 7 8 9 1011121314')









