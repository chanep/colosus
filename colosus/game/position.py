import numpy as np
import math
from .move import Move
from .square import Square
from .side import Side
from .piece import Piece


class Position:
    def __init__(self):
        self.board = np.zeros((Side.COUNT * Piece.COUNT, 8, 8), np.uint8)
        self.side_to_move = Side.WHITE
        self.is_end = False
        self.score = None
        self.move_count = 0
        self.k = [None, None]
        self.r = [None, None]
        self._attacks = [None, None]

    def clone(self):
        new_pos = Position()
        new_pos.board = np.copy(self.board)
        new_pos.side_to_move = self.side_to_move
        new_pos.is_end = self.is_end
        new_pos.score = self.score
        new_pos.move_count = self.move_count
        new_pos.k = list(self.k)
        new_pos.r = list(self.r)
        return new_pos

    def _get_board_index(self, side, piece):
        return (self.side_to_move ^ side) * Side.COUNT + piece

    def piece_at(self, side, rank, file):
        square = Square.square(rank, file)
        return self.piece_at(side, square)
    
    def piece_at(self, side, square):
        if self.k[side] == square:
            return Piece.KING
        elif self.r[side] == square:
            return Piece.ROOK
        return None

    def switch_side(self):
        self.board = np.concatenate((self.board[Piece.COUNT:Piece.COUNT * 2], self.board[0:Piece.COUNT]))
        self.side_to_move = self.side_to_move.change()

    @staticmethod
    def _get_rank_file(piece_board):
        if np.sum(piece_board) == 0:
            return None, None
        rank = np.sum(np.sum(piece_board, axis=1) * np.arange(8)).astype(np.uint8)
        file = np.sum(np.sum(piece_board, axis=0) * np.arange(8)).astype(np.uint8)
        return rank, file

    def _side_attacks(self, side):
        if self._attacks[side] is not None:
            return self._attacks[side]
        k_attacks = []

        # king
        k = self.k[side]
        k_rank, k_file = Square.to_rank_file(k)
        for rank in range(max(0, k_rank - 1), min(8, k_rank + 2)):
            for file in range(max(0, k_file - 1), min(8, k_file + 2)):
                if not (rank == k_rank and file == k_file):
                    k_attacks.append(Square.square(rank, file))

        #rook
        r_attacks = []
        r = self.r[side]
        if r is not None:
            r_rank, r_file = Square.to_rank_file(r)
            r_rank_start = 0
            r_rank_end = 8
            r_file_start = 0
            r_file_end = 8
            if r_rank == k_rank:
                if k_file > r_file:
                    r_file_end = k_file
                else:
                    r_file_start = k_file
            if r_file == k_file:
                if k_file > r_file:
                    r_rank_end = k_rank
                else:
                    r_rank_start = k_rank

            for rank in range(r_rank_start, r_rank_end):
                if rank != r_rank:
                    r_attacks.append(Square.square(rank, r_file))

            for file in range(r_file_start, r_file_end):
                if file != r_file:
                    r_attacks.append(Square.square(r_rank, file))

        self._attacks[side] = k_attacks + r_attacks
        return self._attacks[side]

    def put_piece(self, side, piece, rank, file):
        square = Square.square(rank, file)
        self.put_piece(side, piece, square)

    def put_piece(self, side, piece, square):
        if piece == Piece.KING:
            self.k[side] = square
        elif piece == Piece.ROOK:
            self.r[side] = square

    def remove_piece(self, rank, file):
        square = Square.square(rank, file)
        self.remove_piece(square)

    def remove_piece(self, square):
        if self.k[0] == square:
            self.k[0] = None
        elif self.k[1] == square:
            self.k[1] = None
        elif self.r[0] == square:
            self.r[0] = None
        elif self.r[1] == square:
            self.r[1] = None

    def is_legal(self, move):
        side = self.side_to_move
        orig, dest = Move.to_squares(move)
        orig_rank, orig_file = Square.to_rank_file(orig)
        dest_rank, dest_file = Square.to_rank_file(dest)
        if orig_rank == dest_rank and orig_file == dest_file:
            return False

        # king
        if self.k[side] == orig:
            if abs(dest_rank - orig_rank) > 1 or abs(dest_file - orig_file) > 1:
                return False
            return dest not in self._side_attacks(side.change())

        # rook
        if self.r[side] == orig:
            if orig_rank != dest_rank and orig_file != dest_file:
                return False
            k = self.k[side]
            k_rank, k_file = Square.to_rank_file(k)
            if (orig_rank == dest_rank and orig_rank == k_rank and
                    min(orig_file, dest_file) <= k_file <= max(orig_file, dest_file)):
                return False
            if (orig_file == dest_file and orig_file == k_file and
                    min(orig_rank, dest_rank) <= k_rank <= max(orig_rank, dest_rank)):
                return False
            return True

        return False

    def legal_moves(self):
        moves = []
        for orig in range(64):
            side = self.side_to_move
            if self.k[side] != orig and self.r[side] != orig:
                continue
            for dest in range(64):
                if dest == orig:
                    continue
                move = Move.from_squares(orig, dest)
                if self.is_legal(move):
                    moves.append(move)
        return moves

    def move(self, move):
        side = self.side_to_move
        orig, dest = Move.to_squares(move)
        new_pos = self.clone()
        piece = new_pos.piece_at(side, orig)
        new_pos.remove_piece(orig)
        new_pos.remove_piece(dest)
        new_pos.put_piece(side, piece, dest)
        new_pos.switch_side()
        new_pos.move_count = self.move_count + 1

        new_pos._check_end()

        return new_pos

    def _check_end(self):
        if self.move_count >= 100:
            self.is_end = True
            self.score = 0
        elif self.checkmate():
            print("Mate")
            self.is_end = True
            self.score = -1
        elif np.sum(self.board) <= 2:
            self.is_end = True
            self.score = 0

    def in_check(self, side=None):
        if side is None:
            side = self.side_to_move
        k = self.k[side]
        other_attacks = self._side_attacks(side.change())
        return k in other_attacks

    def checkmate(self):
        if not self.in_check():
            return False
        return len(self.legal_moves()) == 0

    def print(self):
        p_str = ['R', 'K']
        for r in reversed(range(8)):
            rank_str = ''
            for f in range(8):
                p = self.piece_at(Side.WHITE, r, f)
                if p is not None:
                    rank_str += p_str[p] + ' '
                    continue
                p = self.piece_at(Side.BLACK, r, f)
                if p is not None:
                    rank_str += p_str[p].lower() + ' '
                    continue
                rank_str += '- '
            print(rank_str)








