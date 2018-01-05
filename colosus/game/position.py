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

    def _get_board_index(self, side, piece):
        return (self.side_to_move ^ side) * Side.COUNT + piece

    def has_piece(self, side, piece, rank, file):
        return self.board[self._get_board_index(side, piece), rank, file] == 1
    
    def piece_at(self, side, rank, file):
        for p in range(Piece.COUNT):
            if self.has_piece(side, p, rank, file):
                return p
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
        a = np.zeros((8, 8), np.uint8)

        # king
        k_board = self.board[self._get_board_index(side, Piece.KING)]
        k_rank, k_file = self._get_rank_file(k_board)
        b = np.zeros((8, 8), np.uint8)
        b[max(0, k_rank - 1):min(8, k_rank + 2), max(0, k_file - 1):min(8, k_file + 2)] = 1
        b[k_rank, k_file] = 0
        a = a + b

        #rook
        r_board = self.board[self._get_board_index(side, Piece.ROOK)]
        r_rank, r_file = self._get_rank_file(r_board)
        if r_rank is not None:
            b = np.zeros((8, 8), np.uint8)
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
            b[r_rank,r_file_start:r_file_end] = 1
            b[r_rank_start:r_rank_end,r_file] = 1
            b[r_rank, r_file] = 0
            a = a + b
        return a

    def put_piece(self, side, piece, rank, file):
        self.board[self._get_board_index(side, piece), rank, file] = 1

    def remove_piece(self, rank, file):
        self.board[:, rank , file] = 0

    def clone(self):
        cloned = Position()
        cloned.board = np.copy(self.board)
        cloned.side_to_move = self.side_to_move
        return cloned

    def is_legal(self, move):
        side = self.side_to_move
        orig, dest = Move.to_squares(move)
        orig_rank, orig_file = Square.to_rank_file(orig)
        dest_rank, dest_file = Square.to_rank_file(dest)
        if orig_rank == dest_rank and orig_file == dest_file:
            return False

        # king
        if self.has_piece(side, Piece.KING, orig_rank, orig_file):
            if abs(dest_rank - orig_rank) > 1 or abs(dest_file - orig_file) > 1:
                return False
            other_attacks = self._side_attacks(side.change())
            return other_attacks[dest_rank, dest_file] == 0

        # rook
        if self.has_piece(side, Piece.ROOK, orig_rank, orig_file):
            if orig_rank != dest_rank and orig_file != dest_file:
                return False
            path = np.zeros((8, 8), np.uint8)
            path[min(orig_rank, dest_rank):max(orig_rank, dest_rank) + 1, min(orig_file, dest_file):max(orig_file, dest_file) + 1] = 1
            k_board = self.board[self._get_board_index(side, Piece.KING)]
            return np.sum(path * k_board) == 0
        
        return False



    def legal_moves(self):
        moves = []
        for orig in range(64):
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
        orig_rank, orig_file = Square.to_rank_file(orig)
        dest_rank, dest_file = Square.to_rank_file(dest)
        new_pos = self.clone()
        piece = new_pos.piece_at(side, orig_rank, orig_file)
        new_pos.remove_piece(orig_rank, orig_file)
        new_pos.remove_piece(dest_rank, dest_file)
        new_pos.put_piece(side, piece, dest_rank, dest_file)
        new_pos.switch_side()
        new_pos.move_count += 1

        self._check_end()

        return new_pos

    def _check_end(self):
        if self.move_count >= 100:
            self.is_end = True
            self.score = 0
        elif self.checkmate():
            self.is_end = True
            self.score = -1

    def in_check(self, side=None):
        if side is None:
            side = self.side_to_move
        k_board = self.board[self._get_board_index(side, Piece.KING)]
        other_attacks = self._side_attacks(side.change())
        return np.sum(k_board * other_attacks) != 0

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








