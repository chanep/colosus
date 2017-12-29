import numpy as np
import math
from .move import Move
from .square import Square
from .side import Side
from .piece import Piece


class Position:
    def __init__(self):
        self.board = np.empty((Side.COUNT, Piece.COUNT), np.uint64)
        self.side_to_move = Side.WHITE

    def put_piece(self, side, piece, rank, file):
        self.board[side, piece] = self.board[side, piece] | Square.board(rank, file)

    def legal_moves(self):
        moves = []
        for orig in range(64):
            for dest in range(64):
                if dest == orig:
                    continue
                move = Move.from_squares(orig, dest)
                orig_rank, orig_file = Square.to_rank_file(orig)
                dest_rank, dest_file = Square.to_rank_file(dest)
                orig_board = Square.board(orig)
                w_king_rank, w_king_file = Square.board_to_rank_file(self.board[Side.WHITE, Piece.KING])
                b_king_rank, b_king_file = Square.board_to_rank_file(self.board[Side.BLACK, Piece.KING])
                w_rook_rank, w_rook_file = Square.board_to_rank_file(self.board[Side.WHITE, Piece.ROOK])
                if self.side_to_move == Side.WHITE:
                    if self.board[Side.WHITE, Piece.ROOK] | orig_board != 0:
                        if orig_rank == dest_rank or orig_file == dest_file:
                            if (orig_rank == dest_rank and (w_king_rank != dest_rank or not(math.min(orig_file, dest_file) <= w_king_file <= math.max(orig_file, dest_file)))) or \
                                    (orig_file == dest_file and (w_king_file != dest_file or not(math.min(orig_rank, dest_rank) <= w_king_rank <= math.max(orig_rank, dest_rank)))):
                                moves.append(move)
                    if self.board[Side.WHITE, Piece.KING] | orig_board != 0:
                        if (math.abs(orig_rank - dest_rank) <= 1 and math.abs(orig_file - dest_file) <= 1) and \
                                (math.abs(b_king_rank - dest_rank) > 1 or math.abs(b_king_file - dest_file) > 1):
                            moves.append(move)
                else:
                    if self.board[Side.BLACK, Piece.KING] | orig_board != 0:
                        if (math.abs(orig_rank - dest_rank) <= 1 and math.abs(orig_file - dest_file) <= 1) and \
                                (math.abs(w_king_rank - dest_rank) > 1 or math.abs(w_king_file - dest_file) > 1) and\
                                (dest_rank != w_rook_rank and dest_file != w_rook_file):
                            moves.append(move)

    # def move(self, move):






