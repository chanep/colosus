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

    def clone(self):
        cloned = Position()
        cloned.board = np.copy(self.board)
        cloned.side_to_move = self.side_to_move
        return cloned

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
                            if (orig_rank == dest_rank and (w_king_rank != dest_rank or not(min(orig_file, dest_file) <= w_king_file <= max(orig_file, dest_file)))) or \
                                    (orig_file == dest_file and (w_king_file != dest_file or not(min(orig_rank, dest_rank) <= w_king_rank <= max(orig_rank, dest_rank)))):
                                moves.append(move)
                    if self.board[Side.WHITE, Piece.KING] | orig_board != 0:
                        if (math.fabs(orig_rank - dest_rank) <= 1 and math.fabs(orig_file - dest_file) <= 1) and \
                                (math.fabs(b_king_rank - dest_rank) > 1 or math.fabs(b_king_file - dest_file) > 1):
                            moves.append(move)
                else:
                    if self.board[Side.BLACK, Piece.KING] | orig_board != 0:
                        if (math.fabs(orig_rank - dest_rank) <= 1 and math.fabs(orig_file - dest_file) <= 1) and \
                                (math.fabs(w_king_rank - dest_rank) > 1 or math.fabs(w_king_file - dest_file) > 1) and\
                                (dest_rank != w_rook_rank and dest_file != w_rook_file):
                            moves.append(move)

    def move(self, move):
        orig, dest = Move.to_squares(move)
        orig_board = Square.board(orig)
        dest_board = Square.board(dest)
        new_pos = self.clone()
        for p in range(Piece.COUNT):
            piece_boards = new_pos[new_pos.side_to_move]
            if piece_boards[p] | orig_board != 0:
                piece_boards[p] = piece_boards[p] ^ orig_board ^ dest_board
                new_pos.side_to_move = new_pos.side_to_move.change()
                return new_pos
        raise Exception('Invalid move ' + Move.to_string(move))








