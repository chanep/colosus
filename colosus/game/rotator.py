import numpy as np

from typing import List
from colosus.game.model_position import ModelPosition
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.game.square import Square
from colosus.train_record import TrainRecord


class Rotator:
    def __init__(self):
        self.square_flip = flip = []
        self.square_rot90 = rot = []
        size = Position.B_SIZE
        for s in range(size * size):
            rank, file = Square.to_rank_file(s)
            board = np.zeros((size, size), np.uint8)
            board[rank, file] = 1
            board_flip = np.fliplr(board)
            board_rot = np.rot90(board)
            rank, file = Square.piece_rank_file(board_flip)
            square_flip = Square.square(rank, file)
            flip.append(square_flip)
            rank, file = Square.piece_rank_file(board_rot)
            square_rot = Square.square(rank, file)
            rot.append(square_rot)

    def _flip_policy(self, policy: np.array):
        policy_flip = np.zeros_like(policy)
        for m in range(len(policy)):
            move_flip = self.square_flip[m]
            policy_flip[move_flip] = policy[m]
        return policy_flip

    def _rot90_policy(self, policy: np.array):
        policy_rot = np.zeros_like(policy)
        for m in range(len(policy)):
            move_rot = self.square_rot90[m]
            policy_rot[move_rot] = policy[m]
        return policy_rot

    def _flip_position(self, position: ModelPosition):
        board = []
        for s in range(Side.COUNT):
            board.append(np.fliplr(position.board[s]))
        board = np.stack(board)
        return ModelPosition(board)

    def _rot90_position(self, position: ModelPosition):
        board = []
        for s in range(Side.COUNT):
            board.append(np.rot90(position.board[s]))
        board = np.stack(board)
        return ModelPosition(board)

    def rotations(self, record: TrainRecord) -> List[TrainRecord]:
        rotations = []

        position = record.position
        policy = record.policy
        value = record.value

        for i in range(4):
            record = TrainRecord(position, policy, value)
            rotations.append(record)
            policy_f = self._flip_policy(policy)
            position_f = self._flip_position(position)
            record = TrainRecord(position_f, policy_f, value)
            rotations.append(record)

            if i < 3:
                position = self._rot90_position(position)
                policy = self._rot90_policy(policy)

        return rotations







