import numpy as np
import tensorflow as tf
import time
from datetime import datetime


from colosus.colosus_model import ColosusModel
from colosus.colosus_model2 import ColosusModel2
from colosus.config import EvaluatorConfig, PlayerConfig, PlayerType
from colosus.game.position import Position
from colosus.game.square import Square
from colosus.player import Player
from colosus.player2 import Player2
from colosus.player_mb import PlayerMb
from colosus.player_mb2 import PlayerMb2
from colosus.state import State
from colosus.state2 import State2


class Evaluator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._initialize()

    def _initialize(self):
        self.player = None
        self.player2 = None
        self.var = 0.0
        self.var2 = 0.0
        self.short_games = 0
        self.mc_total = 0
        self.final_positions = {}
        self.final_position_rotations = {}

    def evaluate(self, games: int, iterations: int, position_ini: Position, weights_filename, weights_filename2,
                 iterations2=None, times_per_move: float = 1):
        self._initialize()

        seed = hash(str(datetime.now())) % (2 ** 32 - 1)
        np.random.seed(seed)

        iterations_player2 = iterations
        if iterations2 is not None:
            iterations_player2 = iterations2

        colosus = ColosusModel(self.config.colosus_config)
        colosus.build()
        if weights_filename is not None:
            colosus.load_weights(weights_filename)

        colosus2 = ColosusModel2(self.config.colosus2_config)
        colosus2.build()
        if weights_filename2 is not None:
            colosus2.load_weights(weights_filename2)

        if self.config.player_type == PlayerType.player:
            self.player = Player(self.config.player_config, colosus)
        elif self.config.player_type == PlayerType.player2:
            self.player = Player2(self.config.player_config, colosus)
        elif self.config.player_type == PlayerType.player_mb:
            self.player = PlayerMb(self.config.player_config, colosus)
        elif self.config.player_type == PlayerType.player_mb2:
            self.player = PlayerMb2(self.config.player_config, colosus)

        if self.config.player2_type == PlayerType.player:
            self.player2 = Player(self.config.player_config, colosus)
        elif self.config.player2_type == PlayerType.player2:
            self.player2 = Player2(self.config.player_config, colosus)
        elif self.config.player2_type == PlayerType.player_mb:
            self.player2 = PlayerMb(self.config.player_config, colosus)
        elif self.config.player2_type == PlayerType.player_mb2:
            self.player2 = PlayerMb2(self.config.player_config, colosus)

        total_score_1 = 0.0
        total_score_2 = 0.0

        mc_win_1 = 0
        mc_win_2 = 0

        wins_1 = 0
        wins_2 = 0
        win_rate_2 = 0
        black_score = 0

        start_time = time.time()

        for game_num in range(games):
            game_score_2, game_mc = self.play_game(iterations, iterations_player2, times_per_move, game_num, position_ini)
            total_score_1 += 1 - game_score_2
            total_score_2 += game_score_2
            win_rate_2 = total_score_2 / (game_num + 1)
            if game_score_2 == 1:
                wins_2 += 1
                mc_win_2 += game_mc
            elif game_score_2 == 0:
                wins_1 += 1
                mc_win_1 += game_mc

            if game_num % 2 != 0:
                black_score += game_score_2
            else:
                black_score += 1 - game_score_2

            mc_win_1_mean = 0 if wins_1 == 0 else mc_win_1 / wins_1
            mc_win_2_mean = 0 if wins_2 == 0 else mc_win_2 / wins_2
            win_rate_black = black_score / (game_num + 1)

            print("game: {}, {}-{}, wr2:{:.1%}, black:{:.1%}, mc1: {:.3g}, mc2: {:.3g}\n".format(game_num + 1, total_score_1, total_score_2,
                                                                             win_rate_2, win_rate_black, mc_win_1_mean, mc_win_2_mean))

        print(f"different final positions: {len(self.final_positions.keys())}")
        print(f"different final positions with rotations: {(len(self.final_position_rotations.keys()) / 8):.1f}")
        print(f"short games (<20): {self.short_games}")
        print(f"time: {time.time() - start_time}")
        print(f"mc_total: {self.mc_total}")

        return win_rate_2

    def is_two(self, game_num: int, move_num: int):
        return (game_num + move_num) % 2 != 0

    def get_player(self, game_num: int, move_num: int):
        if self.is_two(game_num, move_num):
            return self.player2
        else:
            return self.player

    def play_game(self, iterations: int, iterations_player2: int, time_per_move: float, game_num: int, position_ini: Position):
        move_num = 0
        end = False
        position = position_ini
        self.player.new_game(position.clone(), iterations, time_per_move)
        self.player2.new_game(position.clone(), iterations_player2, time_per_move)
        while not end:
            self.mc_total += 1
            player = self.get_player(game_num, move_num)
            policy, value, move, old_state, new_state = player.move()
            position = new_state.position()
            opponent: Player = self.get_player(game_num, move_num + 1)
            opponent.opponent_move(move)
            if self.is_two(game_num, move_num):
                self.var2 += np.var(policy)
            else:
                self.var += np.var(policy)

            # self.print_children(old_state)
            # position.print()
            # print('N: ' + str(old_state.N))
            # print('move count: ' + str(old_state.position().move_count))
            # old_state.print()
            # old_state.print_children_stats(20)

            end = position.is_end
            if end:

                # print("searches: " + str(self.player.searcher.hash_table.searches))
                # print("hits: " + str(self.player.searcher.hash_table.hits))

                if move_num < 20:
                    self.short_games += 1

                position.print()
                # print(f"move: {move_num + 1}, var1: {self.var}, var2: {self.var2}")
                # win_line = position.win_line()
                # print(win_line)

                score = (-position.score + 1) / 2

                position_hash = self.hash_position(position)
                if position_hash not in self.final_positions.keys():
                    self.final_positions[position_hash] = 1

                position_rotation_hashes = self.hash_position_rotations(position)
                for h in position_rotation_hashes:
                    if h not in self.final_position_rotations.keys():
                        self.final_position_rotations[h] = 1

                # else:
                #     print("same position")
                #     position.print()
                #     self.final_positions[position_hash].print()

                if self.is_two(game_num, move_num):
                    return score, move_num
                else:
                    return (1 - score), move_num
            else:
                move_num += 1

    def hash_position(self, position):
        return str(hash(str(position.to_model_position().board)))

    def hash_position_rotations(self, position):
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

        board = position.to_model_position().board
        hashes = []
        for i in range(4):
            hashes.append(hash_board(board))
            board_f = flip_board(board)
            hashes.append(hash_board(board_f))
            if i < 3:
                board = rot90_board(board)

        return hashes




