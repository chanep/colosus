import numpy as np
import tensorflow as tf
import time

from colosus.colosus_model import ColosusModel
from colosus.colosus_model2 import ColosusModel2
from colosus.config import EvaluatorConfig, PlayerConfig
from colosus.game.position import Position
from colosus.game.square import Square
from colosus.player import Player
from colosus.player2 import Player2
from colosus.player_mp import PlayerMp
from colosus.searcher import Searcher
from colosus.searcher2 import Searcher2
from colosus.state import State
from colosus.state2 import State2


class Evaluator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.player = None
        self.player2 = None
        self.var = 0.0
        self.var2 = 0.0
        self.short_games = 0
        self.mc_total = 0
        self.final_positions = {}

    def evaluate(self, games: int, iterations: int, position_ini: Position, weights_filename, weights_filename2, iterations2=None):

        iterations_player2 = iterations
        if iterations2 is not None:
            iterations_player2 = iterations2
        if self.config.player2_is_mp:
            iterations_player2 = int(iterations * self.config.iterations_mp_factor)

        colosus = ColosusModel(self.config.colosus_config)
        colosus.build()
        if weights_filename is not None:
            colosus.load_weights(weights_filename)

        colosus2 = ColosusModel2(self.config.colosus2_config)
        colosus2.build()
        if weights_filename2 is not None:
            colosus2.load_weights(weights_filename2)

        self.player = Player(self.config.player_config, colosus)
        if self.config.player2_is_mp:
            self.player2 = PlayerMp(self.config.player2_config, colosus2)
        else:
            self.player2 = Player2(self.config.player2_config, colosus2)

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
            game_score_2, game_mc = self.play_game(iterations, iterations_player2, game_num, position_ini)
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
        print(f"short games: {self.short_games}")
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

    def play_game(self, iterations: int, iterations_player2: int, game_num: int, position_ini: Position):
        move_num = 0
        end = False
        position = position_ini
        self.player.new_game(position.clone(), iterations)
        self.player2.new_game(position.clone(), iterations_player2)
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
            # print('')

            end = position.is_end
            if end:
                if move_num < 14:
                    self.short_games += 1

                position.print()
                # print(f"move: {move_num + 1}, var1: {self.var}, var2: {self.var2}")
                # win_line = position.win_line()
                # print(win_line)

                score = (-position.score + 1) / 2

                position_hash = self.hash_position(position)
                if position_hash not in self.final_positions.keys():
                    self.final_positions[position_hash] = position.clone()
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
        position_hash = 0
        for b in position.boards:
            position_hash ^= hash(str(b))
        return str(position_hash)

    def print_children(self, state: State):
        for m in range(len(state.children())):
            c = state.children()[m]
            if c is not None:
                print("{} N: {}, W: {:.3g}, Q: {:.3g}, p:{:.3g}".format(Square.to_string(m), c.N, c.W, c.Q, c.P))




