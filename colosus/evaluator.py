import numpy as np
import tensorflow as tf

from colosus.colosus_model import ColosusModel
from colosus.colosus_model2 import ColosusModel2
from colosus.config import EvaluatorConfig
from colosus.game.position import Position
from colosus.searcher import Searcher
from colosus.searcher2 import Searcher2
from colosus.state import State
from colosus.state2 import State2


class Evaluator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.colosus = None
        self.colosus2 = None
        self.searcher = None
        self.searcher2 = None

    def evaluate(self, games: int, iterations: int, position_ini: Position, weights_filename, weights_filename2):

        self.colosus = ColosusModel(self.config.colosus_config)
        self.colosus.build()
        if weights_filename is not None:
            self.colosus.load_weights(weights_filename)

        self.colosus2 = ColosusModel2(self.config.colosus_config)
        self.colosus2.build()
        if weights_filename2 is not None:
            self.colosus2.load_weights(weights_filename2)

        self.searcher = Searcher(self.config.search_config)
        self.searcher2 = Searcher2(self.config.search_config)

        total_score_1 = 0.0
        total_score_2 = 0.0

        mc_win_1 = 0
        mc_win_2 = 0

        wins_1 = 0
        wins_2 = 0
        win_rate_2 = 0

        for game_num in range(games):
            game_score_2, game_mc = self.play_game(iterations, game_num, position_ini)
            total_score_1 += 1 - game_score_2
            total_score_2 += game_score_2
            win_rate_2 = total_score_2 / (game_num + 1)
            if game_score_2 == 1:
                wins_2 += 1
                mc_win_2 += game_mc
            elif game_score_2 == 0:
                wins_1 += 1
                mc_win_1 += game_mc

            mc_win_1_mean = 0 if wins_1 == 0 else mc_win_1 / wins_1
            mc_win_2_mean = 0 if wins_2 == 0 else mc_win_2 / wins_2

            print("game: {}, {}-{}, wr2:{:.1%}, mc1: {:.3g}, mc2: {:.3g}".format(game_num + 1, total_score_1, total_score_2,
                                                                             win_rate_2, mc_win_1_mean, mc_win_2_mean))

        return win_rate_2

    def is_two(self, game_num: int, move_num: int):
        return (game_num + move_num) % 2 != 0

    def get_state(self, game_num: int, move_num: int, position: Position):
        if self.is_two(game_num, move_num):
            return State2(position.clone(), None, None, self.colosus2, self.config.state_config)
        else:
            return State(position.clone(), None, None, self.colosus, self.config.state_config)

    def get_searcher(self, game_num: int, move_num: int):
        if self.is_two(game_num, move_num):
            return self.searcher2
        else:
            return self.searcher

    def play_game(self, iterations: int, game_num: int, position_ini: Position):
        move_num = 0
        end = False
        position = position_ini
        while not end:
            state = self.get_state(game_num, move_num, position)
            searcher = self.get_searcher(game_num, move_num)
            policy, value, move, new_state = searcher.search(state, iterations)
            position = new_state.position()
            # position.print()
            # print('')
            end = position.is_end
            if end:
                score = (-position.score + 1) / 2
                if self.is_two(game_num, move_num):
                    return score, move_num
                else:
                    return (1 - score), move_num
            else:
                move_num += 1




