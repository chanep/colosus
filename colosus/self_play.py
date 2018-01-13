import numpy as np
import time

from colosus.colosus_model import ColosusModel
from .game.position import Position
from .state import State
from .searcher import Searcher
from .train_record import TrainRecord
from .train_record_set import TrainRecordSet


class SelfPlay:
    def play(self, games: int, iterations_per_move: int, initial_pos: Position, colosus, train_filename):
        train_record_set = TrainRecordSet()
        # initial_state = State(initial_pos, None, None, colosus)
        searcher = Searcher()
        mates = 0
        mc_mates = 0
        for i in range(games):
            state = State(initial_pos, None, None, colosus)
            # print("initial state N: " + str(state.N))
            end = False
            while not end:
                # start_time = time.time()
                policy, value, move, new_state = searcher.search(state, iterations_per_move)
                # print("time: " + str(time.time() - start_time))
                train_record = TrainRecord(state.position.to_model_position(), policy, value)
                train_record_set.append(train_record)
                new_state.parent = None
                end = new_state.is_end
                state = new_state
                mc = state.position.move_count
                # if mc % 20 == 0:
                #     print("move count: " + str(state.position.move_count))
            print("fin game " + str(i))
            # state.print()
            # print("position score: " + str(state.position.score) + " value root: " + str(value))
            if state.position.score != 0:
                mates += 1
                mc_mates += mc

            mates_rate = mates / (i + 1)
            mc_mean = mc_mates / max(1, mates)
            print("mates rate: {}, mc mean: {}".format(mates_rate, mc_mean))

        train_record_set.save_to_file(train_filename)
