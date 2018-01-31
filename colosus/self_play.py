import numpy as np
import time
import threading
from multiprocessing import Process, Lock
import tensorflow as tf

from colosus.colosus_model import ColosusModel
from colosus.config import SearchConfig, SelfPlayConfig
from .game.position import Position
from .state import State
from .searcher import Searcher
from .train_record import TrainRecord
from .train_record_set import TrainRecordSet


class SelfPlay:
    def __init__(self, config: SelfPlayConfig):
        self.config = config

    def play(self, games: int, iterations_per_move: int, initial_pos: Position, train_filename, weights_filename=None, update_stats=None, colosus: ColosusModel = None):
        train_record_set = TrainRecordSet()

        if colosus is None:
            colosus = ColosusModel(self.config.colosus_config)
            colosus.build()
            if weights_filename is not None:
                colosus.load_weights(weights_filename)

        searcher = Searcher(self.config.search_config)
        wins = 0
        mc_wins = 0
        for i in range(games):
            state = State(initial_pos, None, None, colosus, self.config.state_config)
            # print("initial state N: " + str(state.N))
            end = False
            game_records = []
            while not end:
                # start_time = time.time()
                policy, value, move, new_state = searcher.search(state, iterations_per_move)
                # print("time: " + str(time.time() - start_time))
                train_record = TrainRecord(state.position().to_model_position(), policy, value)
                game_records.append(train_record)
                end = new_state.position().is_end
                state = new_state
                mc = state.position().move_count
                # state.position().print()
                # print("mc: {}".format(state.position().move_count))

            state.position().print()

            # z = - state.position().score
            # for j in reversed(range(len(game_records))):
            #     game_records[j].value = z
            #     z = -z

            train_record_set.extend(game_records)

            if update_stats is None:
                print("fin game " + str(i + 1))
                if state.position().score != 0:
                    wins += 1
                    mc_wins += mc
                    # state.position.print()
                wins_rate = wins / (i + 1)
                mc_mean = mc_wins / max(1, wins)
                print("wins rate: {:.1%}, mc mean: {:.3g}".format(wins_rate, mc_mean))
            else:
                mate = state.position().score != 0
                update_stats(mate, mc)

        train_record_set.save_to_file(train_filename)

    def play_parallel(self, games: int, iterations_per_move: int, initial_pos: Position, train_filename, threads: int, weights_filename=None):
        lock = threading.Lock()
        games_played = 0
        wins = 0
        mc_wins = 0

        def update_stats(mate, move_count):
            nonlocal wins, games_played, mc_wins
            games_played += 1
            lock.acquire()
            if mate:
                wins += 1
                mc_wins += move_count
                wins_rate = wins / (games_played)
            mc_mean = mc_wins / max(1, wins)
            print("fin game " + str(games_played))
            print("wins rate: {:.1%}, mc mean: {:.3g}".format(wins_rate, mc_mean))

            lock.release()

        colosus_config = self.config.colosus_config
        colosus_config.thread_safe = True
        colosus = ColosusModel(colosus_config)
        colosus.build()
        if weights_filename is not None:
            colosus.load_weights(weights_filename)


        workers = []
        games_per_worker = games // threads
        remaining_games = games % threads
        worker_games = [games_per_worker] * threads
        for g in range(remaining_games):
            worker_games[g] += 1

        train_filename_parts = train_filename.split(".")
        for i in range(threads):
            worker_train_filename = train_filename_parts[0] + "_" + str(i) + "." + train_filename_parts[1]
            play_args = (worker_games[i], iterations_per_move, initial_pos.clone(), worker_train_filename, weights_filename,
                         update_stats, colosus)
            w = threading.Thread(target=self.play, args=play_args)
            workers.append(w)
            w.start()

        for w in workers:
            w.join()

        print("fin play!")

