import numpy as np
import time
import threading
import tensorflow as tf

from colosus.colosus_model import ColosusModel
from colosus.config import SearchConfig
from .game.position import Position
from .state import State
from .searcher import Searcher
from .train_record import TrainRecord
from .train_record_set import TrainRecordSet


class SelfPlay:
    def play(self, games: int, iterations_per_move: int, initial_pos: Position, train_filename, weights_filename, update_stats=None):
        train_record_set = TrainRecordSet()

        colosus = ColosusModel()
        colosus.build()
        if weights_filename is not None:
            colosus.model.load_weights(weights_filename)

        searcher = Searcher(SearchConfig())
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

            if update_stats is None:
                print("fin game " + str(i))
                if state.position.score != 0:
                    mates += 1
                    mc_mates += mc
                mates_rate = mates / (i + 1)
                mc_mean = mc_mates / max(1, mates)
                print("mates rate: {}, mc mean: {}".format(mates_rate, mc_mean))
            else:
                mate = state.position.score != 0
                update_stats(mate, mc)

        train_record_set.save_to_file(train_filename)

    def play_parallel(self, games: int, iterations_per_move: int, initial_pos: Position, train_filename, threads: int, weights_filename=None):
        lock = threading.Lock()
        games_played = 0
        mates = 0
        mc_mates = 0

        def update_stats(mate, move_count):
            nonlocal mates, games_played, mc_mates

            lock.acquire()
            games_played += 1
            if mate:
                mates += 1
                mc_mates += move_count
            mates_rate = mates / (games_played + 1)
            mc_mean = mc_mates / max(1, mates)
            print("fin game " + str(games_played))
            print("mates rate: {}, mc mean: {}".format(mates_rate, mc_mean))
            lock.release()

        workers = []
        games_per_worker = games // threads
        games_first_worker = games_per_worker + (games - games_per_worker * threads)
        worker_games = [games_per_worker] * threads
        worker_games[0] = games_first_worker
        train_filename_parts = train_filename.split(".")
        for i in range(threads):
            worker_train_filename = train_filename_parts[0] + str(i) + "." + train_filename_parts[1]
            play_args = (worker_games[i], iterations_per_move, initial_pos.clone(), worker_train_filename, weights_filename,
                         update_stats)
            w = threading.Thread(target=self.play, args=play_args)
            workers.append(w)
            w.start()

        for w in workers:
            w.join()

        print("fin play!")

