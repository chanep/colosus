import multiprocessing as mp
import numpy as np

from colosus.colosus_model import ColosusModel
from colosus.config import SelfPlayMpConfig, SearchConfig
from colosus.game.position import Position
from colosus.searcher import Searcher
from colosus.state import State
from colosus.train_record import TrainRecord
from colosus.train_record_set import TrainRecordSet
from datetime import datetime


class ColosusProxy:
    def __init__(self, conn):
        self.conn = conn

    def predict(self, position):
        self.conn.send(position)
        result = self.conn.recv()
        return result

    def close(self):
        self.conn.send("fin")
        self.conn.close()


class Stats:
    def __init__(self, total_games):
        self.total_games = total_games
        self.games_started = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)
        self.wins = mp.Value('i', 0)
        self.mc_total = mp.Value('i', 0)

    def should_continue(self):
        with self.games_started.get_lock():
            self.games_started.value += 1
            return self.games_started.value > self.total_games

    def update(self, win, mc):
        with self.games_played.get_lock():
            self.games_played.value += 1
            if win:
                self.wins.value += 1
                self.mc_total.value += mc
            mc_mean = 0 if self.wins.value == 0 else self.mc_total.value / self.wins.value
        print("games: {}, wins: {}, mc mean: {:.3g}\n".format(self.games_played.value, self.wins.value, mc_mean))


def get_time():
    return datetime.now().time().strftime("%H:%M:%S.%f")


class SelfPlayMp:
    def __init__(self, config: SelfPlayMpConfig):
        self.config = config

    def _play(self, id: int, iterations_per_move: int, initial_pos: Position, train_filename, colosus, stats):
        np.random.seed(id)

        train_record_set = TrainRecordSet()
        searcher = Searcher(self.config.search_config)
        while not stats.should_continue():
            state = State(initial_pos, None, None, colosus, self.config.state_config)
            end = False
            game_records = []
            while not end:
                policy, value, move, new_state = searcher.search(state, iterations_per_move)
                if new_state is None:
                    new_state = State(state.position().move(move), None, None, colosus, self.config.state_config)
                train_record = TrainRecord(state.position().to_model_position(), policy, value)
                game_records.append(train_record)
                new_state.parent = None
                end = new_state.position().is_end
                state = new_state
                mc = state.position().move_count
                # state.position().print()
                # print("mc: {}".format(state.position().move_count))

            state.position().print()

            train_record_set.extend(game_records)

            win = state.position().score != 0
            stats.update(win, mc)

        colosus.close()
        train_record_set.save_to_file(train_filename)

    def play(self, games: int, iterations_per_move: int, initial_pos: Position, train_filename, workers: int, weights_filename=None):
        colosus_config = self.config.colosus_config
        colosus = ColosusModel(colosus_config)
        colosus.build()
        if weights_filename is not None:
            colosus.load_weights(weights_filename)

        stats = Stats(games)

        processes = []
        conns = []

        train_filename_parts = train_filename.split(".")

        for id in range(workers):
            worker_train_filename = train_filename_parts[0] + "_" + str(id) + "." + train_filename_parts[1]
            server_conn, client_conn = mp.Pipe()
            colosusProxy = ColosusProxy(client_conn)
            args = (id, iterations_per_move, initial_pos.clone(), worker_train_filename,
                    colosusProxy, stats)
            p = mp.Process(target=self._play, args=args)
            processes.append(p)
            conns.append(server_conn)
            p.start()

        alive = workers
        while alive > 0:
            positions = []
            position_indexes = []
            for i in range(workers):
                c = conns[i]
                if c is not None:
                    try:
                        if c.poll():
                            pos = c.recv()
                            if isinstance(pos, str) and pos == "fin":
                                print("worker {} finish".format(i))
                                c.close()
                                conns[i] = None
                            else:
                                positions.append(pos)
                                position_indexes.append(i)
                    except EOFError:
                        print("se desconecto")
                        conns[i] = None

            if len(positions) > 0:
                result = colosus.predict_on_batch(positions)
                policies, values = result

                for i in range(len(positions)):
                    policy = policies[i]
                    value = values[i]
                    c = position_indexes[i]
                    conn = conns[c]
                    conn.send((policy, value))

            alive = sum(1 for c in conns if c is not None)

        print("fin play_mp")