import multiprocessing as mp
import numpy as np

from colosus.colosus_model import ColosusModel
from colosus.config import SelfPlayConfig
from colosus.game.position import Position
from colosus.searcher_mb import SearcherMb
from colosus.state_mb import StateMb
from colosus.train_record import TrainRecord
from colosus.train_record_set import TrainRecordSet
from datetime import datetime


class ColosusProxy:
    def __init__(self, conn):
        self.conn = conn

    def predict_on_batch(self, positions):
        self.conn.send(positions)
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


class SelfPlayMpMb:
    def __init__(self, config: SelfPlayConfig):
        self.config = config

    def _get_seed(self, id: int):
        # return id
        return hash(str(datetime.now()) + str(id)) % (2 ** 32 - 1)

    def _play(self, id: int, iterations_per_move: int, initial_pos: Position, train_filename, colosus, stats):
        seed = self._get_seed(id)
        # print("worker: {}, seed: {}".format(id, seed))
        np.random.seed(seed)

        # train_record_set = TrainRecordSet()
        train_record_set_z = TrainRecordSet()
        searcher = SearcherMb(self.config.search_config, colosus)
        while not stats.should_continue():
            state = StateMb(initial_pos, None, None, self.config.state_config)
            end = False
            # game_records = []
            game_records_z = []
            while not end:
                policy, temp_policy, value, move, new_state = searcher.search(state, iterations_per_move)
                if new_state is None:
                    new_state = StateMb(state.position().move(move), None, None, self.config.state_config)
                # train_record = TrainRecord(state.position().to_model_position(), policy, value)
                train_record_z = TrainRecord(state.position().to_model_position(), policy, value)
                # print(f"child.N: {state.children()[move].N}, N: {state.N}")
                # game_records.append(train_record)
                game_records_z.append(train_record_z)
                new_state.parent = None
                end = new_state.position().is_end
                state = new_state
                mc = state.position().move_count
                # state.position().print()

                # print("temp policy")
                # State.print_policy(temp_policy, 10)
                # print("policy")
                # State.print_policy(policy, 10)
                # print("mc: {}".format(state.position().move_count))

            state.position().print()

            z = - state.position().score
            for j in reversed(range(len(game_records_z))):
                game_records_z[j].value = (1 - self.config.z_factor) * game_records_z[j].value + self.config.z_factor * z
                z = -z * state.config.backup_factor

            # train_record_set.extend(game_records)
            train_record_set_z.extend(game_records_z)

            win = state.position().score != 0
            stats.update(win, mc)

        colosus.close()
        # train_record_set.save_to_file(train_filename)
        # train_filename_z = "z" + train_filename
        train_record_set_z.save_to_file(train_filename)

        # for i in range(len(game_records)):
        #     value = game_records[i].value
        #     value_z = game_records_z[i].value
        #     print(f"value / value_z: {value} / {value_z}")

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
            position_starts = []
            position_lengths = []
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
                                position_indexes.append(i)
                                position_starts.append(len(positions))
                                position_lengths.append(len(pos))
                                positions.extend(pos)
                    except EOFError:
                        print("se desconecto")
                        conns[i] = None

            if len(positions) > 0:
                result = colosus.predict_on_batch(positions)
                policies, values = result

                for i in range(len(position_indexes)):
                    start = position_starts[i]
                    end = position_lengths[i] + start
                    pols = policies[start:end]
                    vals = values[start:end]
                    c = position_indexes[i]
                    conn = conns[c]
                    conn.send((pols, vals))

            alive = sum(1 for c in conns if c is not None)

        print("fin play_mp")