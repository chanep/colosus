import numpy as np
import multiprocessing as mp

import time

from colosus.config import SearchConfig
from .state import State


class ColosusProxy:
    def __init__(self, conn):
        self.conn = conn

    def predict(self, position):
        self.conn.send(position)
        result = self.conn.recv()
        return result

    def send_result(self, result):
        self.conn.send(result)
        self.conn.recv()
        self.conn.close()


class SearcherMp:
    def __init__(self, config: SearchConfig):
        self.config = config

    def _search(self, id: int, root_state: State, iterations: int, colosus):
        self._set_colosus(root_state, colosus)
        root_state.config.cpuct = root_state.config.cpuct * (self.config.mp_cpuct0 + self.config.mp_cpuct_factor * id)
        for i in range(iterations):
            root_state.select()
        # policy, value, move, new_state = root_state.play(self._get_temperature(root_state))
        self._set_colosus(root_state, None)
        colosus.send_result(root_state)

    def search(self, root_state: State, iterations: int) -> (np.array, float, int, State):
        if iterations == 1:
            return root_state.play_static_policy(self._get_temperature(root_state))

        colosus = root_state.colosus
        self._set_colosus(root_state, None)

        workers = self.config.workers

        processes = []
        conns = []
        results = [None] * workers

        for worker_id in range(workers):
            server_conn, client_conn = mp.Pipe()

            colosus_proxy = ColosusProxy(client_conn)
            args = (worker_id, root_state, iterations, colosus_proxy)
            p = mp.Process(target=self._search, args=args)
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
                            if isinstance(pos, State):
                                # print("worker {} finish".format(i))
                                c.send("fin")
                                c.close()
                                conns[i] = None
                                results[i] = pos
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

        policy, value, move, new_state = self._consolidate_results(results)
        self._set_colosus(root_state, colosus)
        self._set_colosus(new_state, colosus)
        return policy, value, move, new_state

    def _consolidate_results(self, results):
        main_worker_root_state = results[self.config.mp_main_worker_id]
        children_N = [0] * len(main_worker_root_state.children())
        children_W = [0] * len(main_worker_root_state.children())
        for i in range(len(results)):
            root_state: State = results[i]
            for c in range(len(root_state.children())):
                child_state: State = root_state.children()[c]
                if child_state is not None:
                    children_N[c] += child_state.N
                    children_W[c] += child_state.W

        policy = np.array(children_N) / sum(children_N)
        temperature = self._get_temperature(main_worker_root_state)
        temp_policy = main_worker_root_state.apply_temperature(policy, temperature)
        move = np.random.choice(len(temp_policy), 1, p=temp_policy)[0]
        value = children_W[move] / children_N[move]
        new_state = main_worker_root_state.children()[move]
        new_state.parent = None

        return policy, value, int(move), new_state

    def _set_colosus(self, state: State, colosus):
        state.colosus = colosus
        if state.children() is not None:
            for child in state.children():
                if child is not None:
                    self._set_colosus(child, colosus)

    def _get_temperature(self, state: State):
        if state.position().move_count <= self.config.move_count_temp0:
            return 1.0 * self.config.mp_temp_factor
        else:
            return 0.1

