import multiprocessing as mp
from typing import List

import time


class Stats:
    games = 0
    lock = mp.Lock()

    def reset(self):
        Stats.games = 0

    def update(self):
        Stats.lock.acquire()
        Stats.games += 1
        self._print()
        Stats.lock.release()

    def _print(self):
        print("games: " + str(Stats.games))


class Colosus:
    def predict(self, inputs: List[str]):
        outputs = []
        for i in inputs:
            outputs.append("prediction of " + i)
        time.sleep(0.1)
        return outputs


class ColosusProxy:
    def __init__(self, conn):
        self.conn = conn

    def predict(self, input: str):
        self.conn.send(input)
        output = self.conn.recv()
        return output

    def close(self):
        self.conn.send("fin")
        self.conn.close()


class SelfPlayMp:
    def play(self, id: int, iterations: int, colosus, update):
        for i in range(iterations):
            time.sleep(0.05)
            result = colosus.predict("iteration {} de {}".format(i, id))
            print("result {} de {}: {}".format(i, id, result))
        print("fin de play de " + str(id))
        update()
        colosus.close()

    def play_mp(self):
        p_num = 4
        iterations = 4

        colosus = Colosus()
        stats = Stats()
        stats.reset()

        processes = []
        conns = []

        games = mp.Value('i', 0)
        games_ant = 0

        def update():
            with games.get_lock():
                games.value += 1
            print("update")

        for i in range(p_num):
            server_conn, client_conn = mp.Pipe()
            colosusProxy = ColosusProxy(client_conn)
            args = (i, iterations, colosusProxy, update)
            p = mp.Process(target=self.play, args=args)
            processes.append(p)
            conns.append(server_conn)
            p.start()

        alive = p_num
        while alive > 0:
            inputs = []
            input_indexes = []
            for i in range(p_num):
                c = conns[i]
                if c is not None:
                    try:
                        if c.poll():
                            input = c.recv()
                            if isinstance(input, str) and input == "fin":
                                print("se desconecto por input=fin")
                                conns[i] = None
                            else:
                                inputs.append(input)
                                input_indexes.append(i)
                    except EOFError:
                        print("se desconecto")
                        conns[i] = None

            results = colosus.predict(inputs)

            for i in range(len(results)):
                result = results[i]
                c = input_indexes[i]
                conn = conns[c]
                conn.send(result)

            if games_ant != games.value:
                games_ant = games.value
                print("games: " + str(games_ant))

            alive = sum(1 for c in conns if c is not None)
            print("alive " + str(alive))

        print("fin play_mp")





