import multiprocessing as mp
from typing import List

import time


class Colosus:
    def predict(self, input: List[str]):
        output = []
        for i in input:
            output.append("prediction of " + i)
        time.sleep(0.1)
        return output

class ColosusProxy:
    def __init__(self, conn: mp.Connection):
        self.conn = conn

    def predict(self, input: str):

