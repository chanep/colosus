import unittest
import cProfile, pstats, io
from functools import partial

from joblib import Parallel, delayed
from multiprocessing import Pool

import time


class Child:
    def __init__(self):
        self.N = 50
        self.N_in = 2
        self.P = 0.5


def get_child_score(factor, child: Child):
    return factor * child.P / (1 + child.N + child.N_in)


class ParallelTestCase(unittest.TestCase):
    def test_delayed(self):
        factor = 2.0
        children = [Child() for i in range(50)]

        start = time.time()

        parallel = Parallel(n_jobs=4, backend="threading")
        scores = [0] * 50
        for i in range(100):
            scores = parallel(delayed(get_child_score)(factor, child) for child in children)

        print("time: " + str(time.time() - start))
        print(scores)

    def test_pool(self):
        factor = 2.0
        children = [Child() for i in range(50)]

        start = time.time()

        p = Pool(4)
        fun = partial(get_child_score, factor)
        for i in range(50000):
            scores = p.map(fun, children)

        print("time: " + str(time.time() - start))
        print(scores)

    def test_forloop(self):
        factor = 2.0
        children = [Child() for i in range(50)]

        start = time.time()

        scores = [0] * 50
        for i in range(50000):
            for c in range(len(children)):
                child = children[c]
                scores[c] = get_child_score(factor, child)

        print("time: " + str(time.time() - start))
        print(scores)







if __name__ == '__main__':
    unittest.main()
