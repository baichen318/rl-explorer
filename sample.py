# Author: baichen318@gmail.com

import random
import math
from time import time
import numpy as np
from space import parse_design_space
from util import parse_args, get_configs, create_logger, write_excel, write_txt
from exception import UnDefinedException

class RandomizedTED(object):
    """
        `Nrted`: <int>
        `mu`: <float>
    """

    def __init__(self, kwargs):
        super(RandomizedTED, self).__init__()
        self.Nrted = kwargs["Nrted"]
        self.mu = kwargs["mu"]
        self.sig = kwargs["sig"]

    def f(self, u, v):
        # t = np.linalg.norm(np.array(u,dtype=np.float64)-np.array(v,dtype=np.float64))**2
        # if t > 1:
        #     print('t is ',t)
        return pow(
            math.e,
            -np.linalg.norm(
                np.array(u, dtype=np.float64) - np.array(v, dtype=np.float64)
            )**2 / (2 * self.sig**2)
        )

    def f_same(self, K):
        n = len(K)
        F = []
        for i in range(n):
            t = []
            for j in range(n):
                t.append(self.f(K[i], K[j]))
            F.append(t)
        return np.array(F)

    def update_f(self, F, K):
        n = F.shape[0]
        for i in range(len(K)):
            denom = self.f(K[i], K[i]) + self.mu
            for j in range(n):
                for k in range(n):
                    F[j][k] -= (F[j][i] * F[k][i]) / denom

    def select_mi(self, K, F):
        return K[
            np.argmax(
                [np.linalg.norm(F[i]) ** 2 / (self.f(K[i], K[i]) + self.mu) \
                    for i in range(len(K))]
            )
        ]

    def rted(self, vec, m):
        """
            API for Randomized TED
            vec: <np.array>
            m: <int>: number of a batch
        """
        # NOTICE: `rted` may select duplicated points,
        # in order to avoid this problem, we delete 80%
        # some points randomly
        def _delete_duplicate(vec):
            """
                `vec`: <list>
            """
            return [list(v) for v in set([tuple(v) for v in vec])]

        K_ = []
        for i in range(m):
            M_ = random.sample(list(vec), self.Nrted)
            M_ = M_ + K_
            M_ = [tuple(M_[j]) for j in range(len(M_))]
            M_ = list(set(M_))
            F = self.f_same(M_)
            self.update_f(F, M_)
            K_.append(self.select_mi(M_, F))
        return _delete_duplicate(K_)

class ClusteringRandomizedTED(RandomizedTED):
    """
        `design_space`: <DesignSpace>
    """
    def __init__(self, kwargs, design_space):
        super(ClusteringRandomizedTED, self).__init__(kwargs)
        self.design_space = design_space
        self.Batch = kwargs["Batch"]
        self.batch = kwargs["batch"]
        assert self.Batch > self.batch, "[ERROR] require self.Batch > self.batch"

    def cbted(self):
        x = []
        decodeWidth = self.design_space.bounds["decodeWidth"]
        for i in decodeWidth:
            cnt = 0
            _x = []
            while cnt < self.batch:
                candidates = self.rted(
                    self.design_space.random_sample_v2(i, self.Batch),
                    self.batch - cnt
                )
                cnt += len(candidates)
                for c in candidates:
                    _x.append(c)
                self.design_space.set_random_state(round(time()))
            x.append(_x)

        return np.array(x).reshape((-1, self.design_space.n_dim))

def create_design_space():
    logger = create_logger("logs", configs["initialize-method"])
    design_space = parse_design_space(configs["design-space"])
    logger.info("Design space size: %d" % design_space.size)

    return design_space

def record_sample(design_space, data):
    print(data)
    write_excel(configs["initialize-output-path"] + ".xlsx", data, design_space.features)
    write_txt(configs["initialize-output-path"] + ".txt", data)

def initialize(method):
    design_space = create_design_space()

    if method == "random":
        data = design_space.random_sample(configs["initialize-size"])
    elif method == "crted":
        cbted = ClusteringRandomizedTED(configs, design_space)
        data = cbted.cbted()
    else:
        raise UnDefinedException(method + " method")

    record_sample(design_space, data)

if __name__ == "__main__":
    argv = parse_args()
    configs = get_configs(argv.configs)
    initialize(configs["initialize-method"])
