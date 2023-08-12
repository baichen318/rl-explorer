# Author: baichen318@gmail.com


import random
import math
import torch
from time import time
import numpy as np
from util import tensor_to_array
from utils.utils import assert_error


seed = int(time())
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
sampler = None

class RandomizedTED(object):
    """
        `Nrted`: <int>
        `mu`: <float>
    """

    def __init__(self, configs):
        super(RandomizedTED, self).__init__()
        self.Nrted = configs["Nrted"]
        self.mu = configs["mu"]
        self.sig = configs["sig"]

    def f(self, u, v):
        # t = np.linalg.norm(np.array(u,dtype=np.float64)-np.array(v,dtype=np.float64))**2
        # if t > 1:
        #     print('t is ',t)
        # NOTICE: target value should be discards
        u = u[:-2]
        v = v[:-2]
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
        K_ = []
        for i in range(m):
            M_ = random.sample(list(vec), self.Nrted)
            M_ = M_ + K_
            M_ = [tuple(M_[j]) for j in range(len(M_))]
            M_ = list(set(M_))
            F = self.f_same(M_)
            self.update_f(F, M_)
            K_.append(self.select_mi(M_, F))
        return K_


class MicroAL(RandomizedTED):
    """
        `design_space`: <DesignSpace>
    """
    # dataset constructed after cluster w.r.t. DecodeWidth
    _cluster_dataset = None
   
    def __init__(self, configs, problem):
        super(MicroAL, self).__init__(configs)
        self.configs = configs
        self.num_per_cluster = self.configs["batch"] // self.configs["cluster"]
        assert self.num_per_cluster > 0, \
            assert_error("batch: {} should be larger than cluster: {}.".format(
                    self.configs["batch"], self.configs["cluster"]
                )
            )
        self.decoder_threshold = self.configs["decoder-threshold"]
        # feature dimension
        self.n_dim = problem.n_dim

    @property
    def cluster_dataset(self):
        return self._cluster_dataset

    @cluster_dataset.setter
    def cluster_dataset(self, dataset):
        self._cluster_dataset = dataset

    def set_weight(self, pre_v=None):
        # if `pre_v` is specified, then `weights` will be assigned accordingly
        if pre_v:
            assert isinstance(pre_v, list) and len(pre_v) == self.n_dim, \
                "[ERROR]: unsupported pre_v."
            weights = pre_v
        else:
            # NOTICE: `decodeWidth` should be assignd with larger weights
            weights = [1 for i in range(self.n_dim)]
            weights[4] *= self.decoder_threshold
        return weights

    def distance(self, x, y, l=2, pre_v=None):
        """calculates distance between two points"""
        weights = self.set_weight(pre_v=pre_v)
        return np.sum((x - y) ** l * weights).astype(float)

    def kmeans(self, points, k, max_iter=100, pre_v=None):
        """k-means clustering algorithm"""
        centroids = [points[i] for i in np.random.randint(len(points), size=k)]
        new_assignment = [0] * len(points)
        old_assignment = [-1] * len(points)

        i = 0
        split = False
        while i < max_iter or split == True and new_assignment != old_assignment:
            old_assignment = list(new_assignment)
            split = False
            i += 1

            for p in range(len(points)):
                distances = [self.distance(points[p], centroids[c], pre_v=pre_v) \
                    for c in range(len(centroids))]
                new_assignment[p] = np.argmin(distances)

            for c in range(len(centroids)):
                members = [points[p] for p in range(len(points)) if new_assignment[p] == c]
                if members:
                    centroids[c] = np.mean(members, axis=0).astype(int)
                else:
                    centroids[c] = points[np.random.choice(len(points))]
                    split = True

        loss = np.sum([self.distance(points[p], centroids[new_assignment[p]], pre_v=pre_v) \
            for p in range(len(points))])

        return centroids, new_assignment, loss

    def gather_groups(self, dataset, cluster):
        new_dataset = [[] for i in range(self.configs["cluster"])]

        for i in range(len(dataset)):
            new_dataset[cluster[i]].append(dataset[i])
        for i in range(len(new_dataset)):
            new_dataset[i] = np.array(new_dataset[i])
        return new_dataset

    def micro_al(self, dataset):
        """
            dataset: <numpy.array>: M x n_dim
        """
        # NOTICE: `rted` may select duplicated points,
        # in order to avoid this problem, we delete 80%
        # some points randomly
        def _delete_duplicate(vec):
            """
                `vec`: <list>
            """
            return [list(v) for v in set([tuple(v) for v in vec])]

        centroids, new_assignment, loss = self.kmeans(
            dataset,
            self.configs["cluster"],
            max_iter=self.configs["cluster-iteration"]
        )
        self.cluster_dataset = self.gather_groups(dataset, new_assignment)

        sampled_data = []
        for c in self.cluster_dataset:
            x = []
            while len(x) < min(self.num_per_cluster, len(c)):
                if len(c) > (self.num_per_cluster - len(x)) and \
                    len(c) > self.Nrted:
                    candidates = self.rted(
                        c,
                        self.num_per_cluster - len(x)
                    )
                else:
                    candidates = c
                for _c in candidates:
                    x.append(_c)
                x = _delete_duplicate(x)
            for _x in x:
                sampled_data.append(_x)
        return sampled_data


class Sampler(object):
    """
        Sampler: sample configurations
    """
    def __init__(self, configs, problem):
        super(Sampler, self).__init__()
        self.configs = configs
        self.problem = problem

    def sample(self):
        """
            sample points
        """
        raise NotImplementedError


class RandomSampler(Sampler):
    """
        RandomSampler: randomly sample configurations
    """
    def __init__(self, configs, problem):
        super(RandomSampler, self).__init__(configs, problem)
        self.visited = set()
        self.labeled = set()

    def set_random_state(self, random_state):
        random.seed(random_state)

    def sample(self, batch=1):
        index = []
        for i in range(batch):
            idx = random.sample(range(1, self.problem.design_space.size + 1), k=1)[0]
            while idx in self.visited:
                idx = random.sample(range(1, self.problem.design_space.size + 1), k=1)[0]
            self.visited.add(idx)
            index.append(idx)

        x = []
        for idx in index:
            x.append(self.problem.design_space.idx_to_embedding(idx))
        x = np.array(x)
        return x

    def sample_from_offline_dataset(self, batch=1):
        for i in range(self.problem.n_sample):
            self.labeled.add(
                self.problem.design_space.embedding_to_idx(
                    list(tensor_to_array(self.problem.total_x[i]).astype("int"))
                )
            )
        index = random.sample(list(self.labeled), k=batch)
        x = []
        for idx in index:
            x.append(self.problem.design_space.idx_to_embedding(idx))
        x = np.array(x)
        return x


def micro_al(configs, problem):
    """
        configs: <dict>
        problem: <MultiObjectiveTestProblem>
    """
    global sampler
    sampler = MicroAL(configs, problem)
    x = torch.Tensor(sampler.micro_al(problem.x.numpy()))
    y = problem.evaluate_true(x)
    problem.remove_sampled_data(x)
    return x, y


def initial_random_sample(configs, problem, batch):
    global sampler
    sampler = RandomSampler(configs, problem)
    x = torch.Tensor(sampler.sample_from_offline_dataset(batch))
    y = problem.evaluate_true(x)
    problem.remove_sampled_data(x)
    return x, y

def random_sample(configs, problem, batch):
    """
        configs: <dict>
        problem: <MultiObjectiveTestProblem>
    """
    global sampler
    sampler = RandomSampler(configs, problem)
    x = sampler.sample(batch)
    return x
