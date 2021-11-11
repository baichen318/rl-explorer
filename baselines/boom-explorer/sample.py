# Author: baichen318@gmail.com

import random
import math
import torch
from time import time
import numpy as np
from space import parse_design_space
from util import parse_args, get_configs, create_logger, write_excel, write_txt
from exception import UnDefinedException

# seed = 2021
seed = int(time())
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

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

class ClusteringRandomizedTED(RandomizedTED):
    """
        `design_space`: <DesignSpace>
    """
    def __init__(self, configs):
        super(ClusteringRandomizedTED, self).__init__(configs)
        self.configs = configs
        self.batch_per_cluster = self.configs["batch"] // self.configs["cluster"]
        # feature dimension
        assert self.configs["design"] == "boom", \
            "[ERROR]: %s is not supported." % self.configs["design"]
        self.n_dim = 10

    def distance(self, x, y, l=2):
        """calculates distance between two points"""
        # NOTICE: `decodeWidth` should be assignd with larger weights
        weights = [1 for i in range(self.n_dim)]
        weights[4] *= 5

        return np.sum((x - y)**l * weights).astype(float)

    def clustering(self, points, k, max_iter=100):
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
                distances = [self.distance(points[p], centroids[c]) for c in range(len(centroids))]
                new_assignment[p] = np.argmin(distances)

            for c in range(len(centroids)):
                members = [points[p] for p in range(len(points)) if new_assignment[p] == c]
                if members:
                    centroids[c] = np.mean(members, axis=0).astype(int)
                else:
                    centroids[c] = points[np.random.choice(len(points))]
                    split = True

        loss = np.sum([self.distance(points[p], centroids[new_assignment[p]]) \
            for p in range(len(points))])

        return centroids, new_assignment, loss

    def gather_groups(self, dataset, cluster):
        new_dataset = [[] for i in range(self.configs["cluster"])]

        for i in range(len(dataset)):
            new_dataset[cluster[i]].append(dataset[i])
        for i in range(len(new_dataset)):
            new_dataset[i] = np.array(new_dataset[i])
        if self.configs["vis-crted"]:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE

            fig = plt.figure()
            ax = fig.add_subplot(111)
            markers = [
                '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
                '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
                'X', 'D', 'd', '|', '_'
            ]
            colors = [
                'c', 'b', 'g', 'r', 'm', 'y', 'k', 'w'
            ]
            tsne = TSNE(n_components=2)
            i = 0
            h = []
            anno = ["cluster %d" % d for d in range(1, len(new_dataset) + 1)]
            for c in new_dataset:
                tsne.fit_transform(c)
                for _c in tsne.embedding_:
                    h.append(
                        ax.scatter(
                            _c[-2],
                            _c[-1],
                            s=4,
                            marker=markers[2],
                            c=colors[i],
                            label=anno
                        )
                    )
                i += 1
            # plt.legend(tuple(h), labels=anno, loc=0, frameon=False)
            plt.xlabel("c.c.")
            plt.ylabel("Power")
            plt.title("Clusters on design space")
            plt.grid()
            plt.show()
        return new_dataset

    def crted(self, dataset):
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

        centroids, new_assignment, loss = self.clustering(
            dataset,
            self.configs["cluster"],
            max_iter=500
        )
        new_dataset = self.gather_groups(dataset, new_assignment)

        data = []
        for c in new_dataset:
            x = []
            while len(x) < min(self.batch_per_cluster, len(c)):
                if len(c) > (self.batch_per_cluster - len(x)) and \
                    len(c) > self.configs["Nrted"]:
                    candidates = self.rted(
                        c,
                        self.batch_per_cluster - len(x)
                    )
                else:
                    candidates = c
                for _c in candidates:
                    x.append(_c)
                x = _delete_duplicate(x)
            for _x in x:
                data.append(_x)

        return data

class Sampler(object):
    """
        Sampler: sample configurations
    """
    def __init__(self, configs):
        super(Sampler, self).__init__()
        self.configs = configs

    def sample(self):
        """
            sample points
        """
        raise NotImplementedError

class RandomSampler(Sampler):
    """
        RandomSampler: randomly sample configurations
    """
    def __init__(self, configs):
        super(RandomSampler, self).__init__(configs)
        random.seed(2021)
    def set_random_state(self, random_state):
        random.seed(random_state)

    def sample(self, x, y, batch=5):
        """
            dataset: <numpy.ndarray> (M x N)
        """
        idx = random.sample(range(len(x)), min(batch, len(x)))
        _x, _y = [], []
        for i in idx:
            _x.append(x[i])
            _y.append(y[i])
        x = np.delete(x, idx, axis=0)
        y = np.delete(y, idx, axis=0)
        return (x, y), (np.array(_x), np.array(_y))

    def sample_v2(self, problem, batch=1):
        idx = random.sample(range(0, problem.n_sample), batch)
        x = problem.x[idx]
        y = problem.evaluate_true(x)
        return x.to(torch.float32), y

def crted_sample(configs, problem, mode):
    """
        configs: <dict>
        problem: <MultiObjectiveTestProblem>
        mode: <str>
    """
    if mode == "online":
        sampler = ClusteringRandomizedTED(configs)
        x = torch.Tensor(
            sampler.crted(
                problem.design_space.sample_v1(configs["batch"]).numpy()
            )
        )
        y = torch.Tensor([])
        for _x in x:
            _y = problem.evaluate_microarchitecture(_x.numpy().squeeze(0))
            y = torch.cat((y, _y), 0)
    else:
        assert mode == "offline"
        sampler = ClusteringRandomizedTED(configs)
        x = torch.Tensor(sampler.crted(problem.x.numpy()))
        y = problem.evaluate_true(x)
        problem.remove_sampled_data(x)
    return x, y

def random_sample(configs, x, y, batch=1):
    sampler = RandomSampler(configs)
    return sampler.sample(x, y, batch)

def random_sample_v2(configs, problem, batch):
    """
        configs: <dict>
        problem: <MultiObjectiveTestProblem>
    """
    sampler = RandomSampler(configs)
    x, y = sampler.sample_v2(problem, batch)
    problem.remove_sampled_data(x)
    return x, y
