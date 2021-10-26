import sys
sys.path.append("..")
import os
import random
import math
import heapq
import time
import numpy as np
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from sklearn.linear_model import LinearRegression
from util import load_dataset, get_configs, parse_args, \
    get_pareto_points, recover_data, write_txt
from sample import random_sample
from space import parse_design_space
from vis import plot_pareto_set

seed = 2021
seed = int(time.time())
random.seed(seed)
np.random.seed(seed)

def f(u, v):
    return pow(
        math.e,
        -np.linalg.norm(
            np.array(
                u,
                dtype=np.float64
            ) -
            np.array(
                v,
                dtype=np.float64
            )
        ) ** 2 / (2 * sig ** 2)
    )

def F_same(K):
    n = len(K)
    F = []
    for i in range(n):
        t = []
        for j in range(n):
            t.append(f(K[i],K[j]))
        F.append(t)
    return np.array(F)

def update_F(F, K):
    n = F.shape[0]
    for i in range(len(K)):
        denom = f(K[i], K[i]) + mu
        for j in range(n):
            for k in range(n):
                F[j][k] -= (F[j][i] * F[k][i]) / denom

def select_mi(K, F):
    return K[np.argmax([np.linalg.norm(F[i]) ** 2 / (f(K[i], K[i]) + mu) for i in range(len(K))])]

def RandomizedTED(K, num_points):
    K = list(K)
    K_ = []
    for i in range(num_points):
        M_ = random.sample(K, Nrted)
        M_ = M_ + K_
        M_ = [tuple(M_[j]) for j in range(len(M_))]
        M_ = list(set(M_))
        F = F_same(M_)
        update_F(F, M_)
        K_.append(select_mi(M_, F))
    return K_

def random_walk(point, x, y):
    idx = []
    def _in(point, x, y):
        for i in range(len(x)):
            # if (np.abs(point - x[i]) < 1e-4).all():
            if ((point - x[i]) < 1e-4).all() or ((x[i] - point) < 1e-4).all():
                idx.append(i)
                return True
        return False
    point = design_space.random_walk(point)
    while not _in(point, x, y):
        point = design_space.random_walk(point)
    return point, np.delete(x, idx, axis=0), np.delete(y, idx, axis=0)

class DesignDataSet(object):
    """ DesignDataSet """
    def __init__(self, configs):
        super(DesignDataSet, self).__init__()
        self.configs = configs
        self.total_x, self.total_y = load_dataset(configs["dataset-output-path"])
        self.x, self.y = self.total_x.copy(), self.total_y.copy()

    def _get_idx(self, x):
        return np.array([(x == i).all() for i in self.x])

    def evaluate_true(self, x):
        """
            x: <numpy.ndarray>
        """
        idx = np.array([False for i in self.x])
        for _x in x:
            idx = idx + self._get_idx(_x)
        return self.y[idx]

    def remove_sampled_data(self, x):
        idx = np.array([False for i in self.x])
        for _x in x:
            idx = idx + self._get_idx(_x)
        self.x = np.delete(self.x, idx, axis=0)
        self.y = np.delete(self.y, idx, axis=0)

        
def build_model(x, y):
    return LinearRegression().fit(x, y)

def sa_search(model, dataset, logger=None, top_k=5, n_iter=500,
    early_stop=100, parallel_size=128, log_interval=50):
    """
        model: <sklearn.model>
        dataset: <tuple>
        return:
        heap_items: <list> (<tuple> in <list>), specifically,
        <tuple> is (<int>, <list>) or (hv, configurations)
    """
    x, y = dataset
    n_dim = x.shape[-1]
    (x, y), (points, _y) = random_sample(configs, x, y, batch=parallel_size)
    scores = model.predict(points)
    # the larger `scores` is, the better the point
    # (i.e., the area is larger, however, c.c. and power are inversed)
    scores = np.prod(scores, axis=1)

    # build heap
    heap_items = [(float('-inf'), - 1 - i) for i in range(top_k)]
    heapq.heapify(heap_items)

    for p, s in zip(points, scores):
        if s > heap_items[0][0]:
            pop = heapq.heapreplace(heap_items, (s, design_space.knob2point(p)))

    temp = (1, 0)
    cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)
    t = temp[0]
    k_last_modify = 0
    k = 0
    while k < n_iter and k < k_last_modify + early_stop:
        new_points = np.empty_like(points)
        for i, p in enumerate(points):
            new_points[i], x, y = random_walk(p, x, y)
        new_scores = model.predict(new_points)
        new_scores = np.prod(new_scores, axis=1)
        ac_prob = np.exp(np.minimum((new_scores - scores) / (t + 1e-5), 1))
        ac_index = np.random.random(len(ac_prob)) < ac_prob
        points[ac_index] = new_points[ac_index]
        scores[ac_index] = new_scores[ac_index]

        for p, s in zip(new_points, new_scores):
            if s > heap_items[0][0]:
                pop = heapq.heapreplace(heap_items, (s, design_space.knob2point(p)))
                k_last_modify = k

        k += 1
        t -= cool

        if log_interval and k % log_interval == 0:
            t_str = "%.8f" % t
            msg = "SA iter: %d\tlast update: %d\tmax-0: %.8f\ttemp: %s\t" % (k,
                k_last_modify, heap_items[0][0], t_str)
            if logger:
                logger.info("[INFO]: %s" % msg)
            else:
                print("[INFO]: %s" % msg)

    # big -> small
    heap_items.sort(key=lambda item: -item[0])
    return heap_items

def _transfer_xlayout(x):
    l = len(x)
    for i in range(l):
        x[i] = np.array(x[i])
    return x

def main():
    dataset = DesignDataSet(configs)

    _x = RandomizedTED(dataset.x, configs["initialize"])
    # transfer x layout
    _x = _transfer_xlayout(_x)
    _y = dataset.evaluate_true(_x)
    dataset.remove_sampled_data(_x)

    # build model
    model = build_model(_x, _y)

    # sa search
    heap = sa_search(
        model,
        (dataset.x, dataset.y),
        top_k=30,
        n_iter=10,
        early_stop=35,
        parallel_size=3,
        log_interval=10
    )
    pred = []
    for i, point in heap:
        pred.append(design_space.point2knob(point))
    pred = np.array(pred)

    # add `_x` into `pred`
    for i in _x:
        pred = np.insert(pred, len(pred), i, axis=0)
    # get corresponding `_y`
    idx = []
    for _pred in pred:
        for i in range(len(dataset.total_x)):
            if (np.abs(_pred - dataset.total_x[i]) < 1e-4).all():
                idx.append(i)
                break
    pareto_set = get_pareto_points(dataset.total_y[idx])
    plot_pareto_set(
        recover_data(pareto_set),
        dataset_path=configs["dataset-output-path"],
        output=os.path.join(
            configs["fig-output-path"]
        )
    )

    # write results
    # pareto set
    write_txt(
        os.path.join(
            configs["rpt-output-path"]
        ),
        np.array(pareto_set),
        fmt="%f"
    )
    # model
    joblib.dump(
        model,
        os.path.join(
            configs["model-output-path"]
        )
    )

if __name__ == '__main__':
    configs = get_configs(parse_args().configs)
    design_space = parse_design_space(configs["design-space"])
    sig = configs["sig"]
    mu = configs["mu"]
    Nrted = configs["Nrted"]
    main()
