# Author: baichen318@gmail.com


import sys, os
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir
    )
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir,
        "utils"
    )
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir,
        "simulation"
    )
)
os.environ["MKL_THREADING_LAYER"] = "GNU"
import random
import itertools
import numpy as np
from time import time
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from dse.env.boom.design_space import parse_design_space
from simulation.boom.simulation import Gem5Wrapper
from utils import parse_args, get_configs, info, load_txt


# Boosting algorithm which uses another metric for success.
# Algorithm from Wu et al (2008)
# Freund (2003)
# https://fr.wikipedia.org/wiki/RankBoost

# NOTE: I convert y from {0,1} to {-1,+1} and then back again because
# it makes it easier for the learning method :P

# refer it to:
# 	https://github.com/rpmcruz/machine-learning/blob/master/ensemble/boosting/rankboost.py
class RankBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, T, base_estimator=None):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=3)
        self.estimator = base_estimator
        self.T = T
        self.classes_ = (-1, 1)

    def fit(self, X, y):
        n0 = np.sum(y == -1)
        n1 = np.sum(y == 1)
        D = np.repeat(1./(n0 * n1), n0 * n1)

        ys = np.r_[np.zeros(n0 * n1), np.ones(n0 * n1)]

        nn = np.sum(y == 1) * np.sum(y == -1)
        len_diff = nn * 2
        Xs = np.zeros((len_diff, X.shape[1]))

        for i, x0 in enumerate(X[y == -1]):
            for j in range(n1):
                Xs[i * n1 + j] = x0

        for i, x1 in enumerate(X[y ==1 ]):
            for j in range(n0):
                Xs[nn + i + n1 * j] = x1

        self.h = [None] * self.T
        self.a = [0] * self.T
        # to avoid division by zero (Schapire and Singer, 1999)
        epsilon = 1e-6

        for t in range(self.T):
            # Train weak ranker ft based on distribution Dt
            Ds = np.r_[D, D] / 2
            self.h[t] = clone(self.estimator).fit(Xs, ys, Ds)

            # Choose alpha
            # there are apparently several approaches for this; we are using
            # one for a classification base estimator
            f_X0 = self.h[t].predict(X[y == -1])
            f_X1 = self.h[t].predict(X[y == 1])

            df = np.repeat(f_X0, n1) - np.tile(f_X1, n0)

            # right
            Wneg = np.sum(D[df == -1])
            # wrong
            Wpos = np.sum(D[df == +1])

            self.a[t] = 0.5 * np.log((Wneg + epsilon) / (Wpos + epsilon))

            # Update D
            D = D * np.exp(self.a[t] * df)
            # normalize distribution
            D = D / np.sum(D)
        return self

    def predict(self, X):
        return np.sum([a * h.predict(X) for a, h in zip(self.a, self.h)], 0)



# refer it to:
#   https://github.com/ashishpatel26/sklearn-ranking/blob/master/sklearn-ranking/utils/transform_pairwise.py
def transform_pairwise(X, y):
    """
        Transforms data into pairs with balanced labels for ranking
        Transforms a n-class ranking problem into a two-class classification
        problem. Subclasses implementing particular strategies for choosing
        pairs should override this method.
        In this method, all pairs are choosen, except for those that have the
        same target value. The output is an array of balanced classes, i.e.
        there are the same number of -1 as +1
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data
        y : array, shape (n_samples,) or (n_samples, 2)
            Target labels. If it's a 2D array, the second column represents
            the grouping of samples, i.e., samples with different groups will
            not be considered.
        Returns
        -------
        X_trans : array, shape (k, n_feaures)
            Data as pairs
        y_trans : array, shape (k,)
            Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


def create_mlp(hidden):
    return MLPRegressor(
        hidden_layer_sizes=(16, hidden),
        activation="logistic",
        solver="adam",
        alpha=0.0001,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=50000,
        momentum=0.5,
        early_stopping=True
    )


def load_dataset():
    dataset = load_txt(
        os.path.join(
            rl_explorer_root,
            configs["dataset"]
        ),
        fmt=float
    )
    # for the baseline implementation, we trunk the dataset
    return dataset[:, :-3]


def evaluate_microarchitecture(vec, idx=5):
    manager = Gem5Wrapper(
        configs,
        design_space,
        vec,
        idx
    )
    perf = manager.evaluate_perf()
    power, area = manager.evaluate_power_and_area()
    perf = perf_model.predict(np.expand_dims(
            np.concatenate((vec, [perf])),
            axis=0
        )
    )[0]
    power = power_model.predict(np.expand_dims(
            np.concatenate((vec, [power])),
            axis=0
        )
    )[0]
    area = area_model.predict(np.expand_dims(
            np.concatenate((vec, [area])),
            axis=0
        )
    )[0]
    return np.array([perf, power, area])


def construct_sorted_list(rank_list):
    """
        sort from small to large
        rank_list: <numpy.ndarray>, (n x n)
    """
    sorted_list = [i for i in range(rank_list.shape[0])]

    info("construct sorted list...")
    def swap(i, j):
        k = sorted_list[i]
        sorted_list[i] = sorted_list[j]
        sorted_list[j] = k

    for i in range(rank_list.shape[0]):
        for j in range(rank_list.shape[0]):
            if rank_list[i, j] > 300:
                swap(i, j)
    return sorted_list


def partition_sorted_list(design_pool, sorted_list, threshold, metric="power"):
    info("partition sorted list...")
    metric = 1 if metric == "power" else 2
    n_samples = design_pool.shape[0]
    k = 1 / 2
    pivot = sorted_list[round(k * n_samples)]
    p = evaluate_microarchitecture(design_pool[pivot])[metric]
    while p >= threshold:
        k /= 2
        if round(k * n_samples) == 0:
            return round(k * n_samples)
        pivot = sorted_list[round(k * n_samples)]
        p = evaluate_microarchitecture(design_pool[pivot])[metric]
    return round(k * n_samples)


def construct_rank_list(design_pool, ranker):
    comb = itertools.combinations(range(design_pool.shape[0]), 2)
    rank_list = np.zeros((design_pool.shape[0], design_pool.shape[0]))
    info("constructing rank list...")
    for k, (i, j) in enumerate(comb):
        rank_list[i, j] = ranker.predict(
            np.expand_dims(design_pool[i] - design_pool[j], axis=0)
        )
    return rank_list


def construct_ranker(dataset, metric):
    ranker = RankBoost(T=200)
    info("training {} model...".format(metric))
    if metric == "perf":
        metric = -3
    elif metric == "power":
        metric = -2
    else:
        assert metric == "area"
        metric = -1
    # align with the original paper
    X_new, y_new = transform_pairwise(dataset[:50, :-3], dataset[:50, metric])
    return ranker.fit(X_new, y_new)


def sample_from_design_space(k=200):
    index = random.sample(range(design_space.size), k=k)
    design_pool = []
    info("sampling {} designs...".format(k))
    for idx in index:
        design_pool.append(design_space.idx_to_vec(idx))
    design_pool = np.array(design_pool)
    return index, design_pool


def main():
    dataset = load_dataset()
    # we perform constrained DSE w.r.t. the paper
    start = time()
    index, design_pool = sample_from_design_space()
    # construct area model
    ranker = construct_ranker(dataset, "area")
    rank_list = construct_rank_list(design_pool, ranker)
    sorted_list = construct_sorted_list(rank_list)
    pivot = partition_sorted_list(design_pool, sorted_list, threshold_area, metric="area")
    # construct power model
    ranker = construct_ranker(dataset, "power")
    # clip the design pool
    index = sorted_list[:pivot + 1]
    design_pool = design_pool[np.array(index)]
    rank_list = construct_rank_list(design_pool, ranker)
    sorted_list = construct_sorted_list(rank_list)
    pivot = partition_sorted_list(design_pool, sorted_list, threshold_power, metric="power")
    # construct perf model
    ranker = construct_ranker(dataset, "perf")
    # clip the design pool
    index = sorted_list[:pivot + 1]
    design_pool = design_pool[np.array(index)]
    rank_list = construct_rank_list(design_pool, ranker)
    sorted_list = construct_sorted_list(rank_list)
    end = time()
    with open(
        os.path.join(
            rl_explorer_root,
            "baselines",
            "isca14",
            "{}-solution.txt".format(configs["design"]).replace(' ', '-')
        ),
        'w'
    ) as f:
        solutions = design_pool[np.array(sorted_list)]
        f.write("obtained solution: {}.\n".format(solutions.shape[0]))
        for solution in solutions:
            f.write("{} \n".format(solution))
        f.write("cost time: {} s.\n".format(end - start))
    info("ISCA14 done.")


if __name__ == "__main__":
    args = parse_args()
    configs = get_configs(args.configs)
    design_space = parse_design_space(configs)
    configs["configs"] = args.configs
    configs["logger"] = None
    rl_explorer_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir
        )
    )
    ppa_model_root = os.path.join(
        rl_explorer_root,
        configs["ppa-model"]
    )
    perf_root = os.path.join(
        ppa_model_root,
        "boom-perf.pt"
    )
    power_root = os.path.join(
        ppa_model_root,
        "boom-power.pt"
    )
    area_root = os.path.join(
        ppa_model_root,
        "boom-area.pt"
    )
    perf_model = joblib.load(perf_root)
    power_model = joblib.load(power_root)
    area_model = joblib.load(area_root)

    # set constraint DSE
    ppa = {
        # ipc power area
        # Small SonicBOOM
        "1-wide 4-fetch SonicBOOM": [0.766128848, 0.0212, 1504764.403],
        "1-wide 8-fetch SonicBOOM": [0.766128848, 0.0212, 1504764.403],
        # Medium SonicBOOM
        "2-wide 4-fetch SonicBOOM": [1.100314122, 0.0267, 1933210.356],
        "2-wide 8-fetch SonicBOOM": [1.100314122, 0.0267, 1933210.356],
        # Large SonicBOOM
        "3-wide 4-fetch SonicBOOM": [1.312793895, 0.0457, 3205484.562],
        "3-wide 8-fetch SonicBOOM": [1.312793895, 0.0457, 3205484.562],
        # Mega SonicBOOM
        "4-wide 4-fetch SonicBOOM": [1.634452069, 0.0592, 4805888.807],
        "4-wide 8-fetch SonicBOOM": [1.634452069, 0.0592, 4805888.807],
        # Giga SonicBOOM
        "5-wide SonicBOOM": [1.644617524, 0.0715, 5069115.916]
    }
    threshold_power = ppa[configs["design"]][1]
    threshold_area = ppa[configs["design"]][2]
    main()
