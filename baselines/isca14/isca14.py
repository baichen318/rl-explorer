# Author: baichen318@gmail.com


"""
    We do not find any open-source repo. for DAC16 baseline.
    So, we implement according to the original manuscript.
"""


import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import random
import itertools
import numpy as np
from time import time
from datetime import datetime
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from simulation.boom.simulation import Gem5Wrapper as BOOMGem5Wrapper
from simulation.rocket.simulation import Gem5Wrapper as RocketGem5Wrapper
from dse.env.boom.design_space import parse_design_space as parse_boom_design_space
from dse.env.rocket.design_space import parse_design_space as parse_rocket_design_space
from utils.utils import parse_args, get_configs, mkdir, info, load_txt, Timer, if_exist


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
    """
        Since the RankingBoost's runtime is extremely high,
        we need to use a small dataset.
    """
    design = configs["algo"]["design"]
    if "BOOM" in design:
        name = "boom.txt"
    else:
        name = "rocket.txt"
    _dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if_exist(_dataset, strict=True)
    dataset = load_txt(_dataset, fmt=float)
    return dataset


def evaluate_microarchitecture(embedding):
    manager = Simulator(
        configs,
        design_space,
        embedding,
        configs["env"]["sim"]["idx"]
    )
    perf, stats = manager.evaluate_perf()
    power, area = manager.evaluate_power_and_area()
    area *= 1e6
    stats_feature = []
    for k, v in stats.items():
        stats_feature.append(v)
    stats_feature = np.array(stats_feature)
    perf = perf_model.predict(np.expand_dims(
            np.concatenate((
                    np.array(embedding),
                    stats_feature,
                    [perf]
                )
            ),
            axis=0
            )
        )[0]
    power = power_model.predict(np.expand_dims(
            np.concatenate((
                    np.array(embedding),
                    stats_feature,
                    [power]
                )
            ),
            axis=0
            )
        )[0]
    area = area_model.predict(np.expand_dims(
            np.concatenate((
                    np.array(embedding),
                    stats_feature,
                    [area]
                )
            ),
            axis=0
            )
        )[0]
    area *= 1e-6
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
    X_new, y_new = transform_pairwise(dataset[:90, :-3], dataset[:90, metric])
    return ranker.fit(X_new, y_new)


def sample_from_design_space(k=1000):
    index = random.sample(range(1, design_space.size + 1), k=k)
    design_pool = []
    info("sampling {} designs...".format(k))
    for idx in index:
        design_pool.append(design_space.idx_to_embedding(idx))
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

    # create the result directory
    result_root = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "result"
    )
    mkdir(result_root)

    with open(os.path.join(
            result_root, "{}.txt".format(datetime.now()).replace(' ', '-')
        ), 'w'
    ) as f:
        solutions = design_pool[np.array(sorted_list)]
        f.write("obtained solution: {}:\n".format(solutions.shape[0]))
        for solution in solutions:
            ppa = evaluate_microarchitecture(list(solution))
            f.write("{}\t{}\n".format(list(solution), ppa.tolist()))
        f.write("\ncost time: {} s.\n".format(end - start))
    info("ISCA14 done.")


if __name__ == "__main__":
    # set random seed
    random.seed(42)
    np.random.seed(42)

    args = parse_args()
    configs = get_configs(args.configs)
    configs["configs"] = args.configs
    configs["logger"] = None
    Simulator = None
    rl_explorer_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir
        )
    )
    ppa_model_root = os.path.join(
        rl_explorer_root,
        configs["env"]["calib"]["ppa-model"]
    )
    if "BOOM" in configs["algo"]["design"]:
        design_space = parse_boom_design_space(configs)
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
        Simulator = BOOMGem5Wrapper
        # set constraint DSE
        ppa = {
            # ipc power area
            "small-SonicBOOM": [0.445552, 0.038699, 306589.000000 * 1e-6],
            "medium-SonicBOOM": [0.577003, 0.051086, 310924.000000 * 1e-6],
            "large-SonicBOOM": [0.671735, 0.090563, 346155.000000 * 1e-6],
            "mega-SonicBOOM": [0.683200, 0.120811, 372036.000000 * 1e-6],
            "giga-SonicBOOM": [0.703640, 0.133239, 381220.000000 * 1e-6],
        }
        threshold_power = ppa["large-SonicBOOM"][1]
        threshold_area = ppa["large-SonicBOOM"][2]
    else:
        assert configs["algo"]["design"] == "Rocket", \
            "{} is not supported.".format(configs["design"])
        design_space = parse_rocket_design_space(configs)
        perf_root = os.path.join(
            ppa_model_root,
            "rocket-perf.pt"
        )
        power_root = os.path.join(
            ppa_model_root,
            "rocket-power.pt"
        )
        area_root = os.path.join(
            ppa_model_root,
            "rocket-area.pt"
        )
        Simulator = RocketGem5Wrapper
        # set constraint DSE
        ppa = {
            # ipc power area
            # Small SonicBOOM
            "Rocket": [0.5863153599, 0.0061203433600000015, 544133.31653 * 1e-6],
        }
        threshold_power = ppa["Rocket"][1]
        threshold_area = ppa["Rocket"][2]

    perf_model = joblib.load(perf_root)
    power_model = joblib.load(power_root)
    area_model = joblib.load(area_root)

    with Timer("ISCA14"):
        main()
