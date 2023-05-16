# Author: baichen318@gmail.com

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import random
import numpy as np
from time import time
from datetime import datetime
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from simulation.boom.simulation import Gem5Wrapper as BOOMGem5Wrapper
from utils.utils import parse_args, get_configs, info, load_txt, Timer
from simulation.rocket.simulation import Gem5Wrapper as RocketGem5Wrapper
from dse.env.boom.design_space import parse_design_space as parse_boom_design_space
from dse.env.rocket.design_space import parse_design_space as parse_rocket_design_space


def generate_L(dataset):
    L = []
    for data in dataset:
        vec = data[:-3].astype(int)
        L.append(design_space.vec_to_idx(list(vec)))
    return L


def generate_P_from_U(k, L):
    P = []
    for i in range(k):
        idx = random.sample(range(design_space.size), k=1)[0]
        while idx in L:
            idx = random.sample(range(design_space.size), k=1)[0]
        P.append(idx)
    return P


def load_dataset():
    dataset = load_txt(
        os.path.join(
            rl_explorer_root,
            configs["env"]["calib"]["dataset"]
        ),
        fmt=float
    )
    return dataset


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


def train_adaboost_rt(dataset, hidden):
    sample = dataset.shape[0]
    model = []
    # max. iteration
    T = 10
    # weight distribution and error
    epsilon = np.zeros(T)
    beta = np.zeros(T)
    fit = np.zeros((T, sample, 3))
    w = np.ones((T, sample))
    w /= np.linalg.norm(w, ord=1, axis=1).reshape(T, 1)
    phi = 0.2

    for t in range(T):
        idx = np.random.choice(range(sample), replace=True, p=w[t])
        x = dataset[idx][:-3]
        y = dataset[idx][-3:]
        # weak learner
        model.append(create_mlp(hidden).fit(
                np.expand_dims(x, axis=0),
                np.expand_dims(y, axis=0)
            )
        )
        # getback to hypothesis
        y_fit = model[t].predict(dataset[:, :-3])
        fit[t] = y_fit
        # adjusted error for each instance (PPA)
        ARE = np.mean(
            abs((dataset[:, -3:] - y_fit) / dataset[:, -3:]),
            axis=1
        )
        # calculate error of hypothesis
        epsilon[t] = np.sum(w[t][ARE > phi])
        # calculate beta
        beta[t] = epsilon[t] ** 2
        # update weight vector
        w[t + 1][ARE <= phi] = w[t][ARE <= phi] * beta[t]
        w[t + 1][ARE > phi] = w[t + 1][ARE > phi]
        # normalize
        w[t + 1] = w[t + 1] / np.sum(w[t + 1])

    # final model
    ww = np.log(1 / beta[1: -1])
    ret = ww * fit[1:T] / np.sum(ww)


def pseudo_train_adaboost_rt(dataset, hidden):
    T = 10
    model = []
    for i in range(T):
        model.append(create_mlp(hidden).fit(dataset[:, :-3], dataset[:, -3:]))
    return model


def predict_adaboost_rt(adaboost_model, x):
    y = []
    for model in adaboost_model:
        y.append(model.predict(x))
    return y


def calc_cv(data1, data2):
    """
        data1: <numpy.ndarray> (M x 3) (given by the first RT)
        data2: <numpy.ndarray> (M x 3) (given by the second RT)
    """
    T = len(data1)
    cv = []
    K = 3
    l = data1[0].shape[0]
    avg = np.zeros((l, K))
    cv = np.zeros((l, K))
    for k in range(K):
        for i in range(l):
            val = 0
            for t in range(T):
                val += data1[t][i][k]
            for t in range(T):
                val += data2[t][i][k]
            avg[i][k] = val / (2 * T)

    for i in range(l):
        for k in range(K):
            diff = 0
            part_sum = 0
            for t in range(T):
                diff += (data1[t][i][k] - avg[i][k]) ** 2
                part_sum += data1[t][i][k]
            for t in range(T):
                diff += (data2[t][i][k] - avg[i][k]) ** 2
                part_sum += data2[t][i][k]
            sigma = np.sqrt(diff / (2 * T))
            mu = part_sum / (2 * T)
            cv[i][k] = sigma / mu
    cv = np.mean(cv, axis=1)
    cv_list = []
    for i, _cv in enumerate(cv):
        cv_list.append((i, _cv))
    # big -> small
    cv = sorted(cv_list, key=lambda t:t[1], reverse=True)
    return cv


def evaluate_microarchitecture(vec):
    manager = Simulator(
        configs,
        design_space,
        vec,
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


def main():
    dataset = load_dataset()
    L =  generate_L(dataset)

    # create a pool P by moving p unlabeled samples from U to P randomly
    P = generate_P_from_U(64, L)

    # use L to train Adaboost.RT H1 & H2 with M1 & M2 hidden neurons
    H1 = pseudo_train_adaboost_rt(dataset, 6)
    H2 = pseudo_train_adaboost_rt(dataset, 8)

    K = 50
    W = 16
    N = 4

    solution_set = []

    start = time()
    for i in range(K):
        # both H1 and H2 predict the P
        p = []
        for idx in P:
            p.append(design_space.idx_to_vec(idx))
        p = np.array(p)
        y1 = predict_adaboost_rt(H1, p)
        y2 = predict_adaboost_rt(H2, p)

        # calculate c.v of each unlabeled sample in P and sort them
        cv = calc_cv(y1, y2)

        # choose N from top W randomly
        idx = random.sample(range(W), N)
        temp = []
        for j in idx:
            temp.append(P[cv[j][0]])
        # simulate the N samples
        for n in temp:
            solution_set.append(n)
            vec = design_space.idx_to_vec(n)
            ppa = evaluate_microarchitecture(vec)
            # move the newly labeled samples from P to L
            P.remove(n)
            dataset = np.insert(
                dataset,
                dataset.shape[0],
                np.concatenate((vec, ppa)),
                axis=0
            )
            L.append(n)

        # rebuild H1, H2 by the new set L
        H1 = pseudo_train_adaboost_rt(dataset, 6)
        H2 = pseudo_train_adaboost_rt(dataset, 8)

        # replenish the P by choosing N examples from U at random
        _P = generate_P_from_U(N, L)
        for p in _P:
            P.append(p)
    end = time()

    with open(
        os.path.join(
            rl_explorer_root,
            "baselines",
            "dac16",
            "solution-{}.txt".format(datetime.now()).replace(' ', '-')
        ),
        'w'
    ) as f:
        f.write("obtained solution: {}.\n".format(len(solution_set)))
        for solution in solution_set:
            f.write("{}\n".format(solution))
        f.write("\n")
        f.write("cost time: {}s.".format(end - start))
    info("DAC16 done.")


if __name__ == "__main__":
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
    else:
        assert configs["algo"]["design"] == "Rocket", \
            "{} is not supported.".format(configs["algo"]["design"])
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
    perf_model = joblib.load(perf_root)
    power_model = joblib.load(power_root)
    area_model = joblib.load(area_root)
    with Timer("DAC16"):
        main()
