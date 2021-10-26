# Author: baichen318@gmail.com

import heapq
import time
import numpy as np
from util import get_configs, parse_args, hyper_volume, load_dataset, split_dataset, rmse, adrs, write_csv
from handle_data import reference
from space import parse_design_space

decodeWidth = 3

def perfcmp(point, score):
    """
        point: `np.array`
        score: `np.array`
    """
    # `point[1]` is decodeWidth
    # scale c.c. and power dissipation
    ref = [reference[int(point[1]) - 1][0] / 90000, reference[int(point[1]) - 1][1] * 10]
    # return hyper_volume(ref, score)
    return -adrs(ref, score)

def _exist_duplicate(s, heap):
    """
        s: `float`
        heap: `list`
    """
    for item in heap:
        if s == item[0]:
            return True

    return False

def sa_search(model, design_space, logger=None, top_k=5, n_iter=500,
    early_stop=100, parallel_size=128, log_interval=50):
    """
        model: <sklearn.model>
        design_space: <tuple>
        return:
        heap_items: <list> (<tuple> in <list>), specifically,
        <tuple> is (<int>, <list>) or (hv, configurations)
    """
    points = design_space.random_sample_v2(decodeWidth, parallel_size)
    _scores = model.predict(points)
    scores = np.empty(parallel_size)
    for i, (p, s) in enumerate(zip(points, _scores)):
        scores[i] = perfcmp(p, s)

    # build heap and insert initial points
    # `performance, knob`
    heap_items = [(-1, list(np.empty(design_space.n_dim))) for i in range(top_k)]
    heapq.heapify(heap_items)
    visited = set()

    for p, s in zip(points, scores):
        _p = design_space.knob2point(p)
        if s > heap_items[0][0] and _p not in visited:
            pop = heapq.heapreplace(heap_items, (s, p))
            visited.add(_p)

    temp = (1, 0)
    cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)
    t = temp[0]
    k_last_modify = 0
    k = 0
    while k < n_iter and k < k_last_modify + early_stop:
        new_points = np.empty_like(points)
        for i, p in enumerate(points):
            new_points[i] = design_space.random_walk_v2(p)
        for i in range(len(new_points)):
            new_points[i] = list(new_points[i])
        _new_scores = model.predict(new_points)
        new_scores = np.empty(parallel_size)
        for i, (p, s) in enumerate(zip(new_points, _new_scores)):
            new_scores[i] = perfcmp(p, s)
        ac_prob = np.exp(np.minimum((new_scores - scores) / (t + 1e-5), 1))
        ac_index = np.random.random(len(ac_prob)) < ac_prob
        points[ac_index] = new_points[ac_index]
        scores[ac_index] = new_scores[ac_index]

        for p, s in zip(new_points, new_scores):
            _p = design_space.knob2point(p)
            if s > heap_items[0][0] and _p not in visited:
                if _exist_duplicate(s, heap_items):
                    continue
                pop = heapq.heapreplace(heap_items, (s, p))
                visited.add(_p)
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

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    dataset = load_dataset(configs["dataset-output-path"])

    design_space = parse_design_space(configs["design-space"])

    markers = [
        '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
        '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
        'X', 'D', 'd', '|', '_'
    ]
    colors = [
        'c', 'b', 'g', 'r', 'm', 'y', 'k', # 'w'
    ]
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    import matplotlib.pyplot as plt
    model = MultiOutputRegressor(
        XGBRegressor(
            max_depth=30,
            gamma=0.0001,
            min_child_weight=1,
            subsample=1.0,
            eta=0.3,
            reg_lambda=1.00,
            alpha=0,
            objective='reg:squarederror',
            n_jobs=-1
        )
    )

    x, y = split_dataset(dataset)
    model.fit(x, y)
    _y = model.predict(x)
    print("RMSE c.c:", rmse(y[:, 0], _y[:, 0]))
    print("RMSE power:", rmse(y[:, 1], _y[:, 1]))

    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600

    for i in range(len(y)):
        plt.scatter(
            y[i][0],
            y[i][1],
            s=1,
            marker=markers[-6],
            c=colors[-1],
        )
    for i in range(len(y)):
        plt.scatter(
            _y[i][0],
            _y[i][1],
            s=1,
            marker=markers[3],
            c=colors[2],
        )
    plt.xlabel('C.C.')
    plt.ylabel('Power')
    plt.title('C.C. vs. Power - ' + "decodeWidth = %d" % decodeWidth)
    # plt.show()

    heap = sa_search(
        model,
        design_space,
        top_k=100,
        n_iter=10000,
        early_stop=500,
        parallel_size=128,
        log_interval=50
    )
    write_csv("rpts/%s.rpt" % decodeWidth, heap, mode='w')

    perf = []
    for (hv, p) in heap:
        perf.append(model.predict(p.reshape(1, -1)).ravel())

    perf = np.array(perf)

    plt.scatter(perf[:, 0], perf[:, 1], s=1, marker=markers[-2], c=colors[1])
    plt.show()
