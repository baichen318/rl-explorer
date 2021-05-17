import sys
sys.path.append("..")
import os
import heapq
import random
import time
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from util import parse_args, get_configs, load_dataset, \
    split_dataset, rmse, strflush, write_csv
from bayesian_opt import calc_hv
from vis import plot_predictions_with_gt
from space import parse_design_space
from handle_data import reference

seed = 2021
random.seed(seed)
np.random.seed(seed)

def create_model(hidden=4):
	mlp = MLPRegressor(
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
	rt = MultiOutputRegressor(
		AdaBoostRegressor(
			n_estimators=50,
			learning_rate=1,
			loss='square'
		)
	)

	return {
		"mlp": mlp,
		"rt": rt
	}

def train_model(H, x, y):
	H["mlp"].fit(x, y)
	H["rt"].fit(H["mlp"].predict(x), y)

def predict_model(H, x):
    return H["rt"].predict(H["mlp"].predict(x))

def init_actboost(L):
	x, y = split_dataset(L)
	H1 = create_model(hidden=6)
	H2 = create_model(hidden=8)

	train_model(H1, x, y)
	train_model(H2, x, y)
	
	return H1, H2

def calc_cv(data1, data2):
	"""
		data1: <numpy.ndarray> (M x 2) (given by the first RT)
		data2: <numpy.ndarray> (M x 2) (given by the second RT)
	"""
	def _calc_cv(data):
		mu = np.mean(data)
		sigma = np.sqrt(np.sum((data - mu) ** 2) / len(data))

		return sigma / mu

	cv = []
	l = len(data1)
	for i in range(l):
		cv1 = _calc_cv(np.array([data1[i][0], data2[i][0]]))
		cv2 = _calc_cv(np.array([data1[i][1], data2[i][1]]))
		cv.append((i, 0.5 * cv1 + 0.5 * cv2))
    # big -> small
	cv = sorted(cv, key=lambda t:t[1], reverse=True)

	return cv

def get_dataset():
	dataset, title = read_csv_v2(configs["dataset-output-path"])
	dataset = validate(dataset)

	# scale dataset to balance c.c. and power dissipation
	dataset[:, -2] = dataset[:, -2] / 10000
	dataset[:, -1] = dataset[:, -1] * 100

	return dataset

def create_pool(dataset, p=64):
	idx = np.random.randint(0, len(dataset), p)
	L = dataset.copy()
	_idx, P = [], []

	for d in L[idx]:
		P.append(d)
		i = 0
		for _d in L:
			if ((_d - d) < 1e-5).all():
				_idx.append(i)
			i += 1
	L = np.delete(L, _idx, axis=0)
	return L, P

def perfcmp(point, score):
    """
        point: `np.array`
        score: `np.array`
    """
    # `point[1]` is decodeWidth
    # scale c.c. and power dissipation
    ref = [reference[point[1] - 1][0] / 10000, reference[point[1] - 1][1] * 100]

    return -adrs(ref, score)
    # return hyper_volume(ref, score)

def _exist_duplicate(s, heap):
    """
        s: `float`
        heap: `list`
    """
    for item in heap:
        if s == item[0]:
            return True

    return False

def sa_search(rt1, rt2, design_space, top_k=5, n_iter=500,
    early_stop=100, parallel_size=128, log_interval=50):
    """
        `rt1`: <sklearn.model>
        `rt2`: <sklearn.model>
        `design_space`: <DesignSpace>
        return:
        `heap_items`: <list> (<tuple> in <list>), specifically,
        <tuple> is (<int>, <list>) or (hv, configurations)
    """
    points = design_space.random_sample(parallel_size)
    _scores = (predict_model(rt1, points) + predict_model(rt2, points)) / 2
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
            new_points[i] = design_space.random_walk(p)
        _new_scores = (predict_model(rt1, new_points) + predict_model(rt2, new_points)) / 2
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
            print("[INFO]: %s" % msg)

    # big -> small
    heap_items.sort(key=lambda item: -item[0])

    return heap_items

def generate_LU(dataset):
    """
        generate L and U dataset
        initially, L contains 64 designs
    """
    idx = random.sample(range(len(dataset)), 64)
    L = []
    for i in idx:
        L.append(dataset[i])
    return np.array(L), np.delete(dataset, idx, axis=0)

def generate_P_from_U(dataset, p=8):
    idx = random.sample(range(len(dataset)), p)
    P = []
    for i in idx:
        P.append(dataset[i])
    return np.array(P), np.delete(dataset, idx, axis=0)

def dump_model(H1, H2):
    # save
    output = os.path.join(
        configs["model-output-path"],
        "dac16-1.mdl"
    )
    joblib.dump(H1, output)
    output = os.path.join(
        configs["model-output-path"],
        "dac16-2.mdl"
    )
    joblib.dump(H2, output)

def main():
    dataset = load_dataset(configs["dataset-output-path"])

    L, U = generate_LU(dataset)

    P, U = generate_P_from_U(U, p=32)

    H1, H2 = init_actboost(L)

    K = 5
    W = 16
    N = 4
    for i in range(K):
        x, y = split_dataset(P)
        y1 = predict_model(H1, x)
        y2 = predict_model(H2, x)
        cv = calc_cv(y1, y2)

        # choosse `N` from top `W` randomly
        idx = random.sample(range(W), N)
        _data = []
        for j in idx:
            _data.append(
                np.concatenate((x[cv[j][0]] * 90000, y[cv[j][0]] / 10))
            )
        _data = np.array(_data)

        # move the newly labeled samples from P to L
        for j in _data:
            L = np.insert(L, len(L), j, axis=0)
        P = np.delete(P, idx, axis=0)

        # rebuild H1, H2 by new set L
        H1, H2 = init_actboost(L)

        # replenish P by choosing `N` from `U` randomly
        idx = random.sample(range(len(U)), N)
        for j in idx:
            P = np.insert(P, len(P), U[j], axis=0)
        U = np.delete(U, idx, axis=0)
    # evaluate on `U`
    x, y = split_dataset(U)
    _y = (predict_model(H1, x) + predict_model(H2, x)) / 2
    msg = "[INFO]: RMSE of c.c.: %.8f, " % rmse(y[:, 0], _y[:, 0]) + \
        "RMSE of power: %.8f on %d test data" % (rmse(y[:, 1], _y[:, 1]), len(U))
    strflush(msg)

    dump_model(H1, H2)

    hv = calc_hv(x, _y)
    # transform `_y` to top predictions
    pred = []
    for (idx, _hv) in hv:
        pred.append(_y[idx])
    pred = np.array(pred)

    # highlight `self.unsampled`
    highlight = []
    for (idx, _hv) in hv:
        highlight.append(y[idx])
    highlight = np.array(highlight)
    # visualize
    plot_predictions_with_gt(
        y,
        pred,
        highlight,
        top_k=configs["top-k"],
        title="dac16",
        output=configs["fig-output-path"],
    )

    # write results
    data = []
    for (idx, _hv) in hv:
        data.append(np.concatenate((x[idx], y[idx])))
    data = np.array(data)
    output = os.path.join(
        configs["rpt-output-path"],
        "dac16" + ".rpt"
    )
    print("[INFO]: saving results to %s" % output)
    write_csv(output, data[:configs["top-k"], :])

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    main()

