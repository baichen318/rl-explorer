import sys
sys.path.append("..")
import os
import heapq
import random
import numpy as np
from time import time
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from util import parse_args, get_configs, load_dataset, \
    rmse, strflush, write_txt, get_pareto_points, recover_data
from vis import plot_pareto_set
from sample import random_sample
from space import parse_design_space

seed = int(time())
random.seed(seed)
np.random.seed(seed)

def create_model(hidden):
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

def init_actboost(L_x, L_y):
	H1 = create_model(hidden=6)
	H2 = create_model(hidden=8)

	train_model(H1, L_x, L_y)
	train_model(H2, L_x, L_y)
	
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

def random_walk_v2(x, y, batch):
    return random_sample(configs, x, y, batch=batch)

def sa_search(model, dataset, logger=None, top_k=5, n_iter=500,
    early_stop=100, parallel_size=128, log_interval=50):
    """
        model: <sklearn.model>
        dataset: <tuple>
        return:
        heap_items: <list> (<tuple> in <list>), specifically,
        <tuple> is (<int>, <list>) or (hv, configurations)
    """
    H1, H2 = model
    x, y = dataset
    n_dim = x.shape[-1]
    (x, y), (points, _y) = random_sample(configs, x, y, batch=parallel_size)
    y1 = predict_model(H1, points)
    y2 = predict_model(H2, points)
    scores = (y1 + y2) /2
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
        # for i, p in enumerate(points):
        #     new_points[i], x, y = random_walk(p, x, y)
        (x, y), (new_points, _y) = random_walk_v2(x, y, parallel_size)
        y1 = predict_model(H1, new_points)
        y2 = predict_model(H2, new_points)
        new_scores = (y1 + y2) / 2
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

def generate_LU(x, y):
    """
        generate L and U dataset
        initially, L contains 64 designs
    """
    idx = random.sample(range(len(x)), 64)
    L_x, L_y = [], []
    for i in idx:
        L_x.append(x[i])
        L_y.append(y[i])
    return (np.array(L_x), np.array(L_y)), \
        (np.delete(x, idx, axis=0), np.delete(y, idx, axis=0))

def generate_P_from_U(x, y, p=8):
    idx = random.sample(range(len(x)), p)
    P_x, P_y = [], []
    for i in idx:
        P_x.append(x[i])
        P_y.append(y[i])
    return (np.array(P_x), np.array(P_y)), \
        (np.delete(x, idx, axis=0), np.delete(y, idx, axis=0))

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
    X, Y = load_dataset(configs["dataset-output-path"])

    (L_x, L_y), (U_x, U_y) = generate_LU(X, Y)

    (P_x, P_y), (U_x, U_y) = generate_P_from_U(U_x, U_y, p=32)

    H1, H2 = init_actboost(L_x, L_y)

    K = 5
    W = 16
    N = 4
    for i in range(K):
        y1 = predict_model(H1, P_x)
        y2 = predict_model(H2, P_x)
        cv = calc_cv(y1, y2)

        # choosse `N` from top `W` randomly
        idx = random.sample(range(W), N)
        temp_x, temp_y = [], []
        for j in idx:
            temp_x.append(P_x[cv[j][0]])
            temp_y.append(P_y[cv[j][0]])
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)

        # move the newly labeled samples from P to L
        for j in range(len(temp_x)):
            L_x = np.insert(L_x, len(L_x), temp_x[j], axis=0)
            L_y = np.insert(L_y, len(L_y), temp_y[j], axis=0)
        P_x = np.delete(P_x, idx, axis=0)
        P_y = np.delete(P_y, idx, axis=0)

        # rebuild H1, H2 by new set L
        H1, H2 = init_actboost(L_x, L_y)

        # replenish P by choosing `N` from `U` randomly
        idx = random.sample(range(len(U_x)), N)
        for j in idx:
            P_x = np.insert(P_x, len(P_x), U_x[j], axis=0)
            P_y = np.insert(P_y, len(P_y), U_y[j], axis=0)
        U_x = np.delete(U_x, idx, axis=0)
        U_y = np.delete(U_y, idx, axis=0)

    # add `P` to `U`
    for i in range(len(P_x)):
        U_x = np.insert(U_x, len(U_x), P_x[i], axis=0)
        U_y = np.insert(U_y, len(U_y), P_y[i], axis=0)

    # search
    heap = sa_search(
        (H1, H2),
        (U_x, U_y),
        top_k=14,
        n_iter=100,
        early_stop=50,
        parallel_size=8,
        log_interval=10
    )
    pred = []
    for i, point in heap:
        pred.append(design_space.point2knob(point))
    pred = np.array(pred)

    # add `L_x` into `pred`
    for i in L_x:
        pred = np.insert(pred, len(pred), i, axis=0)
    # get corresponding `_y`
    idx = []
    for _pred in pred:
        for i in range(len(X)):
            if (np.abs(_pred - X[i]) < 1e-4).all():
                idx.append(i)
                break
    pareto_set = get_pareto_points(Y[idx])
    plot_pareto_set(
        recover_data(pareto_set),
        dataset_path=configs["dataset-output-path"],
        output=os.path.join(
            configs["fig-output-path"],
            "dac16" + ".pdf"
        )
    )

    # write results
    # pareto set
    write_txt(
        os.path.join(
            configs["rpt-output-path"],
            "dac16" + "-pareto-set.rpt"
        ),
        np.array(pareto_set),
        fmt="%f"
    )
    dump_model(H1, H2)

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    design_space = parse_design_space(configs["design-space"])
    main()
