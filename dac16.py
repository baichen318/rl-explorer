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
from util import parse_args, get_configs, read_csv_v2, mse, r2, mape, hyper_volume, write_csv, adrs
from model import split_dataset, kFold
from vis import validate
from space import parse_design_space
from handle_data import reference

def create_model(hidden=4):
	mlp = MLPRegressor(
		hidden_layer_sizes=(16, hidden),
		activation="relu",
		solver="adam",
		alpha=0.0001,
		learning_rate="adaptive",
		learning_rate_init=0.001,
		max_iter=10000,
		momentum=0.5,
		early_stopping=True
	)
	rt = MultiOutputRegressor(
		AdaBoostRegressor(
			n_estimators=100,
			learning_rate=0.001,
			loss='linear'
		)
	)

	return {
		"mlp": mlp,
		"rt": rt
	}

def train_model(h, x, y):
	h["mlp"].fit(x, y)
	h["rt"].fit(h["mlp"].predict(x), y)
	return h

def predict_model(h, x):
	return h["rt"].predict(h["mlp"].predict(x))

def init_actboost(P):
	M1 = 6
	M2 = 8
	x, y = split_dataset(P)
	h1 = create_model(M1)
	h2 = create_model(M2)

	h1 = train_model(h1, x, y)
	h2 = train_model(h2, x, y)
	
	return h1, h2

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

	cv = sorted(cv, key=lambda t:t[1])

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

def main():
	global dataset
	dataset = get_dataset()

	kf = kFold()
	index = kf.split(dataset)

	K = 50
	W = 16
	N = 4
	cnt = 0
	perf = float('inf')
	mse_l, mse_p, r2_l, r2_p, mape_l, mape_p = 0, 0, 0, 0, 0, 0
	max_r2_l, max_r2_p = -float('inf'), -float('inf')
	for train_index, test_index in index:
		print("train:\n%s" % str(train_index))
		print("test:\n%s" % str(test_index))
		L, P = create_pool(dataset[train_index], p=len(dataset[train_index]) - 64)
		h1, h2 = init_actboost(L)
		for i in range(K):
			if len(P) > 0:
				x, y = split_dataset(P)
				ret1 = predict_model(h1, x)
				ret2 = predict_model(h2, x)

				# calculate c.v.
				cv = calc_cv(ret1, ret2)
				# move the newly labeled samples from `P` to `L`
				if len(P) < W:
					for j in range(len(P)):
						L = np.insert(L, len(L), P[j], axis=0)
					P = np.array([])
				else:
					idx = random.sample(range(1, W), N)
					for j in idx:
						data = np.concatenate((x[cv[j][0]], y[cv[j][0]]))
						L = np.insert(L, len(L), data, axis=0)
						_i = 0
						temp = []
						for d in P:
							if ((data - d < 1e-5)).all():
								temp.append(_i)
								break
							_i += 1
						P = np.delete(P, temp, axis=0)
				# rebuild
				x, y = split_dataset(L)
				h1 = train_model(h1, x, y)
				h2 = train_model(h2, x, y)
		# test
		x_test, y_test = split_dataset(dataset[test_index])
		ret1 = predict_model(h1, x_test)
		ret2 = predict_model(h2, x_test)
		ret = (ret1 + ret2) / 2

		# analysis
		_mse_l = mse(y_test[:, 0], ret[:, 0])
		_mse_p = mse(y_test[:, 1], ret[:, 1])
		_r2_l = r2(y_test[:, 0], ret[:, 0])
		_r2_p = r2(y_test[:, 1], ret[:, 1])
		_mape_l = mape(y_test[:, 0], ret[:, 0])
		_mape_p = mape(y_test[:, 1], ret[:, 1])
		print("[INFO]: MSE (latency): %.8f, MSE (power): %.8f" % (_mse_l, _mse_p))
		print("[INFO]: R2 (latency): %.8f, R2 (power): %.8f" % (_r2_l, _r2_p))
		print("[INFO]: MAPE (latency): %.8f, MAPE (power): %.8f" % (_mape_l, _mape_p))
		if perf > (0.5 * _mape_l + 0.5 * _mape_p):
			perf = (0.5 * _mape_l + 0.5 * _mape_p)
			joblib.dump(
				h1,
				os.path.join(
					"model",
					"dac16-h1.mdl"
				)
			)
			joblib.dump(
				h2,
				os.path.join(
					"model",
					"dac16-h2.mdl"
				)
			)
			min_mape_l = _mape_l
			min_mape_p = _mape_p

		if max_r2_l < _r2_l:
			max_r2_l = _r2_l
		if max_r2_p < _r2_p:
			max_r2_p = _r2_p
		cnt += 1
		mse_l += _mse_l
		mse_p += _mse_p
		r2_l += _r2_l
		r2_p += _r2_p
		mape_l += _mape_l
		mape_p += _mape_p
	msg = "[INFO]: achieve the best performance: MAPE (latency): %.8f " %  min_mape_l + \
		"MAPE (power): %.8f in one round. " % min_mape_p + \
		"Average MAPE (latency): %.8f, " % float(mape_l / cnt) + \
		"average MAPE (power): %.8f, " % float(mape_p / cnt) + \
		"average R2 (latency): %.8f, " % float(r2_l / cnt) + \
		"average R2 (power): %.8f, " % float(r2_p / cnt) + \
        "the best R2 (latency): %.8f, " % max_r2_l + \
        "the best R2 (power): %.8f" % max_r2_p
	print(msg)

	h1 = joblib.load(
		os.path.join(
			"model",
			"dac16-h1.mdl"
		)
	)
	h2 = joblib.load(
		os.path.join(
			"model",
			"dac16-h2.mdl"
		)
	)
	# search
	heap = sa_search(h1, h2, design_space, top_k=50,
		n_iter=10000, early_stop=5000, parallel_size=1024, log_interval=100)
    # saving results
	write_csv(
		os.path.join(
			"rpts",
			"dac16" + '-prediction.rpt'
		),
		heap,
		mode='w'
	)


if __name__ == "__main__":
	# global variables
	dataset = None
	argv = parse_args()
	configs = get_configs(argv.configs)
	design_space = parse_design_space(
		configs["design-space"]
	)
	main()
