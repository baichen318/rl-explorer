import os
import heapq
import time
import numpy as np
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

def create_model():
	model = MLPRegressor(
		hidden_layer_sizes=(16, 2),
		activation="logistic",
		solver="adam",
		alpha=0.0001,
		learning_rate="adaptive",
		learning_rate_init=0.001,
		max_iter=5000,
		momentum=0.5,
		early_stopping=True
	)

	return model

def get_dataset():
	dataset, title = read_csv_v2(configs["dataset-output-path"])
	dataset = validate(dataset)

	# scale dataset to balance c.c. and power dissipation
	dataset[:, -2] = dataset[:, -2] / 10000
	dataset[:, -1] = dataset[:, -1] * 100

	return dataset

def perfcmp(point, score):
    """
        point: `np.array`
        score: `np.array`
    """
    # `point[1]` is decodeWidth
    # scale c.c. and power dissipation
    ref = [reference[point[1] - 1][0] / 10000, reference[point[1] - 1][1] * 100]
    return hyper_volume(ref, score)

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
        `model`: <sklearn.model>
        `design_space`: <DesignSpace>
        return:
        `heap_items`: <list> (<tuple> in <list>), specifically,
        <tuple> is (<int>, <list>) or (hv, configurations)
    """
    global pf

    points = design_space.random_sample(parallel_size)
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
            new_points[i] = design_space.random_walk(p)
        _new_scores = model.predict(points)
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

def main():
	global dataset
	global pf
	dataset = get_dataset()

	kf = kFold()
	index = kf.split(dataset)

	perf = float('inf')
	cnt = 0
	mse_l, mse_p, r2_l, r2_p, mape_l, mape_p = 0, 0, 0, 0, 0, 0
	model = create_model()
	for train_index, test_index in index:
		print("train:\n%s" % str(train_index))
		print("test:\n%s" % str(test_index))

		x_train, y_train = split_dataset(dataset[train_index])
		x_test, y_test = split_dataset(dataset[test_index])

		model.fit(x_train, y_train)

		_y = model.predict(x_test)

		# analysis
		_mse_l = mse(_y[:, 0], y_test[:, 0])
		_mse_p = mse(_y[:, 1], y_test[:, 1])
		_r2_l = r2(_y[:, 0], y_test[:, 0])
		_r2_p = r2(_y[:, 1], y_test[:, 1])
		_mape_l = mape(_y[:, 0], y_test[:, 0])
		_mape_p = mape(_y[:, 1], y_test[:, 1])
		print("[INFO]: MSE (latency): %.8f, MSE (power): %.8f" % (_mse_l, _mse_p))
		print("[INFO]: R2 (latency): %.8f, R2 (power): %.8f" % (_r2_l, _r2_p))
		print("[INFO]: MAPE (latency): %.8f, MAPE (power): %.8f" % (_mape_l, _mape_p))
		if perf > (0.5 * _mape_l + 0.5 * _mape_p):
			perf = (0.5 * _mape_l + 0.5 * _mape_p)
			joblib.dump(
				model,
				os.path.join(
					"model",
					"asplos06.mdl"
				)
			)
			min_mape_l = _mape_l
			min_mape_p = _mape_p

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
		"average R2 (power): %.8f" % float(r2_p / cnt)
	print(msg)

	model = joblib.load(
		os.path.join(
			"model",
			"asplos06.mdl"
		)
	)
	exit()
	# search
	heap = sa_search(model, design_space, top_k=50,
		n_iter=10000, early_stop=5000, parallel_size=1024, log_interval=100)
    # saving results
	write_csv(
		os.path.join(
			"rpts",
			"asplos06" + '-prediction.rpt'
		),
		heap,
		mode='w'
	)


if __name__ == "__main__":
	# global variables
	dataset = None
	pf = None
	argv = parse_args()
	configs = get_configs(argv.configs)
	design_space = parse_design_space(
		configs["design-space"]
	)
	main()
