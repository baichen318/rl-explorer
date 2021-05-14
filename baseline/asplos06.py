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
		hidden_layer_sizes=(16,),
		activation="logistic",
		solver="adam",
		learning_rate="adaptive",
		learning_rate_init=0.001,
		max_iter=10000,
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

def sa_search(model_l, model_p, design_space, logger=None, top_k=5, n_iter=500,
    early_stop=100, parallel_size=128, log_interval=50):
    """
        `model`: <sklearn.model>
        `design_space`: <DesignSpace>
        return:
        `heap_items`: <list> (<tuple> in <list>), specifically,
        <tuple> is (<int>, <list>) or (hv, configurations)
    """

    points = design_space.random_sample(parallel_size)
    _scores_l = model_l.predict(points)
    _scores_p = model_p.predict(points)
    temp_scores = []
    for i in range(len(_scores_l)):
        temp_scores.append([_scores_l[i], _scores_p[i]])
    temp_scores = np.array(temp_scores)
    scores = np.empty(parallel_size)
    for i, (p, s) in enumerate(zip(points, temp_scores)):
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
        _scores_l = model_l.predict(points)
        _scores_p = model_p.predict(points)
        temp_scores = []
        for i in range(len(_scores_l)):
            temp_scores.append([_scores_l[i], _scores_p[i]])
        temp_scores = np.array(temp_scores)
        new_scores = np.empty(parallel_size)
        for i, (p, s) in enumerate(zip(new_points, temp_scores)):
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
	dataset = get_dataset()

	kf = kFold()
	index = kf.split(dataset)

	perf = float('inf')
	cnt = 0
	mse_l, mse_p, r2_l, r2_p, mape_l, mape_p = 0, 0, 0, 0, 0, 0
	max_r2_l, max_r2_p = -float('inf'), -float('inf')
	model_l = create_model()
	model_p = create_model()
	for train_index, test_index in index:
		print("train:\n%s" % str(train_index))
		print("test:\n%s" % str(test_index))

		x_train, y_train = split_dataset(dataset[train_index])
		x_test, y_test = split_dataset(dataset[test_index])

		model_l.fit(x_train, y_train[:, 0])
		model_p.fit(x_train, y_train[:, 1])

		_y_l = model_l.predict(x_test)
		_y_p = model_p.predict(x_test)

		# analysis
		_mse_l = mse(y_test[:, 0], _y_l)
		_mse_p = mse(y_test[:, 1], _y_p)
		_r2_l = r2(y_test[:, 0], _y_l)
		_r2_p = r2(y_test[:, 1], _y_p)
		_mape_l = mape(y_test[:, 0], _y_l)
		_mape_p = mape(y_test[:, 1], _y_p)
		print("[INFO]: MSE (latency): %.8f, MSE (power): %.8f" % (_mse_l, _mse_p))
		print("[INFO]: R2 (latency): %.8f, R2 (power): %.8f" % (_r2_l, _r2_p))
		print("[INFO]: MAPE (latency): %.8f, MAPE (power): %.8f" % (_mape_l, _mape_p))
		if perf > (0.5 * _mape_l + 0.5 * _mape_p):
			perf = (0.5 * _mape_l + 0.5 * _mape_p)
			joblib.dump(
				model_l,
				os.path.join(
					"model",
					"asplos06-cc.mdl"
				)
			)
			joblib.dump(
				model_p,
				os.path.join(
					"model",
					"asplos06-power.mdl"
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

	model_l = joblib.load(
		os.path.join(
			"model",
			"asplos06-cc.mdl"
		)
	)
	model_p = joblib.load(
		os.path.join(
			"model",
			"asplos06-power.mdl"
		)
	)
	# search
	heap = sa_search(model_l, model_p, design_space, top_k=50,
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
	argv = parse_args()
	configs = get_configs(argv.configs)
	design_space = parse_design_space(
		configs["design-space"]
	)
	main()
