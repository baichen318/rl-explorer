import sys
sys.path.append("..")
import os
import heapq
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from sample import RandomSampler
from util import parse_args, get_configs, load_dataset, \
    split_dataset, rmse, hyper_volume, adrs, strflush
from vis import plot_predictions_with_gt
from space import parse_design_space
from handle_data import reference

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

class SurrogateModel(object):
    """
        SurrogateModel: 12 traditional basic ML models
    """
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.preprocess = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.model = self.init()

    def init(self):
	    # return Ridge(alpha=0.01, tol=1e-8)
	    return Ridge(tol=1e-8)

    def fit(self, x, y):
        """
            x: <numpy.ndarray> (M x 19)
            y: <numpy.ndarray> (M x 2)
        """
        x = self.preprocess.fit_transform(x)
        y = np.log(y)
        self.model.fit(x, y)

    def predict(self, x):
        """
            x: <numpy.ndarray> (M x 19)
        """
        x = self.preprocess.fit_transform(x)
        return np.exp(self.model.predict(x))

class BayesianOptimization(object):
    """docstring for BayesianOptimization"""
    def __init__(self, configs):
        super(BayesianOptimization, self).__init__()
        self.configs = configs
        # build model
        self.model = SurrogateModel()
        self.space = parse_design_space(self.configs["design-space"])
        self.sampler = RandomSampler(self.configs)
        self.dataset = load_dataset(configs["dataset-output-path"])
        self.unsampled = None

    def sample(self, dataset):
        return self.sampler.sample(dataset)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def run(self):
        x, y = [], []
        dataset = self.dataset.copy()
        for i in range(self.configs["max-bo-steps"]):
            dataset, sample = self.sample(dataset)
            _x, _y = split_dataset(sample)
            # add `_x` & `_y` to `x` & `y` respectively
            if len(x) == 0:
                for j in _x:
                    x.append(j)
                x = np.array(x)
                for j in _y:
                    y.append(j)
                y = np.array(y)
            else:
                for j in _x:
                    x = np.insert(x, len(x), j, axis=0)
                for j in _y:
                    y = np.insert(y, len(y), j, axis=0)
            self.fit(x, y)

            __y = self.predict(x)

            msg = "[INFO]: Training Iter %d: RMSE of c.c.: %.8f, " % ((i + 1), rmse(y[:, 0], __y[:, 0])) + \
                "RMSE of power: %.8f on %d train data" % (rmse(y[:, 1], __y[:, 1]), len(x))
            strflush(msg)
            # validate
            __x, __y = split_dataset(dataset)
            ___y = self.predict(__x)
            msg = "[INFO]: Testing Iter %d: RMSE of c.c.: %.8f, " % ((i + 1), rmse(__y[:, 0], ___y[:, 0])) + \
                "RMSE of power: %.8f on %d test data" % (rmse(__y[:, 1], ___y[:, 1]), len(__x))
            strflush(msg)

        self.unsampled = dataset

    def validate(self, logger=None):
        x, y = split_dataset(self.unsampled)
        _y = self.predict(x)
        msg = "[INFO]: RMSE of c.c.: %.8f, " % rmse(y[:, 0], _y[:, 0]) + \
            "RMSE of power: %.8f on %d test data" % (rmse(y[:, 1], _y[:, 1]), len(self.unsampled))
        strflush(msg)

        # visualize
        plot_predictions_with_gt(
            y,
            _y,
            title="HPCA07",
            output=self.configs["fig-output-path"]
        )

    def save(self):
        output = os.path.join(
            self.configs["model-output-path"],
            "hpca07.mdl"
        )
        joblib.dump(
            self.model,
            output
        )
        msg = "[INFO]: saving model to %s" % output
        strflush(msg)

    def load(self):
        output = os.path.join(
            self.configs["model-output-path"],
            "hpca07.mdl"
        )
        self.model = joblib.load(output)
        msg = "[INFO]: loading model from %s" % output
        strflush(msg)

def main():
    manager = BayesianOptimization(configs)
    manager.run()
    manager.validate()
    manager.save()

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    main()
