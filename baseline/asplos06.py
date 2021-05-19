import sys
sys.path.append("..")
import os
import heapq
import time
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from sample import random_sample
from util import parse_args, get_configs, load_dataset, \
    strflush, write_txt, get_pareto_points, recover_data
from vis import plot_pareto_set
from space import parse_design_space

seed = 2021
np.random.seed(seed)

class SurrogateModel(object):
    """
        SurrogateModel: ASPLOS06 paper
    """
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.model = self.init()

    def init(self):
        return MultiOutputRegressor(
            MLPRegressor(
                hidden_layer_sizes=(16,),
                activation="logistic",
                solver="adam",
                learning_rate="adaptive",
                learning_rate_init=0.001,
                max_iter=1000,
                momentum=0.5,
                early_stopping=False
            )
        )

    def fit(self, x, y):
        """
            x: <numpy.ndarray> (M x 19)
            y: <numpy.ndarray> (M x 2)
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
            x: <numpy.ndarray> (M x 19)
        """
        return self.model.predict(x)

    def save(self, output):
        joblib.dump(
            self.model,
            output
        )
        msg = "[INFO]: saving model to %s" % output
        strflush(msg)

    def load(self, path):
        self.model = joblib.load(path)
        msg = "[INFO]: loading model from %s" % path
        strflush(msg)

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

def sa_search(model, dataset, logger=None, top_k=5, n_iter=500,
    early_stop=100, parallel_size=128, log_interval=50):
    """
        model: <sklearn.model>
        dataset: <tuple>
        return:
        heap_items: <list> (<tuple> in <list>), specifically,
        <tuple> is (<int>, <list>) or (hv, configurations)
    """
    x, y = dataset
    n_dim = x.shape[-1]
    (x, y), (points, _y) = random_sample(configs, x, y, batch=parallel_size)
    scores = model.predict(points)
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
        for i, p in enumerate(points):
            new_points[i], x, y = random_walk(p, x, y)
        new_scores = model.predict(new_points)
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

def main():
    x, y = load_dataset(configs["dataset-output-path"])
    total_x, total_y = x.copy(), y.copy()
    # generate data
    (x, y), (_x, _y) = random_sample(configs, x, y, batch=configs["initialize"])
    model = SurrogateModel()
    # initialize the model
    model.fit(_x, _y)
    # search
    heap = sa_search(
        model,
        (x, y),
        top_k=14,
        n_iter=70,
        early_stop=35,
        parallel_size=8,
        log_interval=10
    )
    pred = []
    for i, point in heap:
        pred.append(design_space.point2knob(point))
    pred = np.array(pred)

    # add `_x` into `pred`
    for i in _x:
        pred = np.insert(pred, len(pred), i, axis=0)
    # get corresponding `_y`
    idx = []
    for _pred in pred:
        for i in range(len(total_x)):
            if (np.abs(_pred - total_x[i]) < 1e-4).all():
                idx.append(i)
                break
    pareto_set = get_pareto_points(total_y[idx])
    plot_pareto_set(
        recover_data(pareto_set),
        dataset_path=configs["dataset-output-path"],
        output=os.path.join(
            configs["fig-output-path"],
            "asplos06" + ".pdf"
        )
    )

    # write results
    # pareto set
    write_txt(
        os.path.join(
            configs["rpt-output-path"],
            "asplos06" + "-pareto-set.rpt"
        ),
        np.array(pareto_set),
        fmt="%f"
    )
    # model
    model.save(
        os.path.join(
            configs["model-output-path"],
            "asplos06" + ".mdl"
        )
    )

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    design_space = parse_design_space(configs["design-space"])
    main()
