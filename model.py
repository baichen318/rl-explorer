# Author: baichen318@gmail.com

import sys
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.externals import joblib

sys.path.append("BayesianOptimization")
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from collections import OrderedDict
from multiprocessing import Process, Queue
# from vlsi.vlsi import vlsi_flow
from space import parse_design_space
from search import sa_search
from util import parse_args, get_configs, get_config_v2, read_csv, read_csv_v2, if_exist, \
    calc_mape, point2knob, knob2point, create_logger, is_pow2, mkdir, \
    execute, mse, r2, mape, write_csv
from vis import handle_vis
from exception import UnDefinedException
from vlsi.vlsi import offline_vlsi_flow_v2

class GP(object):
    FEATURES = []

    def __init__(self, configs):
        self.design_space = configs['design-space']
        self.iteration = configs['iteration']
        self.parallel = configs["parallel"]
        self.report_output_path = os.path.join(
            os.path.abspath(os.curdir),
            configs['report-output-path']
        )
        self.model_output_path = os.path.join(
            os.path.abspath(os.curdir),
            configs["model-output-path"]
        )
        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        self.logger = create_logger("logs", "gp")
        self.visited = set()

        # if_exist("data/init-model.csv", strict=True)

        # variables used by `GP`
        self.bounds = None
        self.optimizer = None
        self.dims = None
        self.size = None
        self.next = None
        self.metrics = None
        self.idx = None
        self.optimum = None

    def init(self):
        mkdir(os.path.dirname(self.report_output_path))
        self.bounds = self.parse_design_space()
        self.optimizer = BayesianOptimization(
                f=None,
                pbounds=self.bounds,
                verbose=2,
                random_state=1
            )

        # initialize the model
        points, targets = self.read_init_data("data/init-model.csv")
        for i, target in enumerate(targets):
            self.next = points[i]
            self.optimizer.register(params=self.next, target=target)
        self.optimizer.savegp(self.model_output_path)

    def read_init_data(self, path):
        # features latency power
        data = read_csv(path)
        points = []
        metrics = []
        for row in data:
            point = {}
            for name in GP.FEATURES:
                point[name] = int(row[GP.FEATURES.index(name)])
            points.append(point)
            metrics.append(
                self.single_objective_cost_function(row[-2], row[-1])
            )

        return points, metrics

    def parse_design_space(self):
        self.dims = []
        self.size = 1
        bounds = OrderedDict()
        for k, v in self.design_space.items():
            # add `FEATURES`
            GP.FEATURES.append(k)
            # calculate the size of the design space
            if 'candidates' in v.keys():
                temp = v['candidates']
                self.size *= len(temp)
                # generate bounds
                bounds[k] = np.array(temp)
                # generate dims
                self.dims.append(len(temp))
            else:
                assert 'start' in v.keys() and 'end' in v.keys() and \
                    'stride' in v.keys(), "[ERROR]: assert failed. YAML includes errors."
                temp = np.arange(v['start'], v['end'] + 1, v['stride'])
                self.size *= len(temp)
                # generate bounds
                bounds[k] = temp
                # generate dims
                self.dims.append(len(temp))

        return bounds

    def get_features(self, _dict):
        vec = []
        for i in range(len(GP.FEATURES)):
            vec.append(round(_dict[GP.FEATURES[i]]))
        return vec

    def features2knob(self, vec):
        ret = []
        for idx in range(len(vec)):
            ret.append(
                np.argwhere(
                    self.bounds[GP.FEATURES[idx]] == vec[idx]
                )[0][0]
            )

        return ret

    def knob2features(self, vec):
        ret = []
        for idx in range(len(vec)):
            ret.append(
                self.bounds[GP.FEATURES[idx]][vec[idx]]
            )

        return ret

    def features2string(self, vector):

        return '''
  fetchWidth: %d
  decodeWidth: %d
  numFetchBufferEntries: %d
  numRobEntries: %d
  numRasEntries: %d
  numIntPhysRegisters: %d
  numFpPhysRegisters: %d
  numLdqEntries: %d
  numStqEntries: %d
  maxBrCount: %d
  mem_issueWidth: %d
  int_issueWidth: %d
  fp_issueWidth: %d
  DCacheParams_nWays: %d
  DCacheParams_nMSHRs: %d
  DCacheParams_nTLBEntries: %d
  ICacheParams_nWays: %d
  ICacheParams_nTLBEntries: %d
  ICacheParams_fetchBytes: %d
''' % (vector[0], vector[1], vector[2], vector[3], vector[4], vector[5],
    vector[6], vector[7], vector[8], vector[9], vector[10], vector[11],
    vector[12], vector[13], vector[14], vector[15], vector[16], vector[17],
    vector[18])

    def sample(self):
        def _sample():
            self.next = self.get_features(
                self.optimizer.suggest(self.utility)
            )
            self.idx = knob2point(
                self.features2knob(self.next),
                self.dims
            )
            if self.idx in self.visited:
                while self.idx in self.visited:
                    self.next = self.get_features(
                        self.optimizer.suggest(self.utility)
                    )
                    self.idx = knob2point(
                        self.features2knob(self.next),
                        self.dims
                    )
            self.visited.add(self.idx)

        def _batch_sample():
            next = self.get_features(
                self.optimizer.suggest(self.utility)
            )
            idx = knob2point(
                self.features2knob(next),
                self.dims
            )
            if idx in self.visited:
                while idx in self.visited:
                    next = self.get_features(
                        self.optimizer.suggest(self.utility)
                    )
                    idx = knob2point(
                        self.features2knob(next),
                        self.dims
                    )
            self.idx.append(idx)
            self.next.append(next)
            self.visited.add(idx)


        self.logger.info("Sampling...")
        if self.parallel:
            self.idx = []
            self.next = []
            # parallel 4 `vlsi_flow`
            for i in range(4):
                _batch_sample()

            assert len(self.idx) == len(self.next), \
                "[ERROR]: assert failed. " \
                "idx: {}, next: {}".format(len(self.idx), len(self.next))
        else:
            _sample()

    def query(self):
        def _construct_kwargs(idx, vec):
            return {
                "dims": self.dims,
                "size": self.size,
                "idx": idx,
                "configs": vec,
                "logger": self.logger
            }

        self.logger.info("Querying...")
        if self.parallel:
            queue = [Queue() for i in range(len(self.idx))]
            processes = []
            for _idx in range(len(self.idx)):
                kwargs = _construct_kwargs(self.idx[_idx], self.next[_idx])
                p = Process(target=vlsi_flow, args=(kwargs, queue[_idx],))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            # latency, power & area
            for _idx in range(len(queue)):
                _queue = queue[_idx].get()
                self.optimizer.register(
                    params=self.next[_idx],
                    target=-self.single_objective_cost_function(
                        _queue["latency"],
                        _queue["power"]
                    )
                )
        else:
            kwargs = {
                "dims": self.dims,
                "size": self.size,
                "idx": self.idx,
                "configs": self.next,
                "logger": self.logger
            }
            # latency, power & area
            self.metrics = vlsi_flow(kwargs)
            self.optimizer.register(
                params=self.next,
                target=-self.single_objective_cost_function(
                    self.metrics["latency"],
                    self.metrics["power"]
                )
            )

    def record(self):
        if self.parallel:
            for _idx in range(len(self.next)):
                with open(self.report_output_path, 'a') as f:
                    msg = '''
The parameter of %s is: %s
                    ''' % (
                        self.idx[_idx],
                        self.features2string(self.next[_idx])
                    )
                    self.logger.info(msg)
                    f.write(msg)
        else:
            msg = '''
The parameter is: %s
        ''' % self.features2string(self.next)
            self.logger.info(msg)
            with open(self.report_output_path, 'a') as f:
                f.write(msg)

    def final_record(self):
        self.optimum = self.optimizer.max["params"]
        msg = '''
The best result is: %s
        ''' % self.features2string(
                self.get_features(self.optimum)
            )
        self.logger.info(msg)
        with open(self.report_output_path, 'a') as f:
            f.write(msg)

    def verification(self):
        configs = self.get_features(self.optimum)
        self.idx = knob2point(
            self.features2knob(
                configs
            ),
            self.dims
        )
        self.optimizer.savegp(self.model_output_path)
        kwargs = {
            'dims': self.dims,
            'size': self.size,
            'idx': self.idx,
            "configs": configs,
            'logger': self.logger
        }
        # latency, power & area
        self.metrics = vlsi_flow(kwargs)

        self.logger.info("idx: %s metrics: %s" % (self.idx, self.metrics))

    def single_objective_cost_function(self, latency, power):

        return 1e-7 * latency + power

def split_dataset(dataset):
    # split dataset into x label & y label
    # dataset: `np.array`
    x = []
    y = []
    for data in dataset:
        x.append(data[0:-2])
        y.append(data[-2:])

    return np.array(x), np.array(y)

def validate(dataset):
    data = []
    for item in dataset:
        _data = []
        for i in item:
            if '.' in i:
                _data.append(float(i))
            else:
                _data.append(int(i))
        data.append(_data)
    data = np.array(data)

    return data

def kFold():
    return KFold(n_splits=10)

def create_model(method):
    if method == "lr":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(n_jobs=-1)
    elif method == "lasso":
        from sklearn.linear_model import Lasso
        model = Lasso()
    elif method == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge()
    elif method == "svr":
        from sklearn.svm import SVR
        model = MultiOutputRegressor(
            SVR()
        )
    elif method == "lsvr":
        from sklearn.svm import LinearSVR
        model = MultiOutputRegressor(
            LinearSVR()
        )
    elif method == "xgb":
        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100
            )
        )
    elif method == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = MultiOutputRegressor(
            RandomForestRegressor()
        )
    elif method == "ab":
        from sklearn.ensemble import AdaBoostRegressor
        model = MultiOutputRegressor(
            AdaBoostRegressor()
        )
    elif method == "gb":
        from sklearn.ensemble import GradientBoostingRegressor
        model = MultiOutputRegressor(
            GradientBoostingRegressor()
        )
    elif method == "bg":
        from sklearn.ensemble import BaggingRegressor
        model = MultiOutputRegressor(
            BaggingRegressor()
        )
    elif method == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, \
            ExpSineSquared , DotProduct, ConstantKernel
        # kernel = ConstantKernel(1.0, (1e-3, 1000)) * \
        #     RationalQuadratic(1.0, 1.0, (1e-5, 1e5), (1e-5, 1e5)) + WhiteKernel(0, (1e-3, 1000))
        kernel = ConstantKernel(1.0, (1e-3, 1000)) * \
            DotProduct(1.0,(1e-5, 1e5)) + WhiteKernel(0.1, (1e-3, 1000))
        model = MultiOutputRegressor(
            GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b')
        )
    elif method == "br":
        from sklearn.linear_model import BayesianRidge
        model = MultiOutputRegressor(
            BayesianRidge()
        )
    else:
        raise UnDefinedException("%s not supported" % method)
    return model

def regression(method, dataset, index):
    def analysis(model, gt, predict):
        # coefficients
        if hasattr(model, "coef_"):
            print("[INFO]: coefficients of LinearRegression:\n", model.coef_)
        elif hasattr(model, "feature_importances_"):
            print("[INFO]: significance of features:\n", model.feature_importances_)

        mse_l = mse(gt[:, 0], predict[:, 0])
        mse_p = mse(gt[:, 1], predict[:, 1])
        r2_l = r2(gt[:, 0], predict[:, 0])
        r2_p = r2(gt[:, 1], predict[:, 1])
        mape_l = mape(gt[:, 0], predict[:, 0])
        mape_p = mape(gt[:, 1], predict[:, 1])
        logger.info("[INFO]: MSE (latency): %.8f, MSE (power): %.8f" % (mse_l, mse_p))
        logger.info("[INFO]: R2 (latency): %.8f, R2 (power): %.8f" % (r2_l, r2_p))
        logger.info("[INFO]: MAPE (latency): %.8f, MAPE (power): %.8f" % (mape_l, mape_p))

        return mse_l, mse_p, r2_l, r2_p, mape_l, mape_p

    model = create_model(method)
    # MSE, R2, MAPE for latency and power
    avg_metrics = [0 for i in range(6)]
    perf = float('inf')
    cnt = 0
    for train_index, test_index in index:
        logger.info("train:\n%s" % str(train_index))
        logger.info("test:\n%s" % str(test_index))
        x_train, y_train = split_dataset(dataset[train_index])
        x_test, y_test = split_dataset(dataset[test_index])
        # train
        model.fit(x_train, y_train)
        # predict
        predict = model.predict(x_test)
        metrics = analysis(model, y_test, predict)
        for i in range(len(avg_metrics)):
            avg_metrics[i] += metrics[i]
        cnt += 1
        if (0.5 * metrics[4] + 0.5 * metrics[5]) < perf:
            perf = 0.5 * metrics[4] + 0.5 * metrics[5]
            # achieve the average minimum in one round
            min_mape_l, min_mape_p = metrics[4], metrics[5]
            joblib.dump(model, configs["model-output-path"])
    msg = "[INFO]: achieve the best performance: MAPE (latency): %.8f, " + \
        "MAPE (power): %.8f in one round. Average MAPE (latency): %.8f, " + \
        "average MAPE (power): %.8f " + "average R2 (latency): %.8f, " + \
        "average R2 (power): %.8f" % (min_mape_l, min_mape_p, avg_metrics[4],
            avg_metrics[5], avg_metrics[2], avg_metrics[3])
    logger.info(msg)

    # test
    model = joblib.load(configs["model-output-path"])
    heap = sa_search(model, design_space, logger, top_k=50,
        n_iter=5000, early_stop=2000, parallel_size=128, log_interval=50)
    # saving results
    write_csv(method + ".predict", heap, mode='a')
    # get the metris
    perf = []
    for (hv, p) in heap:
        if hv != -1:
            perf.append(model.predict(p.reshape(1, -1)).ravel())
    assert len(perf) > 0, "[ERROR]: SA cannot find a good point"
    # visualize
    handle_vis(
        perf,
        os.path.basename(
            configs["model-output-path"]
        ).split('.')[0],
        configs
    )

    return heap

def verify(heap):
    """
        `heap`: <list> (<tuple> in <list>), specifically,
        <tuple> is (<int>, <list>) or (hv, configurations)
        We only extract the first 3 points to verify
    """
    dataset = []
    for (hv, p) in heap:
        if hv != -1:
            dataset.append(p)
    dataset = np.array(dataset)[:3]
    offline_vlsi_flow_v2(dataset)

def handle():
    dataset, title = read_csv_v2(configs["dataset-output-path"])
    dataset = validate(dataset)
    kf = kFold()
    index = kf.split(dataset)

    if configs["model"] == "lr" or \
        configs["model"] == "lasso" or \
        configs["model"] == "ridge" or \
        configs["model"] == "svr" or \
        configs["model"] == "lsvr" or \
        configs["model"] == "xgb" or \
        configs["model"] == "rf" or \
        configs["model"] == "ab" or \
        configs["model"] == "gb" or \
        configs["model"] == "bg" or \
        configs["model"] == "gp" or \
        configs["model"] == "br":
        heap = regression(configs["model"], dataset, index)
    else:
        raise UnDefinedException("%s undefined" % configs["model"])

    verify(heap)

if __name__ == "__main__":
    argv = parse_args()
    configs = get_configs(argv.configs)
    design_space = parse_design_space(
        get_config_v2("configs/design-explorer.yml")["design-space"]
    )
    logger = create_logger("logs", "model")
    handle()
