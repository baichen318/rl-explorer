# Author: baichen318@gmail.com

import sys
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics

sys.path.append("BayesianOptimization")
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from collections import OrderedDict
from multiprocessing import Pool
from vlsi.vlsi import vlsi_flow
from util import parse_args, get_config, read_csv, if_exist, calc_mape, point2knob, knob2point, \
    create_logger, is_pow2, mkdir, execute

class GP(object):
    FEATURES = []

    def __init__(self, configs):
        self.design_space = configs['design-space']
        self.iteration = configs['iteration']
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
            vec.append(int(_dict[GP.FEATURES[i]]))
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

        self.idx = []
        self.next = []
        # parallel 4 `vlsi_flow`
        for i in range(4):
            _sample()

        assert len(self.idx) == self.next, \
            "[ERROR]: assert failed. " \
            "idx: {}, next: {}".format(len(self.idx), len(self.next))

    def query(self):
        def _construct_kwargs(idx):
            return {
                "dims": self.dims,
                "size": self.size,
                "idx": idx,
                "logger": self.logger
            }

        res = []
        with Pool(processes=4) as pool:
            for _idx in range(len(self.idx)):
                kwargs = _construct_kwargs(self.idx[_idx])
                res.append(pool.apply_async(func=vlsi_flow, args=(self.next[0], **kwargs,)))
        # latency, power & area
        for _idx in range(len(res)):
            self.optimizer.register(
                params=self.next[_idx],
                target=-self.single_objective_cost_function(
                    res[_idx].get()["latency"],
                    res[_idx].get()["power"]
                )
            )

    def record(self):
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
        self.idx = knob2point(
            self.features2knob(
                self.get_features(
                    self.optimum
                )
            ),
            self.dims
        )
        self.optimizer.savegp(self.model_output_path)
        kwargs = {
            'dims': self.dims,
            'size': self.size,
            'idx': self.idx,
            'logger': self.logger
        }
        # latency, power & area
        self.metrics = vlsi_flow(self.get_features(self.optimum), **kwargs)

        self.logger.info("idx: %s metrics: %s" % (self.idx, self.metrics))

    def single_objective_cost_function(self, latency, power):

        return 1e-7 * latency + power

def get_feature_from_csv():

    return read_csv('data/micro-features.csv').T

def get_latency_from_csv():

    return read_csv('data/latency.csv')

def get_power_from_csv():

    return read_csv('data/power.csv')

def get_data_from_csv():
    features = get_feature_from_csv()
    latency = get_latency_from_csv()
    power = get_power_from_csv()

    return {
        "features": features,
        "latency": latency,
        "power": power
    }

def get_latency_dataset(data):
    train_latency_data = []
    test_latency_data = []

    for config in configs['config-name']:
        _latency = 0
        cnt = 0
        for item in data:
            if config in item[0]:
                for bmark in configs['benchmark-name']:
                    if bmark in item[0]:
                        if not np.isnan(item[1]):
                            _latency += item[1]
                            cnt += 1
        _latency /= cnt
        if configs['config-name'].index(config) < 12:
            train_latency_data.append(_latency)
        else:
            test_latency_data.append(_latency)

    assert (len(train_latency_data) + len(test_latency_data) == len(configs['config-name'])), \
        "[ERROR]: assert failed. " \
        "train_latency_data: {}, test_latency_data: {}, configs: {}".format(len(train_latency_data),
            len(test_latency_data), len(configs['config-name']))

    return {
        "train": train_latency_data,
        "test": test_latency_data
    }

def get_power_dataset(data):
    train_power_data = []
    test_power_data = []

    for config in configs['config-name']:
        _power = 0
        cnt = 0
        for item in data:
            if config in item[0]:
                for bmark in configs['benchmark-name']:
                    if bmark in item[0]:
                        _power += item[-1]
                        cnt += 1
        _power /= cnt
        if configs['config-name'].index(config) < 12:
            train_power_data.append(_power)
        else:
            test_power_data.append(_power)

    assert (len(train_power_data) + len(test_power_data) == len(configs['config-name'])), \
        "[ERROR]: assert failed. " \
        "train_power_data: {}, test_power_data: {}, configs: {}".format(len(train_power_data),
            len(test_power_data), len(configs['config-name']))

    return {
        "train": train_power_data,
        "test": test_power_data
    }

def get_feature_dataset(data):
    train_features_data = []
    test_features_data = []

    for config in configs['config-name']:
        _features = []
        for item in data:
            if config in item[0]:
                for i in item[1:]:
                    _features.append(int(i))
        if configs['config-name'].index(config) < 12:
            train_features_data.append(_features)
        else:
            test_features_data.append(_features)

    assert (len(train_features_data) + len(test_features_data) == len(configs['config-name'])), \
        "[ERROR]: assert failed. " \
        "train_features_data: {}, test_features_data: {}, configs: {}".format(len(train_features_data),
            len(test_features_data), len(configs['config-name']))

    return {
        "train": np.array(train_features_data),
        "test": np.array(test_features_data)
    }

def combine_target(latency, power):
    train_data, test_data = [], []
    for i in range(len(latency['train'])):
        train_data.append([latency['train'][i], power['train'][i]])
    for i in range(len(latency['test'])):
        test_data.append([latency['test'][i], power['test'][i]])

    return {
        "train": np.array(train_data),
        "test": np.array(test_data)
    }

def split_dataset(data):
    # train: 1-13 & test: 14-15
    latency = get_latency_dataset(data["latency"])
    power = get_power_dataset(data["power"])
    features = get_feature_dataset(data['features'])

    target = combine_target(latency, power)

    return {
        "features": features,
        "target": target
    }

def pareto_model(data):
    def build_xgb_regrssor():
        # return MultiOutputRegressor(
        #     XGBRegressor(
        #         reg_alpha=3,
        #         reg_lambda=2,
        #         gamma=0,
        #         min_child_weight=1,
        #         colsample_bytree=1,
        #         learning_rate=0.02,
        #         max_depth=4,
        #         n_estimators=10000,
        #         subsample=0.1
        #     )
        # )
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100
            )
        )

    model = build_xgb_regrssor()
    model.fit(
        data['features']['train'],
        data['target']['train'],
    )

    # test
    pred = model.predict(data['features']['test'])
    MSE_latency = metrics.mean_squared_error(pred[:, 0], data['target']['test'][:, 0])
    MSE_power = metrics.mean_squared_error(pred[:, 1], data['target']['test'][:, 1])
    MAPE_latency = calc_mape(pred[:, 0], data['target']['test'][:, 0])
    MAPE_power = calc_mape(pred[:, 1], data['target']['test'][:, 1])

    # save
    self.optimizer.savegp(self.model_output_path)

    print("[INFO]: MSE of latency: %.8f, MSE of power: %.8f" % (MSE_latency, MSE_power),
          "MAPE of latency: %.8f, MAPE of power: %.8f" % (MAPE_latency, MAPE_power))

def extract_features(data):
    features = []

    for config in configs['config-name']:
        _feature = []
        for item in data:
            if config in item[0]:
                for f in item[1:]:
                    _feature.append(int(f))
        features.append(_feature)

    assert (len(features) == len(configs['config-name'])), \
    "[ERROR]: assert failed. " \
    "features: {}, configs: {}".format(len(features), len(configs['config-name']))

    return features

def extract_latency(data):
    latency = []

    for config in configs['config-name']:
        _latency = 0
        cnt = 0
        for item in data:
            if config in item[0]:
                for bmark in configs['benchmark-name']:
                    if bmark in item[0]:
                        if not np.isnan(item[1]):
                            _latency += item[1]
                            cnt += 1
        _latency /= cnt
        latency.append(_latency)

    assert (len(latency) == len(configs['config-name'])), \
        "[ERROR]: assert failed. " \
        "latency: {}, configs: {}".format(len(latency), len(configs['config-name']))

    return latency

def extract_power(data):
    power = []

    for config in configs['config-name']:
        _power = 0
        cnt = 0
        for item in data:
            if config in item[0]:
                for bmark in configs['benchmark-name']:
                    if bmark in item[0]:
                        _power += item[-1]
                        cnt += 1
        _power /= cnt
        power.append(_power)

    assert (len(power) == len(configs['config-name'])), \
        "[ERROR]: assert failed. " \
        "power: {}, configs: {}".format(len(power), len(configs['config-name']))

    return power

def merge_data(features, latency, power):
    results = []
    for idx in range(len(features)):
        result = []
        # features
        for _idx in range(len(features[idx])):
            result.append(features[idx][_idx])
        # latency
        result.append(latency[idx])
        result.append(power[idx])
        results.append(result)

    return results

def extract_data(data):
    import pandas as pd

    features = extract_features(data["features"])
    latency = extract_latency(data["latency"])
    power = extract_power(data["power"])

    results = merge_data(features, latency, power)

    columns = [
        "fetchWidth",
        "decodeWidth",
        "numFetchBufferEntries",
        "numRobEntries",
        "numRasEntries",
        "numIntPhysRegisters",
        "numFpPhysRegisters",
        "numLdqEntries",
        "numStqEntries",
        "maxBrCount",
        "mem_issueWidth",
        "int_issueWidth",
        "fp_issueWidth",
        "DCacheParams_nWays",
        "DCacheParams_nMSHRs",
        "DCacheParams_nTLBEntries",
        "ICacheParams_nWays",
        "ICacheParams_nTLBEntries",
        "ICacheParams_fetchBytes",
        "latency",
        "power"
    ]
    writer = pd.DataFrame(columns=columns, data=results)
    writer.to_csv(configs['output-path'], index=False)

def handle():
    data = get_data_from_csv()

    if configs["xgb"]:
        data = split_dataset(data)

        pareto_model(data)
    else:
        # extract data ONLY
        extract_data(data)


if __name__ == "__main__":
    argv = parse_args()
    configs = get_config(argv)
    handle()
