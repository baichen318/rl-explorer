# Author: baichen318@gmail.com

import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics
import joblib
from util import parse_args, get_config, read_csv

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
        return MultiOutputRegressor(
            XGBRegressor(
                reg_alpha=3,
                reg_lambda=2,
                gamma=0,
                min_child_weight=1,
                colsample_bytree=1,
                learning_rate=0.02,
                max_depth=4,
                n_estimators=10000,
                subsample=0.1
            )
        )

    def save_model(model):
        print("saving the model: %s" % configs['output-path'])
        joblib.dump(model, configs['output-path'])

    model = build_xgb_regrssor()
    model.fit(
        data['features']['train'],
        data['target']['train'],
    )

    # test
    pred = model.predict(data['features']['test'])
    MSE_latency = metrics.mean_squared_error(pred[:, 0], data['target']['test'][:, 0])
    MSE_power = metrics.mean_squared_error(pred[:, 1], data['target']['test'][:, 1])

    # save
    save_model(model)

    print("MSE of latency: %.8f, MSE of power: %.8f" % (MSE_latency, MSE_power))

def handle():
    data = get_data_from_csv()

    data = split_dataset(data)

    pareto_model(data)

if __name__ == "__main__":
    argv = parse_args()
    configs = get_config(argv)
    handle()

