# Author: baichen318@gmail.com

import numpy as np
from xgboost import XGBRegressor
from sklearn import metrics
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

def normalize(latency, power):

    def _normalize(arr, min_val, max_val):
        ret = []
        const = max_val - min_val
        for val in arr:
            val = (val - min_val) / const
            ret.append(val)

        return ret

    ext_latency = {
        "max_train": np.max(np.array(latency['train'])),
        "min_train": np.min(np.array(latency['train'])),
        "max_test": np.max(np.array(latency['test'])),
        "min_test": np.min(np.array(latency['test']))
    }

    ext_power = {
        "max_train": np.max(np.array(power['train'])),
        "min_train": np.min(np.array(power['train'])),
        "max_test": np.max(np.array(power['test'])),
        "min_test": np.min(np.array(power['test']))
    }

    return {
        "latency": {
            "train": _normalize(
                latency['train'],
                ext_latency['min_train'],
                ext_latency['max_train']
            ),
            "test": _normalize(
                latency['test'],
                ext_latency['min_test'],
                ext_latency['max_test']
            )
        },
        "power": {
            "train": _normalize(
                power['train'],
                ext_power['min_train'],
                ext_power['max_train']
            ),
            "test": _normalize(
                power['test'],
                ext_power['min_test'],
                ext_power['max_test']
            )
        }
    }

def calc_perf(data):
    # 0.5 * latency + 0.5 * power
    train_data = []
    test_data = []
    for i in range(len(data['latency']['train'])):
        train_data.append(
            0.5 * data['latency']['train'][i] + 0.5 * data['power']['train'][i]
        )
    for i in range(len(data['latency']['test'])):
        test_data.append(
            0.5 * data['latency']['test'][i] + 0.5 * data['power']['test'][i]
        )

    return {
        "train": np.array(train_data),
        "test": np.array(test_data)
    }

def split_dataset(data):
    # train: 1-13 & test: 14-15
    latency = get_latency_dataset(data["latency"])
    power = get_power_dataset(data["power"])
    features = get_feature_dataset(data['features'])

    perf = calc_perf(normalize(latency, power))
    
    return {
        "features": features,
        "perf": perf
    }

def pareto_model(data):
    def build_xgb_regrssor():
        return XGBRegressor(
            reg_alpha=3,
            reg_lambda=2,
            gamma=0,
            min_child_weight=1,
            colsample_bytree=1,
            learning_rate=0.02,
            max_depth=4,
            n_estimators=30,
            subsample=0.1
        )

    def save_model(model):
        print("saving the model: %s" % configs['output-path'])
        model.get_booster().save_model(configs['output-path'])

    model = build_xgb_regrssor()
    model.fit(
        data['features']['train'],
        data['perf']['train'],
        eval_set=[(data['features']['test'], data['perf']['test'])]
    )

    # test
    pred = model.predict(data['features']['test'])
    MSE = metrics.mean_squared_error(pred, data['perf']['test'])
    R2 = metrics.r2_score(pred, data['perf']['test'])

    # save
    save_model(model)

    print("MSE: %.8f, R2: %.8f" % (MSE, R2))

def handle():
    data = get_data_from_csv()

    data = split_dataset(data)

    pareto_model(data)

if __name__ == "__main__":
    argv = parse_args()
    configs = get_config(argv)
    handle()

