# Author: baichen318@gmail.com

import numpy as np
from util import read_csv

# bmarks = [
#     'dhrystone.riscv',
#     'median.riscv',
#     'mm.riscv',
#     'mt-matmul.riscv',
#     'mt-vvadd.riscv',
#     'multiply.riscv',
#     'pmp.riscv',
#     'qsort.riscv',
#     'rsort.riscv',
#     'spmv.riscv',
#     'towers.riscv'
# ]

bmarks = [
    "fft.riscv",
    "sqrt.riscv",
    "whetstone.riscv",
    "hello_world_large.riscv"
]

configs = [
  'GigaBoomProConfig',
  'GigaBoomConfig',
  'GigaBoomSEConfig',
  'MegaBoomProConfig',
  'MegaBoomConfig',
  'MegaBoomSEConfig',
  'LargeBoomProConfig',
  'LargeBoomConfig',
  'LargeBoomSEConfig',
  'MediumBoomProConfig',
  'MediumBoomConfig',
  'MediumBoomSEConfig',
  'SmallBoomProConfig',
  'SmallBoomConfig',
  'SmallBoomSEConfig'
]

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

    for config in configs:
        _latency = 0
        cnt = 0
        for item in data:
            if config in item[0]:
                for bmark in bmarks:
                    if bmark in item[0]:
                        _latency += item[1]
                        cnt += 1
        _latency /= cnt
        if configs.index(config) < 13:
            train_latency_data.append(_latency)
        else:
            test_latency_data.append(_latency)

    assert (len(train_latency_data) + len(test_latency_data) == len(configs)), \
        "[ERROR]: assert failed." \
        "train_latency_data: {}, test_latency_data: {}, configs: {}".format(len(train_latency_data),
            len(test_latency_data), len(configs))

    return {
        "train": train_latency_data,
        "test": test_latency_data
    }

def get_power_dataset(data):
    train_power_data = []
    test_power_data = []

    for config in configs:
        _power = 0
        cnt = 0
        for item in data:
            if config in item[0]:
                for bmark in bmarks:
                    if bmark in item[0]:
                        _power += item[-1]
                        cnt += 1
        _power /= cnt
        if configs.index(config) < 13:
            train_power_data.append(_power)
        else:
            test_power_data.append(_power)

    assert (len(train_power_data) + len(test_power_data) == len(configs)), \
        "[ERROR]: assert failed." \
        "train_power_data: {}, test_power_data: {}, configs: {}".format(len(train_power_data),
            len(test_power_data), len(configs))

    return {
        "train": train_power_data,
        "test": test_power_data
    }

def get_feature_dataset(data):
    train_features_data = []
    test_features_data = []

    for config in configs:
        _features = []
        for item in data:
            if config in item[0]:
                for i in item[1:]:
                    _features.append(int(i))
        if configs.index(config) < 13:
            train_features_data.append(_features)
        else:
            test_features_data.append(_features)

    assert (len(train_features_data) + len(test_features_data) == len(configs)), \
        "[ERROR]: assert failed." \
        "train_features_data: {}, test_features_data: {}, configs: {}".format(len(train_features_data),
            len(test_features_data), len(configs))

    return {
        "train": train_features_data,
        "test": test_features_data
    }

def split_dataset(data):
    # train: 1-13 & test: 14-15
    latency = get_latency_dataset(data["latency"])
    power = get_power_dataset(data["power"])
    features = get_feature_dataset(data['features'])

    # normalize

def handle():
    data = get_data_from_csv()

    split_dataset(data)

if __name__ == "__main__":
    handle()

