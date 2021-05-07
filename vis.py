# Author: baichen318@gmail.com

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from util import parse_args, get_configs, if_exist, read_csv, read_csv_v2
from handle_data import reference
from exception import UnDefinedException

markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]
colors = [
    'c', 'b', 'g', 'r', 'm', 'y', 'k', 'w'
]

def validate(dataset):
    """
        `dataset`: <tuple>
    """
    data = []
    for item in dataset:
        _data = []
        f = item[0].split(' ')
        for i in f:
            _data.append(int(i))
        for i in item[1:]:
            _data.append(float(i))
        data.append(_data)
    data = np.array(data)

    return data

def plot(data, title, kwargs):
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    cnt = 0
    print("[INFO]: data points: ", len(data))
    for d in data:
        plt.scatter(d[0], d[1], s=1, marker=markers[2])
    # if "data" in kwargs.keys() and "baseline_config_name" in kwargs.keys():
    #     i = 0
    #     h = []
    #     for d in kwargs["data"]:
    #         h.append(
    #             plt.scatter(
    #                 d[0],
    #                 d[1],
    #                 s=15,
    #                 marker=markers[-2],
    #                 label=kwargs["baseline_config_name"][i]
    #             )
    #         )
    #         i += 1
    #     plt.legend(handles=h, labels=kwargs["baseline_config_name"], loc='best', ncol=1)
    if "data" in kwargs.keys():
        for d in kwargs["data"]:
            plt.scatter(d[0], d[1], s=2, marker=markers[-2])
    plt.xlabel('Latency (CPI)')
    plt.ylabel('Power')
    plt.title('Latency (CPI) vs. Power ' + title)
    # plt.grid()
    output = os.path.join(
        kwargs["configs"]["fig-output-path"],
        title  + '.jpg'
    )
    print("[INFO]: save the figure", output)
    plt.savefig(output)

def handle_v1():
    """
        API for visualization: baseline + design space
    """
    dataset, title = read_csv_v2(configs["dataset-output-path"])
    dataset = validate(dataset)
    _data = []
    for data in dataset:
        if isinstance(data[-2], float) and isinstance(data[-1], float):
            _data.append((data[-2], data[-1]))
    plot(
        _data,
        "Design-Space",
        {
            "data": reference,
            "baseline_config_name": [
                "SmallBoomConfig"
                "MediumBoomConfig",
                "LargeBoomConfig",
                "MegaBoomConfig",
                "GigaBoomConfig",
            ],
            "configs": configs
        }
    )

def handle_v2(data, title, configs):
    """
        API for visulization: design space + prediction
        data: <list> with <tuple> as elements
    """
    dataset, _ = read_csv_v2(configs["dataset-output-path"])
    dataset = validate(dataset)
    _data = []
    for d in dataset:
        if isinstance(d[-2], float) and isinstance(d[-1], float):
            _data.append((d[-2], d[-1]))
    plot(
        data,
        title,
        {
            "data": _data,
            "configs": configs
        }
    )
if __name__ == "__main__":
    # TODO: verify plot
    # plot original design space
    argv = parse_args()
    configs = get_configs(argv.configs)
    handle_v1()
