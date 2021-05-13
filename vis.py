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

def plot_v1(data, title, kwargs):
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    cnt = 0
    print("[INFO]: data points: ", len(data))

    for i in range(1, 6):
        plt.scatter(data[str(i) + "-x"], data[str(i) + "-y"],
            s=2,
            marker=markers[2],
            c=colors[i + 1],
            label="decodeWidth = %s" % str(i)
        )
    if "data" in kwargs.keys() and "baseline_config_name" in kwargs.keys():
        i = 0
        h = []
        for d in kwargs["data"]:
            h.append(
                plt.scatter(
                    d[0],
                    d[1],
                    s=15,
                    marker=markers[-2],
                    c=colors[i + 2],
                    label=kwargs["baseline_config_name"][i]
                )
            )
            i += 1
        plt.legend(handles=h, labels=kwargs["baseline_config_name"], loc='best', ncol=1, frameon=False)
    elif "data" in kwargs.keys():
        for d in kwargs["data"]:
            plt.scatter(d[0], d[1], s=2, marker=markers[-2])
    plt.xlabel('C.C.')
    plt.ylabel('Power')
    plt.title('C.C. vs. Power -- ' + title)
    # plt.grid()
    output = os.path.join(
        kwargs["configs"]["fig-output-path"]
    )
    print("[INFO]: save the figure", output)
    plt.savefig(output)

def handle_v1():
    """
        API for visualization: baseline + design space
    """
    dataset, title = read_csv_v2(configs["dataset-output-path"])
    dataset = validate(dataset)
    _data = {
        "1-x": np.array([]),
        "1-y": np.array([]),
        "2-x": np.array([]),
        "2-y": np.array([]),
        "3-x": np.array([]),
        "3-y": np.array([]),
        "4-x": np.array([]),
        "4-y": np.array([]),
        "5-x": np.array([]),
        "5-y": np.array([])
    }
    for data in dataset:
        _data[str(round(data[1])) + "-x"] = np.insert(_data[str(round(data[1])) + "-x"], len(_data[str(round(data[1])) + "-x"]), data[-2])
        _data[str(round(data[1])) + "-y"] = np.insert(_data[str(round(data[1])) + "-y"], len(_data[str(round(data[1])) + "-y"]), data[-1])
    plot_v1(
        _data,
        "Design Space",
        {
            "data": reference,
            "baseline_config_name": [
                "SmallBoomConfig",
                "MediumBoomConfig",
                "LargeBoomConfig",
                "MegaBoomConfig",
                "GigaBoomConfig"
            ],
            "configs": configs,
        }
    )

def plot_v2(data, title, kwargs):
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    cnt = 0
    print("[INFO]: data points: ", len(data))

    for d in data:
        plt.scatter(
            d[0],
            d[1],
            s=2,
            marker=markers[2],
        )
    if "data" in kwargs.keys() and "baseline_config_name" in kwargs.keys():
        i = 0
        h = []
        for d in kwargs["data"]:
            h.append(
                plt.scatter(
                    d[0],
                    d[1],
                    s=15,
                    marker=markers[-2],
                    c=colors[i + 2],
                    label=kwargs["baseline_config_name"][i]
                )
            )
            i += 1
        plt.legend(handles=h, labels=kwargs["baseline_config_name"], loc='best', ncol=1, frameon=False)
    elif "data" in kwargs.keys():
        for d in kwargs["data"]:
            plt.scatter(d[0], d[1], s=2, marker=markers[-2])
    plt.xlabel('C.C.')
    plt.ylabel('Power')
    plt.title('C.C. vs. Power -- ' + title)
    # plt.grid()
    output = os.path.join(
        kwargs["configs"]["fig-output-path"]
    )
    print("[INFO]: save the figure", output)
    plt.savefig(output)

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
    plot_v2(
        data,
        title,
        {
            "data": _data,
            "configs": configs
        }
    )

def plot_v3(data):
    from handle_data import reference, baseline

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    cnt = 0
    print("[INFO]: data points: ", len(data))

    for d in data:
        plt.scatter(
            d[0],
            d[1],
            s=2,
            marker=markers[2],
        )
    h = []
    i = 0
    for b in reference:
        h.append(
            plt.scatter(
                b[0],
                b[1],
                s=15,
                marker=markers[-2],
                c=colors[i],
                label=baseline[i] + "BoomConfig"
            )
        )
        i += 1
    plt.legend(handles=h, labels=[b + "BoomConfig" for b in baseline], loc='best', frameon=False)
    plt.xlabel('C.C.')
    plt.ylabel('Power')
    plt.title('C.C. vs. Power -- ' + configs["initialize-method"])
    # plt.grid()
    output = os.path.join(
        configs["fig-output-path"]
    )
    print("[INFO]: save the figure", output)
    plt.savefig(output)

def handle_v3():
    """
        API for verification between prediction & baseline
    """
    dataset, _ = read_csv_v2(configs["dataset-output-path"])
    dataset = validate(dataset)
    _data = []
    for d in dataset:
        if isinstance(d[-2], float) and isinstance(d[-1], float):
            _data.append((d[-2], d[-1]))

    plot_v3(_data)

if __name__ == "__main__":
    # TODO: verify plot
    # plot original design space
    argv = parse_args()
    configs = get_configs(argv.configs)
    # handle_v1()
    handle_v3()

