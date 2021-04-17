# Author: baichen318@gmail.com

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from util import parse_args, get_config, if_exist, read_csv
from exception import UnDefinedException

markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]
colors = [
    'c', 'b', 'g', 'r', 'm', 'y', 'k', 'w'
]

def plot(data, title, **kwargs):
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    cnt = 0
    for d in data:
        plt.scatter(d[0], d[1], s=1, marker=markers[2])
    if kwargs is not None:
        i = 0
        h = []
        for d in kwargs["data"]:
            h.append(
                plt.scatter(
                    d[0],
                    d[1],
                    s=15,
                    marker=markers[-2],
                    label=kwargs["baseline_config_name"][i]
                )
            )
            i += 1
        plt.legend(handles=h, labels=kwargs["baseline_config_name"], loc='best', ncol=1)
    plt.xlabel('Latency')
    plt.ylabel('Power')
    plt.title('Latency vs. Power (' + title + '@' + '%s)' % kwargs["configs"]["benchmark"])
    plt.grid()
    output = os.path.join(
        kwargs["configs"]["fig-output-path"],
        title + '-' + '%s-predict.jpg' % kwargs["configs"]["benchmark"]
    )
    print("[INFO]: save the figure", output)
    plt.savefig(output)

def handle_vis(data, title, configs):
    """
        API for visualize: baseline + prediction
        data: <list> (<np.array> in <list>)
    """
    latency = read_csv("data/baseline-latency.csv")
    power = read_csv("data/baseline-power.csv")

    baseline_config_name = [
        "GigaBoomConfig",
        "MegaBoomConfig",
        "LargeBoomConfig",
        "MediumBoomConfig",
        "SmallBoomConfig"
    ]
    baseline = []
    for c in baseline_config_name:
        for l in latency:
            _l1 = l[0].split('.')[2].split('-')[0]
            _l2 = l[0].split('-')[-1]
            if c == _l1 and _l2 == configs["benchmark"]:
                for p in power:
                    _p1 = p[0].split('-')[0]
                    _p2 = p[0].split('-')[-1]
                    if c == _p1 and _p2 == configs["benchmark"]:
                        baseline.append([l[-1], p[-1]])
    baseline = np.array(baseline)

    plot(
        data,
        title,
        {
            "data": baseline,
            "baseline_config_name": baseline_config_name,
            "configs": configs
        }
    )

if __name__ == "__main__":
    # TODO: verify plot
    pass
