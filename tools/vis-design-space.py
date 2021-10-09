# Author: baichen318@gmail.com

import sys, os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "dse")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "util")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "vlsi")
)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import load_txt, get_configs, parse_args

def vis_design_space_v1():
    dataset = load_txt(os.path.join(os.path.pardir, configs["dataset-output-path"]), fmt=float)
    remove_idx = []
    perf = dataset[:, -3]
    power = dataset[:, -2]
    area = dataset[:, -1]
    # remove invalid data
    for i in range(perf.shape[0]):
        if perf[i] == 0:
            remove_idx.append(i)
            continue
        if power[i] == 0:
            remove_idx.append(i)
            continue
    perf = np.delete(perf, remove_idx)
    power = np.delete(power, remove_idx)
    area = np.delete(area, remove_idx)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(perf, power, area)
    path = os.path.join(os.path.dirname(__file__), "vis-%s.png" % configs["design"])
    plt.savefig(
        path,
        dpi=600
    )
    print("[INFO]: save %s" % path)
    plt.show()


def vis_design_space_v2():
    # this is for detail dataset
    pass


if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    vis_design_space_v1()
