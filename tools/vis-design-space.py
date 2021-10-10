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


markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]
colors = [
    'c', 'b', 'g', 'r', 'm', 'y', 'k', # 'w'
]


def vis_design_space_v1():
    # this is for 3D views
    dataset = load_txt(os.path.join(os.path.pardir, configs["dataset-output-path"]), fmt=float)
    remove_idx = []
    perf = dataset[:, -3]
    power = dataset[:, -2]
    area = dataset[:, -1]
    # remove invalid data
    for i in range(perf.shape[0]):
        if power[i] == 0:
            remove_idx.append(i)
            continue
        if area[i] == 0:
            remove_idx.append(i)
            continue
    perf = np.delete(perf, remove_idx)
    power = np.delete(power, remove_idx)
    area = np.delete(area, remove_idx)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(perf, power, area, s=3, marker=markers[-10])
    ax.set_title("%s PPA" % configs["design"], pad=15, fontsize="10")
    ax.set_xlabel("IPC")
    ax.set_ylabel("Power")
    ax.set_zlabel("Area")
    ax.view_init(elev=22, azim=33)
    path = os.path.join(os.path.dirname(__file__), "vis-%s.png" % configs["design"])
    plt.savefig(
        path,
        dpi=600
    )
    print("[INFO]: save %s" % path)
    plt.show()


def vis_design_space_v2():
    # this is for power-performance views
    dataset = load_txt(os.path.join(os.path.pardir, configs["dataset-output-path"]), fmt=float)
    remove_idx = []
    perf = dataset[:, -3]
    power = dataset[:, -2]
    # remove invalid data
    for i in range(perf.shape[0]):
        if power[i] == 0:
            remove_idx.append(i)
            continue
    perf = np.delete(perf, remove_idx)
    power = np.delete(power, remove_idx)
    plt.scatter(power, perf, s=2, marker=markers[-10], c=colors[1])
    plt.xlabel("Power")
    plt.ylabel("IPC")
    plt.title("power-performance %s" % configs["design"])
    path = os.path.join(os.path.dirname(__file__), "power-performance-%s.png" % configs["design"])
    plt.savefig(
        path,
        dpi=600
    )
    print("[INFO]: save %s" % path)
    plt.show()


def vis_design_space_v3():
    # this is for area-performance views
    dataset = load_txt(os.path.join(os.path.pardir, configs["dataset-output-path"]), fmt=float)
    remove_idx = []
    perf = dataset[:, -3]
    area = dataset[:, -1]
    # remove invalid data
    for i in range(perf.shape[0]):
        if area[i] == 0:
            remove_idx.append(i)
            continue
    perf = np.delete(perf, remove_idx)
    power = np.delete(area, remove_idx)
    plt.scatter(area, perf, s=2, marker=markers[-10], c=colors[1])
    plt.xlabel("Area")
    plt.ylabel("IPC")
    plt.title("area-performance %s" % configs["design"])
    path = os.path.join(os.path.dirname(__file__), "area-performance-%s.png" % configs["design"])
    plt.savefig(
        path,
        dpi=600
    )
    print("[INFO]: save %s" % path)
    plt.show()


if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    vis_design_space_v3()
