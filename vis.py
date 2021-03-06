import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from util import parse_args, get_config, if_exist, read_csv

def handle_latency(latency, bmark, name):
    for item in latency:
        if name in item[0] and bmark in item[0]:
            if not np.isnan(item[1]):
                latency_points.append(item[1])
            else:
                latency_points.append(0)

def handle_power(power, bmark, name):
    for item in power:
        if name in item[0] and bmark in item[0]:
            power_points.append(item[-1])

def plot(bmark):
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(1)
    # 1st figure
    plt.subplot(2, 1, 1)
    plt.plot(latency_points, power_points, color='r', marker='o', linestyle='dashed')
    # plt.xlabel('Latency')
    plt.ylabel('Power')
    plt.title('Latency vs. Power' + '(' + bmark + ')' + ' scatter')
    # plt.text(latency_points, power_points, config['config-name'])
    plt.grid()

    # 2nd figure
    plt.subplot(2, 1, 2)
    np_latency_points = np.array(latency_points)
    np_power_points = np.array(power_points)
    np_power_points = np_power_points[np.argsort(np_latency_points)]
    np_latency_points = np.sort(np_latency_points)
    fq = interp1d(np_latency_points, np_power_points, kind='quadratic')
    np_latency_points_new = np.linspace(
        np_latency_points.min(),
        np_latency_points.max(),
        1000
    )
    np_power_points_new = fq(np_latency_points_new)
    plt.plot(np_latency_points_new, np_power_points_new, color='r', marker='o', linestyle='dashed')
    plt.xlabel('Latency')
    plt.ylabel('Power')
    plt.title('Latency vs. Power' + '(' + bmark + ')' + ' curve')
    plt.grid()

    # save
    output = os.path.join(config['output-path'], bmark + '.jpg')
    print("[INFO]: save the figure:", output)
    plt.savefig(output)

def handle_vis(latency, power):
    for bmark in config['benchmark-name']:
        for name in config['config-name']:
            handle_latency(latency, bmark, name)
            handle_power(power, bmark, name)

        assert len(latency_points) == len(power_points), "[ERROR]: assert error. " \
            "latency_points: {}, power_points: {}".format(len(latency_points),
                len(power_points))

        print("[INFO]: points for", bmark, ": ", len(power_points))

        plot(bmark)

        latency_points.clear()
        power_points.clear()

def handle():
    latency = read_csv('data/latency.csv')
    power = read_csv('data/power.csv')
    handle_vis(latency, power)

if __name__ == "__main__":
    argv = parse_args()
    config = get_config(argv)
    # define global variables
    latency_points = []
    power_points = []
    handle()
