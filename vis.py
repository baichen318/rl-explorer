import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from util import parse_args, get_config, if_exist, read_csv

markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]

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

def plot_v2(data, c_name):
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300

    i = 0
    h = []
    for d in data:
        for x, y in zip(*d):
            h.append(plt.scatter(x, y, s=1, label=c_name[i], marker=markers[2 % len(markers)]))
        i += 1
    # plt.legend(handles=h, labels=c_name, loc='best', ncol=1)
    plt.xlabel('Latency')
    plt.ylabel('Power')
    plt.title('Latency vs. Power (whetstone)')
    plt.grid()
    output = os.path.join(config['output-path'], 'whetstone.jpg')
    print('[INFO]: save the figure', output)
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

def extract_configs(latency, power):
    configs = set()
    for l in latency:
        configs.add(l[0].split('-')[0].split('.')[-1].lstrip('BOOM').rstrip('Config'))
    for p in power:
        configs.add(p[0].split('-')[0])

    return configs

def save_mat(data):
    print('[INFO]: saving...')
    with open('data/data.mat', 'w') as f:
        f.write(str(data))
        f.write('\n')

def handle_vis_v2(latency, power):
    configs = extract_configs(latency, power)

    ret = []
    c_name = []
    for c in configs:
        c_name.append(c)
        _cx = []
        _cy = []
        for l in latency:
            _l = l[0].split('-')
            if c == _l[0].split('.')[-1].lstrip('BOOM').rstrip('Config'):
                for p in power:
                    _p = p[0].split('-')
                    if (_l[0].split('.')[-1].lstrip('BOOM').rstrip('Config') == _p[0]) and \
                        (_l[-1] == _p[-1]) and (_l[-1] == 'whetstone.riscv'):
                        if np.isnan(l[-1]) or np.isnan(p[-1]) or (l[-1] == 0) or (p[-1] == 0):
                            continue
                        _cx.append(l[-1])
                        _cy.append(p[-1])
        assert len(_cx) == len(_cy), "[ERROR]: assert error. " \
            "_cx: {}, _cy: {}".format(len(_cx), len(_cy))
        ret.append((_cx, _cy))

    # save_mat(ret)
    print("[INFO]: total points: %d" % len(ret))

    plot_v2(ret, c_name)

def handle():
    latency = read_csv('data/sample-latency.csv')
    power = read_csv('data/sample-power.csv')
    # handle_vis(latency, power)
    handle_vis_v2(latency, power)

if __name__ == "__main__":
    argv = parse_args()
    config = get_config(argv)
    # define global variables
    latency_points = []
    power_points = []
    handle()
