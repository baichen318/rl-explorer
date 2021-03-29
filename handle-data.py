# Author: baichen318@gmail.com

import os
import re
import pandas as pd
import numpy as np
from util import parse_args, get_config, if_exist, read_csv, write_csv
from exception import UnDefinedException

def handle_power_report(report, root, bmark):
    if if_exist(report):
        result = []
        result.append(root + '-' + bmark)
        with open(report, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line[0:9] == "boom_tile":
                    result += (line.split()[2:6])
        results.append(result)

def handle_latency_report(report, root, bmark):
    if if_exist(report):
        result = []
        result.append(root + '-' + bmark)
        with open(report, 'r') as f:
            res = f.readlines()[-1].split('after')
            if 'PASSED' in res[0]:
                res = re.findall(r"\d+\.?\d*", res[1].strip())[0]
                result.append(res)
        results.append(result)

def handle_area_report(report, root):
    if if_exist(report):
        result = []
        result.append(root)
        with open(report, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line[0:9] == "boom_tile":
                    result.append(line.split()[-1])
        results.append(result)

def handle_power():
    if if_exist(config['data-path']):
        for root in config['config-name']:
            for bmark in os.listdir(os.path.join(config['data-path'], root)):
                report = os.path.join(config['data-path'],
                    root,
                    bmark,
                    'reports',
                    'vcdplus.power.avg.max.report'
                )
                handle_power_report(report, root, bmark)
        power = ['Configure', 'Int Power', 'Switch Power', 'Leak Power', 'Total Power']
        writer = pd.DataFrame(columns=power, data=results)
        writer.to_csv(config['output-path'], index=False)

def handle_latency():
    if if_exist(config['data-path']):
        for root in config['config-name']:
            bmarks = os.path.join(config['data-path'], root, 'sim-syn-rundir', 'output')
            for bmark in os.listdir(bmarks):
                report = os.path.join(config['data-path'],
                    root,
                    'sim-syn-rundir',
                    'output',
                    bmark,
                    bmark + '.out'
                )
                handle_latency_report(report, root, bmark)
        latency = ['Configure', 'Cycles']
        writer = pd.DataFrame(columns=latency, data=results)
        writer.to_csv(config['output-path'], index=False)

def handle_area():
    if if_exist(config['data-path']):
        for root in config['config-name']:
            report = os.path.join(config['data-path'],
                root,
                'syn-rundir',
                'reports',
                'final_area.rpt')
            handle_area_report(report, root)
        area = ['Configure', 'Area']
        writer = pd.DataFrame(columns=area, data=results)
        writer.to_csv(config['output-path'], index=False)

def _handle_dataset(features, power, latency):
    FEATURES = [
        'fetchWidth',
        'decodeWidth',
        'numFetchBufferEntries',
        'numRobEntries',
        'numRasEntries',
        'numIntPhysRegisters',
        'numFpPhysRegisters',
        'numLdqEntries',
        'numStqEntries',
        'maxBrCount',
        'mem_issueWidth',
        'int_issueWidth',
        'fp_issueWidth',
        'DCacheParams_nWays',
        'DCacheParams_nMSHRs',
        'DCacheParams_nTLBEntries',
        'ICacheParams_nWays',
        'ICacheParams_nTLBEntries',
        'ICacheParams_fetchBytes',
        'latency',
        'power'
    ]

    data = []
    for idx in range(len(features)):
        _data = features[idx].strip().split('\t')
        idx += 1
        c_name = "Config%s" % str(idx)
        for l in latency:
            _l = l[0].split('-')[0].split('.')[-1].lstrip('BOOM').rstrip('Config')
            if (c_name == _l) and (l[0].split('-')[-1] == "whetstone.riscv"):
                assert (not np.isnan(l[-1])) and (l[-1] != 0)
                _data.append(l[-1])
        for p in power:
            _p = p[0].split('-')
            if (c_name == _p[0]) and (_p[-1] == "whetstone.riscv"):
                assert (not np.isnan(p[-1])) and (p[-1] != 0)
                _data.append(p[-1])
        data.append(_data)

    write_csv(config["output-path"], data, FEATURES)


def handle_dataset():
    if_exist(config["feature-path"], strict=True)
    if_exist(config["power-path"], strict=True)
    if_exist(config["latency-path"], strict=True)
    # if_exist('data/sample-area.csv', strict=True)

    power = read_csv('data/sample-power.csv')
    latency = read_csv('data/sample-latency.csv')

    with open('data/sample.txt', 'r') as f:
        features = f.readlines()

    _handle_dataset(features, power, latency)


def handle():
    print("Handling...")

    if config['mode'] == 'power':
        handle_power()
    elif config['mode'] == 'latency':
        handle_latency()
    elif config['mode'] == 'area':
        handle_area()
    elif config['mode'] == 'dataset':
        handle_dataset()
    else:
        raise UnDefinedException(config['mode'])

    print("Done.")

if __name__ == "__main__":
    argv = parse_args()
    config = get_config(argv)
    # define the global variable `results`
    results = []
    handle()

