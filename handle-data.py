# Author: baichen318@gmail.com

import os
import re
import pandas as pd
import numpy as np
from util import parse_args, get_configs, if_exist, read_csv, write_csv
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
    if if_exist(configs['pt-pwr-path']):
        prefix = "" if configs['initialize-method'] == "random" \
            else configs['initialize-method'].upper()
        for root in configs['config-name']:
            root = prefix + 'Config' + root.split('Config')[1] + '-benchmarks'
            for bmark in os.listdir(os.path.join(configs['pt-pwr-path'], root)):
                report = os.path.join(configs['pt-pwr-path'],
                    root,
                    bmark,
                    'reports',
                    'vcdplus.power.avg.max.report'
                )
                handle_power_report(report, root, bmark)
        power = ['Configure', 'Int Power', 'Switch Power', 'Leak Power', 'Total Power']
        writer = pd.DataFrame(columns=power, data=results)
        writer.to_csv(configs['power-output-path'], index=False)
        results.clear()

def handle_latency():
    if if_exist(configs['vlsi-build-path']):
        for root in configs['config-name']:
            bmarks = os.path.join(configs['vlsi-build-path'], root, 'sim-syn-rundir', 'output')
            for bmark in os.listdir(bmarks):
                report = os.path.join(configs['vlsi-build-path'],
                    root,
                    'sim-syn-rundir',
                    'output',
                    bmark,
                    bmark + '.out'
                )
                handle_latency_report(report, root, bmark)
        latency = ['Configure', 'Cycles']
        writer = pd.DataFrame(columns=latency, data=results)
        writer.to_csv(configs['latency-output-path'], index=False)
        results.clear()

def handle_area():
    if if_exist(configs['vlsi-build-path']):
        for root in configs['config-name']:
            report = os.path.join(configs['vlsi-build-path'],
                root,
                'syn-rundir',
                'reports',
                'final_area.rpt')
            handle_area_report(report, root)
        area = ['Configure', 'Area']
        writer = pd.DataFrame(columns=area, data=results)
        writer.to_csv(configs['area-output-path'], index=False)
        results.clear()

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
    for i in range(len(features)):
        _data = features[i].strip().split('\t')
        idx = i + configs['idx']
        prefix = "" if configs['initialize-method'] == "random" \
            else configs['initialize-method'].upper()
        prefix = prefix if configs["flow"] == "initialize" \
            else configs["model"]
        c_name = "%sConfig%s" % (prefix, str(idx))
        for l in latency:
            _l = l[0].split('-')[0].split('.')[-1].lstrip('BOOM').rstrip('Config')
            if (c_name == _l) and (l[0].split('-')[-1] == configs['benchmark']):
                assert (not np.isnan(l[-1])) and (l[-1] != 0)
                # insert latency
                _data.append(l[-1])
        for p in power:
            _p = p[0].split('-')
            if (c_name == _p[0]) and (_p[-1] == configs['benchmark']):
                assert (not np.isnan(p[-1])) and (p[-1] != 0)
                # insert power
                _data.append(p[-1])
        data.append(_data)
    write_csv(configs["dataset-output-path"], data, col_name=FEATURES)

def handle_dataset():
    with open(configs['initialize-output-path'], 'r') as f:
        features = f.readlines()
    power = read_csv(configs['power-output-path'])
    latency = read_csv(configs['latency-output-path'])
    # area = read_csv(configs['area-output-path'])
    _handle_dataset(features, power, latency)

def handle():
    print("Handling...")

    handle_power()
    handle_latency()
    handle_area()
    handle_dataset()

    print("Done.")

if __name__ == "__main__":
    argv = parse_args()
    configs = get_configs(argv.configs)
    # define the global variable `results`
    results = []
    handle()

