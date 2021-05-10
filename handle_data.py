# Author: baichen318@gmail.com

import os
import re
import pandas as pd
import numpy as np
from util import parse_args, get_configs, if_exist, read_csv, write_csv, write_txt
from exception import UnDefinedException

# NOTICE: If you extract baseline data-set
# modify `benchmarks` with `mt-vvadd.riscv-2G` and `whetstone.riscv-2G`
benchmarks = {
    # "median.riscv": 2455,
    # "mt-matmul.riscv": 2508,
    "mt-vvadd.riscv": 12454,
    "whetstone.riscv": 1184
    # "fft.riscv": 1159,
    # "h264_dec.riscv": 1171
}

baseline = [
    "Small",
    "Medium",
    "Large",
    "Mega",
    "Giga"
]

reference = [
    # Small
    np.array([1.3790802167305407, 4.63e-02]),
    # Medium
    np.array([1.3690366063169026, 5.83e-02]),
    # Large
    np.array([1.3608431346636713, 9.45e-02]),
    # Mega
    np.array([1.352054975551738, 0.136]),
    # Giga
    np.array([1.351658517245936, 0.133])
]

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
                # NOTICE: CPI is aften used as an evaluation index
                # We use c.c. here
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
            for base in baseline:
                if base in root:
                    base = True
                    break
                else:
                    base = False
            if base:
                root = root.split('.')[-1].split('-')[0] + '-benchmarks'
            else:
                root = prefix + 'Config' + root.split('Config')[1] + '-benchmarks'
            for bmark in benchmarks.keys():
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
            for bmark in benchmarks.keys():
                report = os.path.join(configs['vlsi-build-path'],
                    root,
                    'sim-syn-rundir',
                    'output',
                    bmark.strip('-2G'),
                    bmark.strip('-2G') + '.out'
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

def normalize(power, latency):
    # Get max & min from power and latency
    extl = []
    for bmark in benchmarks.keys():
        _data = []
        for l in latency:
            if l[0].split('ChipTop-')[-1] == bmark:
                if np.isnan(l[-1]) or l[-1] == 0:
                    continue
                _data.append(l[-1])
        extl.append((np.min(_data), np.max(_data)))

    extp = []
    for bmark in benchmarks.keys():
        _data = []
        for p in power:
            if p[0].split('benchmarks-')[-1] == bmark:
                if np.isnan(p[-1]) or p[-1] == 0:
                    continue
                _data.append(p[-1])
        extp.append((np.min(_data), np.max(_data)))

    # save the extreme value for data recovery
    write_txt(
        os.path.join(
            os.path.dirname(configs["dataset-output-path"]),
            "extp.txt"
        ),
        np.array(extp),
        fmt="%.8f"
    )
    write_txt(
        os.path.join(
            os.path.dirname(configs["dataset-output-path"]),
            "extl.txt"
        ),
        np.array(extl),
        fmt="%.8f"
    )

    return extp, extl

def _handle_dataset(method, features, power, latency):
    """
        baseline, random, CRTED & PRTED uses different parsers
    """
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
    if method == "crted":
        crted_idx = [
            2, 3, 4, 5, 14, 16, 23, 31, 38, 61, 68, 72, 73, 76, 77, 78, 83, 89, 108, 114, 121, 123,
            126, 131, 142, 147, 164, 172, 181, 187, 189, 193, 207, 211, 213, 218, 229, 230, 241, 243,
            246, 257, 259, 265, 266, 280, 281, 288, 295
        ]
        for i in range(301, 558):
            crted_idx.append(i)
        assert len(features) == len(crted_idx)
        for i in range(len(features)):
            _data = features[i].strip().split('\t')
            c_name = "CRTEDConfig%s" % str(crted_idx[i])
            _bl = []
            _bp = []
            for bmark in benchmarks.keys():
                for l in latency:
                    _l = l[0].split('-')[0].split('.')[-1].lstrip('BOOM').rstrip('Config')
                    if (c_name == _l) and (l[0].split('ChipTop-')[-1] == bmark):
                        if not (np.isnan(l[-1]) or l[-1] == 0):
                            _bl.append(l[-1])
                for p in power:
                    _p = p[0].split('-')
                    if (c_name == _p[0]) and (p[0].split('benchmarks-')[-1] == bmark):
                        if not (np.isnan(p[-1]) or p[-1] == 0):
                            _bp.append(p[-1])
            _data.append(np.mean(_bl))
            _data.append(np.mean(_bp))
            data.append(_data)
    elif method == "prted":
        for i in range(len(features)):
            _data = features[i].strip().split('\t')
            idx = i + configs["idx"]
            c_name = "PRTEDConfig%s" % str(idx)
            _bl = []
            _bp = []
            for bmark in benchmarks.keys():
                for l in latency:
                    _l = l[0].split('-')[0].split('.')[-1].lstrip('BOOM').rstrip('Config')
                    if (c_name == _l) and (l[0].split('ChipTop-')[-1] == bmark):
                        if not (np.isnan(l[-1]) or l[-1] == 0):
                            _bl.append(l[-1])
                for p in power:
                    _p = p[0].split('-')
                    if (c_name == _p[0]) and (p[0].split('benchmarks-')[-1] == bmark):
                        if not (np.isnan(p[-1]) or p[-1] == 0):
                            _bp.append(p[-1])
            _data.append(np.mean(_bl))
            _data.append(np.mean(_bp))
            data.append(_data)
    elif method == "random":
        for i in range(len(features)):
            _data = features[i].strip().split('\t')
            idx = i + configs["idx"]
            c_name = "Config%s" % str(idx)
            _bl = []
            _bp = []
            for bmark in benchmarks.keys():
                for l in latency:
                    _l = l[0].split('-')[0].split('.')[-1].lstrip('BOOM').rstrip('Config')
                    if (c_name == _l) and (l[0].split('ChipTop-')[-1] == bmark):
                        if not (np.isnan(l[-1]) or l[-1] == 0):
                            _bl.append(l[-1])
                for p in power:
                    _p = p[0].split('-')
                    if (c_name == _p[0]) and (p[0].split('benchmarks-')[-1] == bmark):
                        if not (np.isnan(p[-1]) or p[-1] == 0):
                            _bp.append(p[-1])
            _data.append(np.mean(_bl))
            _data.append(np.mean(_bp))
            data.append(_data)
    else:
        # baseline
        for i in range(len(features)):
            _data = features[i].strip().split('\t')
            c_name = latency[i][0].split('-')[0].split('.')[-1].strip('BoomConfig')
            _bl = []
            _bp = []
            for bmark in benchmarks.keys():
                for l in latency:
                    _l = l[0].split('-')[0].split('.')[-1].strip('BoomConfig')
                    if (c_name == _l) and (l[0].split('ChipTop-')[-1] == bmark):
                        if not (np.isnan(l[-1]) or l[-1] == 0):
                            _bl.append(l[-1])
                for p in power:
                    _p = p[0].split('-')[0].strip('BoomConfig')
                    if (c_name == _p) and (p[0].split('benchmarks-')[-1] == bmark):
                        if not (np.isnan(p[-1]) or p[-1] == 0):
                            _bp.append(p[-1])
            _data.append(np.mean(_bl))
            _data.append(np.mean(_bp))
            data.append(_data)
    write_csv(configs["dataset-output-path"], data, col_name=FEATURES)

def handle_dataset(method):
    with open(configs['initialize-output-path'], 'r') as f:
        features = f.readlines()
    power = read_csv(configs['power-output-path'])
    latency = read_csv(configs['latency-output-path'])
    # area = read_csv(configs['area-output-path'])
    _handle_dataset(method, features, power, latency)

def handle():
    print("Handling...")

    handle_power()
    handle_latency()
    handle_area()
    handle_dataset(configs["initialize-method"])

    print("Done.")

if __name__ == "__main__":
    argv = parse_args()
    configs = get_configs(argv.configs)
    # define the global variable `results`
    results = []
    handle()

