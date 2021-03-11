# Author: baichen318@gmail.com

import os
import re
import pandas as pd
from util import parse_args, get_config, if_exist
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
                    bmark + '.' + root.split('-ChipTop')[0] + '.out'
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

def handle():
    print("Handling...")

    if config['mode'] == 'power':
        handle_power()
    elif config['mode'] == 'latency':
        handle_latency()
    elif config['mode'] == 'area':
        handle_area()
    else:
        raise UnDefinedException(config['mode'])

    print("Done.")

if __name__ == "__main__":
    argv = parse_args()
    config = get_config(argv)
    # define the global variable `results`
    results = []
    handle()

