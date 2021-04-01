# Author: baichen318@gmail.com
import os
import argparse
import yaml
import time
import csv
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn import metrics
from exception import NotFoundException, UnDefinedException

def parse_args():

    def initialize_parser(parser):
        parser.add_argument('-c', '--config',
            required=True,
            type=str,
            default='config.yml',
            help='YAML file to be handled')

        return parser

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = initialize_parser(parser)

    return parser.parse_args()

def get_config(argv):
    if hasattr(argv, 'config'):
        with open(argv.config, 'r') as f:
            configs = yaml.load(f)

        return configs
    else:
        raise UnDefinedException('config')

def if_exist(path, strict=False):
    try:
        if os.path.exists(path):
            return True
        else:
            raise NotFoundException(path)
    except NotFoundException as e:
        print(e)
        if not strict:
            return False
        else:
            exit(1)

def mkdir(path):
    if not if_exist(path):
        print("[INFO]: create directory: %s" % path)
        os.makedirs(path, exist_ok=True)

def read_csv(data):
    if if_exist(data, strict=True):

        return np.array(pd.read_csv(data))

def read_csv_v2(file):
    data = []
    if_exist(file, strict=True)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        title = next(reader)
        for row in reader:
            data.append(row)

    return data, title

def calc_mape(x, y):
        return np.mean(np.abs((np.array(x) - np.array(y)) / np.array(y)))

def timer():

    return datetime.now()

def point2knob(p, dims):
    """convert point form (single integer) to knob form (vector)"""
    knob = []
    for dim in dims:
        knob.append(p % dim)
        p //= dim

    return knob

def knob2point(knob, dims):
    """convert knob form (vector) to point form (single integer)"""
    p = 0
    for j, k in enumerate(knob):
        p += int(np.prod(dims[:j])) * k

    return p

def execute(cmd, logger=None):
    if logger:
        logger.info("executing: %s " % cmd)
    else:
        print("[INFO]: executing: %s" % cmd)

    os.system(cmd)

def create_logger(path, name):
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_file = '{}_{}.log'.format(name, time_str)
    mkdir(path)
    log_file = os.path.join(path, log_file)
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def dump_yaml(path, yml_dict):
    with open(path, 'w') as f:
        yaml.dump(yml_dict, f)

def is_pow2(num):
    if not (num & (num - 1)):

        return True
    else:

        return False

def write_csv(path, data, mode='w', col_name=None):

    print('[INFO]: writing csv to %s' % path)
    with open(path, mode) as f:
        writer = csv.writer(f)
        if col_name:
            writer.writerow(col_name)
        writer.writerows(data)

def write_excel(path, data, features):
    """
        data: np.array
    """
    writer = pd.ExcelWriter(path)
    _data = pd.DataFrame(data)
    _data.columns = features
    _data.to_excel(writer, 'page_1')
    print("[INFO]: saving to %s" % path)
    writer.save()

def write_txt(path, data, fmt='%i'):
    """
        data: np.array
    """
    dims = len(data.shape)
    if dims > 2:
        print("[WARN]: cannot save to %s" % path)
        return
    print("[INFO]: saving to %s" % path)
    np.savetxt(path, data, fmt)

def load_txt(path, fmt=int):
    if if_exist(path):
        print("[INFO]: loading to %s" % path)
        return np.loadtxt(path, dtype=fmt)
    else:
        print("[WARN]: cannot load %s" % path)

def mse(gt, predict):
    # gt: `np.array`
    # predict: `np.array`
    return metrics.mean_squared_error(gt, predict)

def r2(gt, predict):
    # gt: `np.array`
    # predict: `np.array`
    return metrics.r2_score(gt, predict)

def mape(gt, predict):
    # gt: `np.array`
    # predict: `np.array`
    return np.mean(np.abs((predict - gt) / gt)) * 100

def hyper_volume(reference, point):
    """
        reference: `np.array`
        point: `np.array`
        (latency, power)
    """
    sign = np.sign(reference[0] - point[0]) + \
        np.sign(reference[1] - point[1])
    if sign != -2:
        return -1
    else:
        hv = (np.abs(reference[0] - point[0]) / point[0]) * \
            (np.abs(reference[1] - point[1]) / point[1])
        return hv
