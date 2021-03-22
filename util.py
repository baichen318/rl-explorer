# Author: baichen318@gmail.com
import os
import argparse
import yaml
import time
import pandas as pd
import numpy as np
from datetime import datetime
from exception import NotFoundException, UnDefinedException
import logging

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
        np.loadtxt(path, dytpe=fmt)
    else:
        print("[WARN]: cannot load %s" % path)
