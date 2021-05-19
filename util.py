# Author: baichen318@gmail.com
import os
import argparse
import yaml
import time
import csv
import torch
import pandas as pd
import numpy as np
import logging
from typing import Union
from datetime import datetime
from sklearn import metrics
from exception import NotFoundException, UnDefinedException

def parse_args():

    def initialize_parser(parser):
        parser.add_argument('-c', '--configs',
            required=True,
            type=str,
            default='configs.yml',
            help='YAML file to be handled')

        return parser

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = initialize_parser(parser)

    return parser.parse_args()

def get_configs(fyaml):
    if_exist(fyaml, strict=True)
    with open(fyaml, 'r') as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            configs = yaml.load(f)

    return configs

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

def read_csv(data, header=0):
    """
        data: <str>
    """
    if_exist(data, strict=True)
    return np.array(pd.read_csv(data, header=header))

def load_dataset(csv_path, preprocess=True):
    """
        csv_path: <str>
    """
    def _read_csv(csv_path):
        data = []
        if_exist(csv_path, strict=True)
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            title = next(reader)
            for row in reader:
                data.append(row)

        return data, title

    def validate(dataset):
        """
            `dataset`: <tuple>
        """
        data = []
        for item in dataset:
            _data = []
            f = item[0].split(' ')
            for i in f:
                _data.append(int(i))
            for i in item[1:]:
                _data.append(float(i))
            data.append(_data)
        data = np.array(data)

        return data

    dataset, _ = _read_csv(csv_path)
    dataset = validate(dataset)
    # scale the data by `max - x / \alpha`
    x = []
    y = []
    for data in dataset:
        x.append(data[:-2])
        if preprocess:
            y.append(np.array([(90000 - data[-2]) / 20000, (0.2 - data[-1]) * 10]))
        else:
            y.append(np.array([data[-2], data[-1]]))

    return np.array(x), np.array(y)

def recover_data(data: Union[torch.Tensor, np.ndarray]):
    """
        data: <torch.Tensor>: M x 2
    """
    data[:, 0] = 90000 - 20000 * data[:, 0]
    data[:, 1] = 0.2 - data[:, 1] / 10

    return data

def split_dataset(dataset):
    # split dataset into x label & y label
    # dataset: <numpy.ndarray>
    x = []
    y = []
    for data in dataset:
        x.append(data[0:-2])
        # TODO: max - x / \alpha
        # scale c.c. & power approximately
        y.append(np.array([data[-2] / 90000, data[-1] * 10]))

    return np.array(x), np.array(y)

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
    """
        path: path to logs. directory
        name: prefix name of a log file
    """
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
    logger.info("[INFO]: create logger: %s/%s" % (path, name))

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
        `path`: path to the output file
        `data`: <np.array>
        `features`: <list> column names
    """
    writer = pd.ExcelWriter(path)
    _data = pd.DataFrame(data)
    _data.columns = features
    _data.to_excel(writer, 'page_1')
    print("[INFO]: saving to %s" % path)
    writer.save()

def write_txt(path, data, fmt='%i'):
    """
        `path`: path to the output path
        `data`: <np.array>
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
    # return np.mean(np.abs(predict - gt) / gt)) * 100
    return metrics.mean_absolute_percentage_error(gt, predict)

def rmse(gt, predict):
    """
        gt: <numpy.ndarray>
        predict: <numpy.ndarray>
    """
    return np.mean(np.sqrt(np.power(gt - predict, 2)))

def strflush(msg, logger=None):
    if logger:
        logger.info(msg)
    else:
        print(msg)

def hyper_volume(reference, point):
    """
        reference: `np.array`
        point: `np.array`
        (latency, power)
    """
    sign = np.sign(point[0] - reference[0]) + \
        np.sign(point[1] - reference[1])
    if sign != -2:
        return -1
    else:
        hv = (np.abs(reference[0] - point[0]) / reference[0]) * \
            (np.abs(reference[1] - point[1]) / reference[1])
        return hv

def adrs(reference, point):
    """
        reference: <numpy.ndarray>
        point: <numpy.ndarray>
    """
    return np.sqrt((reference[0] - point[0]) ** 2 + (reference[1] - point[1]) ** 2)

def adrs_v2(reference, learned_pareto_set):
    """
        reference: <torch.Tensor>
        learned_pareto_set: <torch.Tensor>
    """
    # calculate average distance to the reference set
    ADRS = 0
    try:
        reference = reference.cpu()
        learned_pareto_set = learned_pareto_set.cpu()
    except:
        pass
    for omiga in reference:
        mini = float('inf')
        for gama in learned_pareto_set:
            mini = min(mini, np.linalg.norm(omiga - gama))
        ADRS += mini
    ADRS = ADRS / len(reference)
    return ADRS

def get_pareto_points(data_array):
    num_points = data_array.shape[0]
    dim = data_array.shape[1]
    data_array = data_array.reshape((num_points, dim))

    data_array = np.array(list(set([tuple(t) for t in data_array])))
    num_points = data_array.shape[0]

    delpoints = []
    for i in range(num_points):
        temp = data_array[i,:][None,:]
        temp2 = np.delete(data_array, i, axis=0)
        acjudge = 0
        for j in range(num_points - 1):
            # NOTICE: we have to use `>` since we rescale data with inverse directions
            judge = temp > temp2[j,:]
            judge = judge + 0
            if max(judge[0, :]) == 1:
                acjudge = acjudge + 1
        if acjudge < num_points - 1:
            delpoints = delpoints + [i]
    pareto_set = np.delete(data_array, delpoints, axis=0)

    return pareto_set
