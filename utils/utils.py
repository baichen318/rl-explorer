# Author: baichen318@gmail.com


import os
import csv
import yaml
import time
import torch
import shutil
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from math import ceil, log
from sklearn import metrics
from datetime import datetime
from typing import Union, List
from utils.exceptions import NotFoundException


def parse_args():
    def initialize_parser(parser):
        parser.add_argument(
            "-c",
            "--configs",
            required=True,
            type=str,
            default="configs.yml",
            help="YAML file to be handled")
        return parser

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = initialize_parser(parser)
    return parser.parse_args()


def get_configs(fyaml: str):
    if_exist(fyaml, strict=True)
    with open(fyaml, 'r') as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            configs = yaml.load(f)
    return configs


def get_configs_from_command():
    return get_configs(parse_args().configs)


def if_exist(path: str, strict=False):
    try:
        if os.path.exists(path):
            return True
        else:
            raise NotFoundException(path)
    except NotFoundException as e:
        warn(e)
        if not strict:
            return False
        else:
            exit(1)


def mkdir(path: str):
    if not if_exist(path):
        info("create directory: {}".format(path))
        os.makedirs(path, exist_ok=True)


def copy(src: str, tgt: str):
    shutil.copy(src, tgt)
    info("copy {} to {}".format(src, tgt))


def remove(path: str):
    if if_exist(path):
        if os.path.isfile(path):
            os.remove(path)
            info("remove %s" % path)
        elif os.path.isdir(path):
            if not os.listdir(path):
                # empty directory
                os.rmdir(path)
            else:
                shutil.rmtree(path)
            info("remove %s" % path)


def timer():
    return datetime.now()


def execute(cmd: str, logger=None):
    if logger:
        logger.info("executing: {}".format(cmd))
    else:
        print("[INFO]: executing: {}".format(cmd))
    return os.system(cmd)


def execute_with_subprocess(cmd: str, logger=None):
    if logger:
        logger.info("executing: {}".format(cmd))
    else:
        print("[INFO]: executing: {}".format(cmd))
    subprocess.call(["bash", "-c", cmd])


def create_logger(path: str, name: str):
    """
        path: the path of logs (directory)
        name: the prefix name of a log file
    """
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_file = "{}-{}.log".format(name, time_str)
    mkdir(path)
    log_file = os.path.join(path, log_file)
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    logger.info("[INFO]: create logger: {}".format(log_file))
    # logger.removeHandler(console)
    return logger, log_file


def strflush(msg: str, logger=None):
    if logger:
        logger.info(msg)
    else:
        info(msg)


def dump_yaml(path: str, yml_dict: dict):
    with open(path, 'w') as f:
        yaml.dump(yml_dict, f)
    info("dump YAML to {}".format(path))


def is_power_of_two(num: int):
    if not (num & (num - 1)):
        return True
    else:

        return False


def load_txt(path: str, fmt=int):
    if if_exist(path):
        info("loading from {}".format(path))
        return np.loadtxt(path, dtype=fmt)
    else:
        warn("cannot load {}".format(path))


def load_csv(data: str, header=0):
    """
        data: path of the CSV data
    """
    if_exist(data, strict=True)
    return np.array(pd.read_csv(data, header=header))


def load_excel(path: str, sheet_name: Union[int, str] = 0):
    """
        path: path of the Excel file
        sheet_name: index of the sheet
    """
    if_exist(path, strict=True)
    data = pd.read_excel(path, sheet_name=sheet_name)
    info("read the sheet {} of excel from {}".format(sheet_name, path))
    return data


def write_txt(path: str, data: np.ndarray, fmt="%i"):
    """
        path: path to the output path
        data: numpy.ndarray data
    """
    dims = len(data.shape)
    if dims > 2:
        warn("cannot save to {}".format(path))
        return
    info(
        "saving matrix (%d x %d) to %s" %
        (data.shape[0], data.shape[1], path)
    )
    np.savetxt(path, data, fmt)


def remove_prefix(s: str, prefix: str):
    if prefix and s.startswith(prefix):
        return s[len(prefix):]
    else:
        return s[:]


def remove_suffix(s: str, suffix: str):
    """
        s: the original string
        suffix: the suffix to be removed
    """
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s[:]


def write_csv(path: str, data: np.ndarray, mode='w', col_name=None):
    with open(path, mode) as f:
        writer = csv.writer(f)
        if col_name:
            writer.writerow(col_name)
        writer.writerows(data)
    info("writing csv to {}".format(path))


def write_excel(path: str, data: np.ndarray, features: List[str]):
    """
        path: the path of the output file
        data: numpy.ndarray data
        features: column names
    """
    writer = pd.ExcelWriter(path)
    _data = pd.DataFrame(data)
    _data.columns = features
    _data.to_excel(writer, "page_1")
    writer.save()
    info("saving to %s" % path)


def timestamp():
    return time.time()


def mse(gt: np.ndarray, predict: np.ndarray):
    """
        gt: ground-truth values
        predict: predictions
    """
    return metrics.mean_squared_error(gt, predict)


def r2(gt: np.ndarray, predict: np.ndarray):
    """
        gt: ground-truth values
        predict: predictions
    """
    return metrics.r2_score(gt, predict)


def mape(gt: np.ndarray, predict: np.ndarray):
    """
        gt: ground-truth values
        predict: predictions
    """
    return metrics.mean_absolute_percentage_error(gt, predict)


def rmse(gt: np.ndarray, predict: np.ndarray):
    """
        gt: ground-truth values
        predict: predictions
    """
    return np.mean(np.sqrt(np.power(gt - predict, 2)))


def round_power_of_two(x: int):
    return pow(2, ceil(log(x)/log(2)))


def info(msg: str):
    """
        msg: message
    """
    print("[INFO]: {}".format(msg))


def test(msg: str):
    """
        msg: <str>
    """
    print("[TEST]: {}".format(msg))


def warn(msg: str):
    """
        msg: message
    """
    print("[WARN]: {}".format(msg))


def error(msg: str):
    """
        msg: message
    """
    print("[ERROR]: {}".format(msg))
    exit(1)


def assert_error(msg: str):
    return "[ERROR]: {}".format(msg)


class MultiLogHandler(logging.Handler):
    """
        support for multiple loggers
    """
    def __init__(self, dirname):
        super(MultiLogHandler, self).__init__()
        self._loggers = {}
        self._dirname = dirname
        mkdir(self.dirname)

    @property
    def loggers(self):
        return self._loggers

    @property
    def dirname(self):
        return self._dirname

    def flush(self):
        self.acquire()
        try:
            for logger in self.loggers.values():
                logger.flush()
        finally:
            self.release()

    def _get_or_open(self, key):
        self.acquire()
        try:
            if key in self.loggers.keys():
                return self.loggers[key]
            else:
                logger = open(os.path.join(self.dirname, "{}.log".format(key)), 'a')
                self.loggers[key] = logger
                return logger
        finally:
            self.release()

    def emit(self, record):
        try:
            logger = self._get_or_open(record.threadName)
            msg = self.format(record)
            logger.write("{}\n".format(msg))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class Timer(object):
    def __init__(self, msg):
        super(Timer, self).__init__()
        self.msg = msg
        self.time = None
        self.duration = 0

    @property
    def now(self):
        return time.time()

    def __enter__(self):
        self.time = self.now

    def __exit__(self, type, value, trace):
        self.duration = self.now - self.time
        info("[{}]: duration: {} s".format(
            self.msg, self.duration)
        )
