# Author: baichen318@gmail.com
import os
import argparse
import yaml
import pandas as pd
import numpy as np
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
            config = yaml.load(f)

        return config
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

def read_csv(data):
    if if_exist(data, strict=True):

        return np.array(pd.read_csv(data))

