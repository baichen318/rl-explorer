# Author: baichen318@gmail.com

import random
import numpy as np
from space import parse_design_space
from util import parse_args, get_config, create_logger, write_excel, write_txt
from exception import UnDefinedException

def random_sample():
    logger = create_logger("logs", "random_sample")
    logger.info("Create logger: %s" % "logs/random_sample")
    design_space = parse_design_space(configs["design-space"])
    logger.info("Size of the design space: %d" % design_space.size)

    data = design_space.random_sample(configs["sample-size"])

    write_excel(configs["sample-output-path"] + ".xlsx", data, design_space.features)
    write_txt(configs["sample-output-path"] + ".txt", data)

def do_sample(method):
    if method == "random":
        random_sample()
    else:
        raise UnDefinedException(method + " method")

def sample():
    if "sample-method" in configs.keys():
        do_sample(configs["sample-method"])
    else:
        do_sample("random")

if __name__ == "__main__":
    argv = parse_args()
    configs = get_config(argv)
    sample()

