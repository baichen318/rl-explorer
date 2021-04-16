# Author: baichen318@gmail.com

import random
import numpy as np
from space import parse_design_space
from util import parse_args, get_config, create_logger, write_excel, write_txt
from exception import UnDefinedException

def create_design_space():
    logger = create_logger("logs", "random-initialize")
    design_space = parse_edsign_space(configs["design-space"])
    logger.info("Design space size: %d" % design_space.size)

    return design_space

def record_sample(design_space, data):
    write_excel(configs["initialize-output-path"] + ".xlsx", data, design_space.features)
    write_txt(configs["initialize-output-path"] + ".txt", data)

def initialize(method):
    design_space = create_design_space()

    if method == "random":
        data = design_space.random_sample(configs["initialize-size"])
    else:
        raise UnDefinedException(method + " method")

    record_sample(design_space, data)

if __name__ == "__main__":
    argv = parse_args()
    configs = get_configs(argv.configs)
    initalize(configs["initialize-method"])
