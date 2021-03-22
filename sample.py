# Author: baichen318@gmail.com

import random
import numpy as np
from collections import OrderedDict
from space import DesignSpace
from util import parse_args, get_config, create_logger, write_excel, write_txt
from exception import UnDefinedException

def parse_design_space(design_space):
    bounds = OrderedDict()
    dims = []
    size = 1
    features = []
    for k, v in design_space.items():
        # add `features`
        features.append(k)
        # calculate the size of the design space
        if 'candidates' in v.keys():
            temp = v['candidates']
            size *= len(temp)
            # generate bounds
            bounds[k] = np.array(temp)
            # generate dims
            dims.append(len(temp))
        else:
            assert 'start' in v.keys() and 'end' in v.keys() and \
                'stride' in v.keys(), "[ERROR]: assert failed. YAML includes errors."
            temp = np.arange(v['start'], v['end'] + 1, v['stride'])
            size *= len(temp)
            # generate bounds
            bounds[k] = temp
            # generate dims
            dims.append(len(temp))

    return DesignSpace(features, bounds, dims, size)

def random_sample():
    logger = create_logger("logs", "random_sample")
    logger.info("Create logger: %s" % "logs/random_sample")
    design_space = parse_design_space(configs["design-space"])
    logger.info("Size of the design space: %d" % design_space.size)

    data = design_space.random_sample(configs["sample-size"])

    # write_excel(configs["sample-output-path"], data, design_space.features)
    write_txt(configs["sample-output-path"], data)

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

