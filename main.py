# Author: baichen318@gmail.com


import sys, os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "dse")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "utils")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "simulation")
)
import getpass
import socket
import torch
from dse.algo.a2c.a2c import a2c
from time import time
from utils import parse_args, get_configs, write_txt, if_exist, \
    mkdir, create_logger, execute


def pre_requisite():
    logger, log_file = create_logger(
        configs["output-path"],
        "{}-{}-{}".format(
            configs["design"].replace(' ', '-'),
            getpass.getuser(),
            socket.gethostname()
        )
    )
    configs["logger"] = logger
    configs["log-path"] = os.path.join(
        configs["output-path"],
        log_file
    )
    configs["model-path"] = os.path.join(
        configs["output-path"],
        "models",
    )
    mkdir(configs["model-path"])


def rl_explorer():
    pre_requisite()
    if "BOOM" in configs["design"]:
        from dse.env.boom.env import BOOMEnv as env
    # elif configs["design"] == "rocket":
    #     assert configs["design"] == "rocket", \
    #         "[ERROR]: {} is not supported.".format(configs["design"])
    #     from dse.env.rocket.design_env import RocketEnv as env
    a2c(env, configs)


def main():
    rl_explorer()


if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    configs["configs"] = parse_args().configs
    main()
