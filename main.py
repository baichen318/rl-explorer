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
import time
import torch
import socket
import getpass
from dse.algo.a2c.a2c import a2c
from utils import parse_args, get_configs, write_txt, if_exist, \
    mkdir, create_logger, execute


def pre_requisite():
    configs["output-path"] = os.path.join(
        configs["output-path"],
        "{}-{}-{}-{}-{}".format(
            configs["mode"],
            configs["design"].replace(' ', '-'),
            getpass.getuser(),
            socket.gethostname(),
            time.strftime("%Y-%m-%d-%H-%M")
        )
    )
    logger, log_file = create_logger(
        configs["output-path"],
        "{}-{}-{}-{}".format(
            configs["mode"],
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
    configs["summary-writer"] = os.path.join(
        configs["output-path"],
        "summary-{}".format(log_file)
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
    else:
        assert configs["design"] == "Rocket", \
            "[ERROR]: {} is not supported.".format(configs["design"])
        from dse.env.rocket.env import RocketEnv as env
    a2c(env, configs)


def main():
    rl_explorer()


if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    configs["configs"] = parse_args().configs
    main()
