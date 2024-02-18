# Author: baichen318@gmail.com


import os
import sys
import time
import torch
import socket
import getpass
from dse.algo.a3c.a3c import a3c
from utils.utils import parse_args, get_configs, write_txt, \
    assert_error, if_exist, mkdir, create_logger, execute, \
    get_configs_from_command, copy


def pre_requisite(configs):
    """
        set `configs` with
        1. logger
        2. log-path
        3. summary-writer
        4. model-path
        - output-path/<task name>:
        ---- log
        ---- summary-log
        ---- models/
    """
    configs["algo"]["output-root"] = os.path.join(
        configs["algo"]["output-root"],
        "{}-{}-{}-{}-{}".format(
            configs["algo"]["mode"],
            configs["algo"]["design"].replace(' ', '-'),
            getpass.getuser(),
            socket.gethostname(),
            time.strftime("%Y-%m-%d-%H-%M")
        )
    )
    logger, log_file = create_logger(
        configs["algo"]["output-root"], "log"
    )
    # set `logger`
    configs["logger"] = logger
    # set `log-path`
    configs["log-path"] = os.path.join(
        configs["algo"]["output-root"],
        log_file
    )
    # set `summary-writer`
    configs["summary-writer"] = os.path.join(
        configs["algo"]["output-root"],
        "summary-{}".format(log_file)
    )
    # set `model-path`
    configs["model-path"] = os.path.join(
        configs["algo"]["output-root"],
        "models",
    )
    if configs["algo"]["mode"] == "train":
        mkdir(configs["model-path"])


def rl_explorer(configs):
    pre_requisite(configs)
    # an interface for the simulation
    configs["configs"] = fyaml
    # copy the YAML to the output directory
    copy(fyaml, configs["algo"]["output-root"])
    if "BOOM" in configs["algo"]["design"]:
        from dse.env.boom.env import BOOMEnv as env
    else:
        assert configs["algo"]["design"] == "Rocket", \
            assert_error(
                "{} is not supported.".format(
                    configs["algo"]["design"]
                )
            )
        from dse.env.rocket.env import RocketEnv as env
    a3c(env, configs)


def main(configs):
    rl_explorer(configs)


if __name__ == "__main__":
    fyaml = parse_args().configs
    main(get_configs_from_command())
