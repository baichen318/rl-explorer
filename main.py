# Author: baichen318@gmail.com

import sys, os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "dse")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "util")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "vlsi")
)
import torch
from dse.algo.a3c.a3c import a3c
from time import time
from util import parse_args, get_configs, write_txt, if_exist, \
    mkdir, create_logger, execute


def generate_design():
    logger, time_str = create_logger(
        os.path.dirname(os.path.abspath(os.path.dirname(__file__))),
        os.path.basename("generate-design")
    )
    configs["logger"] = logger
    if configs["design"] == "boom":
        from dse.env.boom.design_space import parse_design_space
        from vlsi.boom.vlsi import offline_vlsi, test_offline_vlsi

        design_space = parse_design_space(
            configs["design-space"],
            random_state=round(time()),
            basic_component=configs["basic-component"]
        )
        design = design_space.sample_v1(batch=configs["batch"])
        write_txt(configs["design-output-path"], design.numpy())
        if configs["debug"]:
            test_offline_vlsi(configs)
        else:
            offline_vlsi(configs)
    elif configs["design"] == "rocket":
        from dse.env.rocket.design_space import parse_design_space
        from vlsi.rocket.vlsi import offline_vlsi, test_offline_vlsi

        design_space = parse_design_space(
            configs["design-space"],
            random_state=round(time()),
            basic_component=configs["basic-component"]
        )
        design = design_space.sample(batch=configs["batch"])
        write_txt(configs["design-output-path"], design.numpy())
        if configs["debug"]:
            test_offline_vlsi(configs)
        else:
            offline_vlsi(configs)
    else:
        assert configs["design"] == "cva6", \
            "[ERROR]: deisgn: %s not support." % configs["design"]
        pass


def sim():
    from vlsi.vlsi import offline_vlsi

    if_exist(configs["design-output-path"], strict=True)
    offline_vlsi(configs)


def generate_dataset():
    if configs["design"] == "boom":
        from vlsi.boom.vlsi import generate_dataset
        generate_dataset(configs)
    elif configs["design"] == "rocket":
        from vlsi.rocket.vlsi import generate_dataset
        generate_dataset(configs)
    elif configs["design"] == "cva6":
        from vlsi.cva6.vlsi import generate_dataset
        generate_dataset(configs)
    else:
        assert configs["design"] == "cva6", \
            "[ERROR]: design: %s not support." % configs["design"]
        pass


def set_torch():
    torch.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def rl_explorer():
    logger, time_str, log_file = create_logger(
        os.path.dirname(configs["log"]),
        os.path.basename(configs["log"])
    )
    configs["logger"] = logger
    configs["model-path"] = os.path.join("models", "%s-%s" % (
        os.path.basename(configs["log"]),
        time_str)
    )
    configs["log-file"] = log_file
    mkdir(configs["model-path"])

    if configs["design"] == "boom":
        from dse.env.boom.design_env import BoomDesignEnv
        # Notice: we should modify `self.configs["batch"]`
        configs["batch"] = configs["batch"] * 5
        a3c(BoomDesignEnv, configs)
    elif configs["design"] == "rocket":
        from dse.env.rocket.design_env import RocketDesignEnv
        a3c(RocketDesignEnv, configs)
    else:
        assert configs["design"] == "cva6", \
            "[ERROR]: deisng: %s not support." % configs["design"]
        from dse.env.cva6.design_env import CVA6DesignEnv
        a3c(CVA6DesignEnv, configs)



if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    mode = configs["mode"]
    if mode == "generate-design":
        generate_design()
    elif mode == "generate-dataset":
        generate_dataset()
    elif mode == "rl":
        rl_explorer()
    else:
        raise NotImplementedError()
