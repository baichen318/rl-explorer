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
from time import time
from vlsi import PreSynthesizeSimulation
from util import parse_args, get_configs, write_txt, if_exist, \
    mkdir, create_logger, execute

def generate_design():
    from dse.env.design_space import parse_design_space

    design_space = parse_design_space(
        configs["design-space"],
        random_state=round(time()),
        basic_component=configs["basic-component"]
    )
    design = design_space.sample_v1(batch=configs["batch"], f=configs["design-output-path"])
    write_txt(configs["design-output-path"], design.numpy())

def sim():
    from vlsi.vlsi import offline_vlsi

    if_exist(configs["design-output-path"], strict=True)
    offline_vlsi(configs)

def generate_dataset():
    from vlsi.vlsi import generate_dataset

    generate_dataset(configs)

def rl_explorer():
    from dse.env.boom_design_env import BoomDesignEnv
    from dse.algo.dqn import DQN

    logger, time_str = create_logger(
        os.path.dirname(configs["log"]),
        os.path.basename(configs["log"])
    )
    configs["logger"] = logger
    configs["model-path"] = os.path.join("models", "model-%s" % time_str)
    mkdir(configs["model-path"])

    # Notice: we should modify `self.configs["batch"]`
    configs["batch"] = configs["batch"] * 5
    env = BoomDesignEnv(configs)
    agent = DQN(env)
    PreSynthesizeSimulation.set_tick(configs["idx"], configs["logger"])

    for i in range(1, configs["episode"] + 1):
        agent.run(i)
        agent.save()
    # agent.search()

def test_rl_explorer():
    """
        debug version of `rl_explorer`
    """
    from dse.env.boom_design_env import BoomDesignEnv
    from dse.algo.dqn import DQN

    logger, time_str = create_logger(
        os.path.dirname(configs["log"]),
        os.path.basename(configs["log"])
    )
    configs["logger"] = logger
    configs["model-path"] = os.path.join("models", "model-%s" % time_str)
    mkdir(configs["model-path"])
    execute("rm -rf test")

    # Notice: we should modify `self.configs["batch"]`
    configs["batch"] = configs["batch"] * 5
    env = BoomDesignEnv(configs)
    agent = DQN(env)
    PreSynthesizeSimulation.set_tick(configs["idx"], configs["logger"])

    for i in range(1, configs["episode"] + 1):
        agent.test_run(i)
        agent.save()
    # agent.search()

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    mode = configs["mode"]
    if mode == "generate-design":
        generate_design()
    elif mode == "sim":
        sim()
    elif mode == "generate-data":
        generate_dataset()
    elif mode == "rl":
        if configs["debug"]:
            test_rl_explorer()
        else:
            rl_explorer()
    else:
        raise NotImplementedError()
