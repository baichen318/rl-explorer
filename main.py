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
        test_offline_vlsi(configs)
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
        test_offline_vlsi(configs)
    else:
        assert configs["design"] == "cva6", \
            "[ERROR]: deisng: %s not support." % configs["design"]
        pass

def sim():
    from vlsi.vlsi import offline_vlsi

    if_exist(configs["design-output-path"], strict=True)
    offline_vlsi(configs)

def generate_dataset():
    from vlsi.vlsi import generate_dataset

    generate_dataset(configs)

def rl_explorer():
    from dse.algo.dqn import DQN
    logger, time_str = create_logger(
        os.path.dirname(configs["log"]),
        os.path.basename(configs["log"])
    )
    configs["logger"] = logger
    configs["model-path"] = os.path.join("models", "%s-%s" % (
        os.path.basename(configs["log"]),
        time_str)
    )
    mkdir(configs["model-path"])

    if configs["design"] == "boom":
        from dse.env.boom.design_env import BoomDesignEnv
        from vlsi.boom.vlsi import PreSynthesizeSimulation
        # Notice: we should modify `self.configs["batch"]`
        configs["batch"] = configs["batch"] * 5
        env = BoomDesignEnv(configs)
    elif configs["design"] == "rocket":
        from dse.env.rocket.design_env import RocketDesignEnv
        from vlsi.rocket.vlsi import PreSynthesizeSimulation
        env = RocketDesignEnv(configs)
    else:
        assert configs["design"] == "cva6", \
            "[ERROR]: deisng: %s not support." % configs["design"]
        from dse.env.cva6.design_env import CVA6DesignEnv
        from vlsi.cva6.vlsi import PreSynthesizeSimulation
        configs["design"] = CVA6DesignEnv(configs)
    agent = DQN(env)
    PreSynthesizeSimulation.set_tick(configs["idx"], configs["logger"])

    for i in range(1, configs["rl-round"] + 1):
        agent.run(i)
    # agent.search()

def test_rl_explorer():
    """
        debug version of `rl_explorer`
    """
    from dse.algo.dqn import DQN
    logger, time_str = create_logger(
        os.path.dirname(configs["log"]),
        os.path.basename(configs["log"])
    )
    configs["logger"] = logger
    configs["model-path"] = os.path.join("models", "%s-%s" % (
        os.path.basename(configs["log"]),
        time_str)
    )
    mkdir(configs["model-path"])

    if configs["design"] == "boom":
        from dse.env.boom.design_env import BoomDesignEnv
        from vlsi.boom.vlsi import PreSynthesizeSimulation
        # Notice: we should modify `self.configs["batch"]`
        configs["batch"] = configs["batch"] * 5
        env = BoomDesignEnv(configs)
    elif configs["design"] == "rocket":
        from dse.env.rocket.design_env import RocketDesignEnv
        from vlsi.rocket.vlsi import PreSynthesizeSimulation
        env = RocketDesignEnv(configs)
    else:
        assert configs["design"] == "cva6", \
            "[ERROR]: deisng: %s not support." % configs["design"]
        from dse.env.cva6.design_env import CVA6DesignEnv
        from vlsi.cva6.vlsi import PreSynthesizeSimulation
        configs["design"] = CVA6DesignEnv(configs)
    agent = DQN(env)
    PreSynthesizeSimulation.set_tick(configs["idx"], configs["logger"])

    for i in range(1, configs["rl-round"] + 1):
        agent.test_run(i)
    # agent.search()

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    mode = configs["mode"]
    if mode == "generate-design":
        generate_design()
    elif mode == "rl":
        if configs["debug"]:
            test_rl_explorer()
        else:
            rl_explorer()
    else:
        raise NotImplementedError()
