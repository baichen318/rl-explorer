# Author: baichen318@gmail.com

from time import time
from util.util import parse_args, get_configs, write_txt

def generate_design():
    from dse.problem.design_space import parse_design_space

    design_space = parse_design_space(
        configs["design-space"],
        random_state=round(time()),
        basic_component=configs["basic-component"]
    )
    design = design_space.sample(batch=10)
    write_txt(configs["design-output-path"], design.numpy())

def sim():
    pass

def rl_explorer():
    from dse.problem.boom_design_problem import BoomDesignProblem
    from dse.algo.env import MicroArchEnv
    from dse.algo.dqn import DQN

    problem = BoomDesignProblem(configs)
    env = MicroArchEnv(problem)
    agent = DQN(env)

    agent.train()
    agent.test()

# 1. split dataset into A and B, train on A, evaluate ADRS and visualize on B

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    mode = configs["mode"]
    if mode == "generate-design":
        generate_design()
    elif mode == "sim":
        sim()
    elif mode == "rl":
        rl_explorer()
    else:
        raise NotImplementedError()
