# Author: baichen318@gmail.com

from dse.problem.boom_design_problem import BoomDesignProblem
from dse.algo.env import MicroArchEnv
from dse.algo.dqn import DQN
from util import parse_args, get_configs

def rl_explorer():
    problem = BoomDesignProblem(configs)
    env = MicroArchEnv(problem)
    agent = DQN(env)

    agent.train()
    agent.test()

# 1. split dataset into A and B, train on A, evaluate ADRS and visualize on B

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    rl_explorer()
