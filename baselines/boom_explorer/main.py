# Author: baichen318@gmail.com


import os
from utils.utils import get_configs, Timer
from baselines.boom_explorer.util.util import parse_args
from baselines.boom_explorer.algo.problem import define_problem
from baselines.boom_explorer.algo.boom_explorer import boom_explorer


def main():
    problem = define_problem(configs)
    boom_explorer(configs, settings, problem)


if __name__ == "__main__":
    args = parse_args()
    configs = get_configs(args.configs)
    settings = get_configs(args.settings)
    configs["rl-explorer-root"] = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir,
        )
    )
    configs["configs"] = args.configs
    configs["logger"] = None
    with Timer("BOOM-Explorer"):
        main()
