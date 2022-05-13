# Author: baichen318@gmail.com

import sys, os
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir,
        "dse"
    )
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir,
        "simulation"
    )
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir,
        "utils"
    )
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "algo"
    )
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "util"
    )
)
from boom_explorer import boom_explorer
from problem import define_problem
from util import parse_args
from utils import get_configs


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
    main()
