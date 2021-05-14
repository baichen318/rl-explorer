
from bayesian_opt import BayesianOptimization
from util import parse_args, get_configs

def main():
    manager = BayesianOptimization(configs)
    manager.run()
    manager.validate()

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    main()
