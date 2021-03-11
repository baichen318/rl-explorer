# Author: baichen318@gmail.com

from model import GP
from util import parse_args, get_config, if_exist, timer

def init():
    model = GP(configs)
    model.init()

    return model

def do_design_explore(model):
    for _ in range(model.iteration):
        model.sample()
        model.query()
        model.record()
    model.final_record()

def design_explorer():
    model = init()
    start_time = timer()
    do_design_explore(model)
    end_time = timer()
    print("[INFO]: time elapsed: %s h" % str((end_time - start_time).seconds / 3600))

if __name__ == "__main__":
    argv = parse_args()
    configs = get_config(argv)
    design_explorer()
