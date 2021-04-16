# Author: baichen318@gmail.com

import os

# from model import GP
# from util import parse_args, get_config, if_exist, timer

# def init():
#     model = GP(configs)
#     model.init()

#     return model

# def do_design_explore(model):
#     for _ in range(model.iteration):
#         model.sample()
#         model.record()
#         model.query()
#     model.final_record()

# def design_explorer():
#     model = init()
#     start_time = timer()
#     do_design_explore(model)
#     end_time = timer()
#     print("[INFO]: time elapsed: %s h" % str((end_time - start_time).seconds / 3600))
#     model.verification()

# if __name__ == "__main__":
#     argv = parse_args()
#     configs = get_config(argv)
#     design_explorer()

def design_explorer():
    if configs["flow"] == "initialize" or \
        configs["flow"] == "all":
        # initialize
        cmd = "python sample.py -c configs/design-explorer.yml"
        print("[INFO]: executing: %s" % cmd)
        os.system(cmd)
        # offline VLSI flow
        cmd = "python vlsi/vlsi.py -c configs/design-explorer.yml"
        print("[INFO]: executing: %s" % cmd)
        os.system(cmd)
    if configs["flow"] == "search" or \
        configs["flow"] == "all":
        # TODO: search
        # TODO: plot
        pass

if __name__ == "__main__":
    argv = parse_args()
    configs = get_configs(argv.configs)
    design_explorer()
