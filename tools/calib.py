# Author: baichen318@gmail.com

import sys, os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir)
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "dse")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "util")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "vlsi")
)
import numpy as np
from sklearn.model_selection import KFold
from util import parse_args, get_configs, load_txt, write_txt
from dse.env.rocket.design_space import parse_design_space


def load_dataset():
    dataset = load_txt(
        os.path.join(os.path.pardir, configs["dataset-output-path"]),
        fmt=float
    )
    remove_idx = []
    perf = dataset[:, -3]
    power = dataset[:, -2]
    area = dataset[:, -1]
    # remove invalid data
    for i in range(perf.shape[0]):
        if perf[i] == 0:
            remove_idx.append(i)
            continue
        if power[i] == 0:
            remove_idx.append(i)
            continue
        if area[i] == 0:
            remove_idx.append(i)
            continue
    dataset = np.delete(dataset, remove_idx, axis=0)
    return dataset


def load_design_space():
    design_space = parse_design_space(
        configs["design-space"],
        basic_component=configs["basic-component"],
        random_state=configs["seed"],
    )
    return design_space


def split_dataset(dataset):
    return KFold(n_splits=10, shuffle=True, random_state=configs["seed"])


def main():
    dataset = load_dataset()
    design_space = load_design_space()
    # construct pre-generated dataset
    new_dataset = []
    for data in dataset:
        print("[INFO]: evaluate microarchitecture:", data[:-3])
        ipc, power, area = design_space.evaluate_microarchitecture(
            configs,
            # architectural feature
            data[:-3].astype(int),
            1,
            split=True
        )
        new_dataset.append(
            np.insert(data, len(data), values=np.array([ipc, power, area * 1e6]), axis=0)
        )
    new_dataset = np.array(new_dataset)
    write_txt(
        os.path.join(
            os.path.pardir,
            os.path.dirname(configs["dataset-output-path"]),
            os.path.splitext(os.path.basename(configs["dataset-output-path"]))[0] + "-E.txt"
        ),
        new_dataset,
        fmt="%f"
    )


if __name__ == '__main__':
    configs = get_configs(parse_args().configs)
    configs["logger"] = None
    main()
