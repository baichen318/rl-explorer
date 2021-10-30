# Author: baichen318@gmail.com

import sys, os
sys.path.insert(
    0,
    os.path.join(os.path.pardir, os.path.dirname(__file__))
)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.pardir, "util")
    )
)
import numpy as np
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from dse.env.boom.design_space import parse_design_space
from util import parse_args, get_configs, load_txt


def load_design_space():
    design_space = parse_design_space(
        configs["design-space"],
        basic_component=configs["basic-component"],
        random_state=configs["seed"]
    )

    return design_space


def load_model():
    ipc_model = joblib.load(
        os.path.join(
            configs["ppa-model"],
            configs["design"] + '-' + "ipc.pt"
        )
    )
    power_model = joblib.load(
        os.path.join(
            configs["ppa-model"],
            configs["design"] + '-' + "power.pt"
        )
    )
    area_model = joblib.load(
        os.path.join(
            configs["ppa-model"],
            configs["design"] + '-' + "area.pt"
        )
    )
    return {
        "ipc": ipc_model,
        "power": power_model,
        "area": area_model
    }

def main():
    dataset = load_txt(
        os.path.join(
            os.path.pardir, "data", "boom", "misc", "boom-design-baseline.txt"
        )
    )
    design_space = load_design_space()
    ppa_models = load_model()
    pred_ipc, pred_power, pred_area = [], [], []
    ipc_reward, power_reward ,area_reward = [], [], []
    for data in dataset:
        ipc, power, area = design_space.evaluate_microarchitecture(
            configs,
            data.astype(int),
            '1'
        )
        pred_ipc.append(ipc)
        pred_power.append(power)
        pred_area.append(area)
        ipc = ppa_models["ipc"].predict(
            np.expand_dims(
                np.concatenate((data, [ipc])),
                axis=0
            )
        )
        power = ppa_models["power"].predict(
            np.expand_dims(
                np.concatenate((data, [power])),
                axis=0
            )
        )
        area = ppa_models["area"].predict(
            np.expand_dims(
                np.concatenate((data, [area])),
                axis=0
            )
        )
        # NOTICE: Refer to `dse/env/boom/design_env.py`
        ipc = 2 * ipc
        power = 2 * 10 * power
        area = 0.5 * 1e-6 * area
        ipc_reward.append(ipc)
        power_reward.append(-power)
        area_reward.append(-area)
    for idx in range(dataset.shape[0]):
        print("[INFO]: baseline: {}, IPC: {}, power: {}, area: {}".format(
                dataset[idx],
                pred_ipc[idx],
                pred_power[idx],
                pred_area[idx]
            )
        )
        print("[INFO]: IPC reward: {}, power reward: {}, area reward: {}".format(
                ipc_reward[idx],
                power_reward[idx],
                area_reward[idx]
            )
        )


if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    configs["logger"] = None
    main()