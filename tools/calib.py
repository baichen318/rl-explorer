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
    os.path.join(os.path.dirname(__file__), os.path.pardir, "utils")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "simulation")
)
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from xgboost import XGBRegressor
from utils import get_configs, load_txt, \
    write_txt, mkdir, info
from simulation.boom.simulation import Gem5Wrapper


markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]
colors = [
    'c', 'b', 'g', 'r', 'm', 'y', 'k', # 'w'
]


class Dataset(object):

    metrics = ["perf", "power", "area"]

    def __init__(self, dataset, dims_of_state):
        super(Dataset, self).__init__()
        self.perf_feature = dataset[
            :, [i for i in range(dims_of_state)] + [-3]
        ]
        self.perf_gt = dataset[
            :, dims_of_state
        ]
        self.power_feature = dataset[
            :, [i for i in range(dims_of_state)] + [-2]
        ]
        self.power_gt = dataset[
            :, dims_of_state + 1
        ]
        self.area_feature = dataset[
            :, [i for i in range(dims_of_state)] + [-1]
        ]
        self.area_gt = dataset[
            :, dims_of_state + 2
        ]

    def get_perf_dataset(self):
        return self.perf_feature, self.perf_gt

    def get_power_dataset(self):
        return self.power_feature, self.power_gt

    def get_area_dataset(self):
        return self.area_feature, self.area_gt


class CalibModel(object):
    def __init__(self, metric):
        super(CalibModel, self).__init__()
        self.metric = metric
        self.model = self.init_xgb()
        self.mae = None
        self.mse = None
        self.mape = None
        self.kendall_tau = None

    def init_xgb(self):
        if self.metric == "perf":
            return XGBRegressor(
                max_depth=5,
                gamma=0.0000001,
                min_child_weight=1,
                subsample=1.0,
                eta=0.25,
                reg_alpha=0,
                reg_lambda=0.1,
                booster="gbtree",
                objective="reg:squarederror",
                eval_metric="mae",
                n_jobs=-1
            )
        elif self.metric == "power":
            return XGBRegressor(
                max_depth=3,
                gamma=0.0000001,
                min_child_weight=1,
                subsample=1.0,
                eta=0.25,
                reg_alpha=0,
                reg_lambda=0.1,
                booster="gbtree",
                objective="reg:squarederror",
                eval_metric="mae",
                n_jobs=-1
            )
        else:
            assert self.metric == "area", \
                "[ERROR]: {} is not supported.".format(metric)
            return XGBRegressor(
                max_depth=3,
                gamma=0.0000001,
                min_child_weight=1,
                subsample=1.0,
                eta=0.25,
                reg_alpha=0,
                reg_lambda=0.1,
                booster="gbtree",
                objective="reg:squarederror",
                eval_metric="mae",
                n_jobs=-1
            )

    def fit(self, train_feature, train_gt):
        self.model.fit(train_feature, train_gt)

    def predict(self, test_feature, test_gt):
        pred = self.model.predict(test_feature)
        self.mae = mean_absolute_error(test_gt, pred)
        self.mse = mean_squared_error(test_gt, pred)
        self.mape = mean_absolute_percentage_error(test_gt, pred)
        self.kendall_tau, _ = stats.kendalltau(pred, test_gt)
        return pred

    def save(self):
        output_path = os.path.join(
            rl_explorer_root,
            configs["ppa-model"]
        )
        mkdir(output_path)
        if "BOOM" in configs["design"]:
            name = "boom"
        else:
            assert "rocket" == configs["design"], \
                    "[ERROR]: {} is not supported.".format(configs["design"])
            name = "rocket"
        output_path = os.path.join(
            output_path,
            name + '-' + self.metric + ".pt"
        )
        joblib.dump(
            self.model,
            output_path
        )
        info("save {}.".format(output_path))


class Stats(object):
    def __init__(self, metrics):
        super(Stats, self).__init__()
        self.metrics = metrics
        self.index = ["mae", "mse", "mape", "kendall_tau"]
        self.stats = self.init_stats()

    def init_stats(self):
        stats = OrderedDict()
        for metric in self.metrics:
            stats[metric] = OrderedDict()
            for idx in self.index:
                stats[metric][idx] = []
        return stats

    def update(self, model):
        """
            model: <class "CalibModel">
        """
        for idx in self.index:
            self.stats[model.metric][idx].append(
                getattr(model, "{}".format(idx))
            )

    def show_current_status(self):
        for metric in self.metrics:
            for idx in self.index:
                try:
                    info("{} {}: {}".format(
                            metric,
                            idx,
                            self.stats[metric][idx][-1]
                        )
                    )
                except IndexError:
                    pass

    def summary(self):
        for metric in self.metrics:
            avg = []
            for idx in self.index:
                avg.append(np.average(self.stats[metric][idx]))
            info("{} mae: {}, mse: {}, mape: {}, kendall tau: {}".format(
                    metric, avg[0], avg[1], avg[2], avg[3]
                )
            )


def parse_args():
    def initialize_parser(parser):
        parser.add_argument(
            "-c",
            "--configs",
            required=True,
            type=str,
            default="configs.yml",
            help="YAML file to be handled")
        parser.add_argument(
            "-m",
            "--mode",
            required=True,
            type=str,
            default="simulation",
            choices=["simulation", "calib", "validate"],
            help="working mode specification"
        )
        parser.add_argument(
            "-s",
            "--save",
            action="store_true",
            help="model saving specification (used when mode = calib)"
        )
        return parser

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = initialize_parser(parser)
    return parser.parse_args()


def load_dataset(path):
    dataset = load_txt(path, fmt=float)
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
    if "BOOM" in configs["design"]:
        from dse.env.boom.design_space import parse_design_space
    # else:
    #     assert configs["design"] == "rocket", \
    #         "[ERROR]: {} is not supported.".format(configs["design"])
    #     from dse.env.rocket.design_space import parse_design_space
    return parse_design_space(configs)


def split_dataset(dataset):
    # NOTICE: we omit the rest of groups
    kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
    for train, test in kfold.split(dataset):
        yield train, test


def visualize(metric, dataset, model, save_path=None):
    feature, gt = getattr(
        dataset,
        "get_{}_dataset".format(metric)
    )()
    pred = model.predict(feature, gt)
    if save_path:
        prefix_txt_save_path = save_path.split('.')[0]
        gt_txt_save_path = "{}-gt.txt".format(prefix_txt_save_path)
        pred_txt_save_path = "{}-pred.txt".format(prefix_txt_save_path)
        np.savetxt(
            gt_txt_save_path,
            gt
        )
        info("save dataset to {}.".format(gt_txt_save_path))
        np.savetxt(
            pred_txt_save_path,
            pred
        )
        info("save dataset to {}.".format(pred_txt_save_path))
    plt.scatter(
        np.array(pred),
        np.array(gt),
        s=2,
        c=colors[1],
        marker=markers[2],
    )
    plt.axline(
        (0, 0),
        slope=1,
        color=colors[3],
        transform=plt.gca().transAxes
    )
    plt.grid()
    plt.xlabel("prediction")
    plt.ylabel("gt")
    plt.title("{} mape = {} kendall tau = {}".format(
        metric, model.mape, model.kendall_tau)
    )
    if save_path:
        plt.savefig(save_path)
        info("save fig. to {}".format(save_path))
    plt.show()


def calib_xgboost(design_space, dataset):
    stats = Stats(Dataset.metrics)
    fold = 1
    for train, test in split_dataset(dataset):
        train_dataset = Dataset(
            dataset[train],
            len(design_space.descriptions[configs["design"]].keys())
        )
        test_dataset = Dataset(
            dataset[test],
            len(design_space.descriptions[configs["design"]].keys())
        )
        for metric in Dataset.metrics:
            model = CalibModel(metric)
            train_feature, train_gt = getattr(
                train_dataset,
                "get_{}_dataset".format(metric)
            )()
            test_feature, test_gt = getattr(
                test_dataset,
                "get_{}_dataset".format(metric)
            )()
            model.fit(train_feature, train_gt)
            model.predict(test_feature, test_gt)
            stats.update(model)
            stats.show_current_status()
            visualize(
                metric,
                test_dataset,
                model,
                save_path=os.path.join(
                    configs["vis-root"],
                    "{}-{}.jpg".format(
                        metric,
                        fold
                    )
                )
            )
        fold += 1
    stats.summary()
    if args.save:
        all_dataset = Dataset(
            dataset[range(dataset.shape[0])],
            len(design_space.descriptions[configs["design"]].keys())
        )
        for metric in Dataset.metrics:
            model = CalibModel(metric)
            all_feature, all_gt = getattr(
                all_dataset,
                "get_{}_dataset".format(metric)
            )()
            model.fit(all_feature, all_gt)
            model.predict(all_feature, all_gt)
            model.save()
            stats.summary()
            visualize(metric, all_dataset, model)


def adjust_data(design_space, data):
    """
        Re-cycle from DAC to ICCAD
        branchPredictor: data[0],
        fetchWidth: data[1],
        IFU: data[2],
        maxBrCount: data[3],
        ROB: data[5],
        PRF: data[6],
        ISU: data[7],
        LSU: data[8],
        D-Cache: data[9]
    """
    def get_data_by_physical_val(i):
        for k, v in design_space.components_mappings[
                design_space.components[i]
            ].items():
            if len(v) == 1 and v[0] == data[i]:
                return int(k)

    isu = [
        # numEntries of IQT_MEM IQT_INT IQT_FP
        [8, 8, 8],
        [12, 20, 16],
        [16, 32, 24],
        [24, 40, 32],
        [10, 14, 12],
        [14, 26, 20],
        [20, 36, 28],
        [4, 4, 4],
        [6, 6, 6],
        [12, 28, 20],
        [14, 30, 22],
        [26, 42, 34]
    ]

    dcache = [
        # dcache.nMSHRs dcache.nTLBWays
        [2, 8],
        [4, 16],
        [8, 32],
        [16, 48],
        [4, 8]
    ]

    data[0] +=1
    data[1] = get_data_by_physical_val(1)
    data[2] += 1
    data[3] = get_data_by_physical_val(3)
    data[5] = get_data_by_physical_val(5)
    data[6] += 1
    params = isu[int(data[7])]
    for k, v in design_space.components_mappings[
            design_space.components[7]
        ].items():
        if params[0] == v[1] and \
            params[1] == v[4] and \
            params[2] == v[7] and \
            data[4] == v[3]:
            data[7] = int(k)
            break
    data[8] += 1
    params = dcache[int(data[9])]
    for k, v in design_space.components_mappings[
            design_space.components[9]
        ].items():
        if params[0] == v[1] and \
            params[1] == v[2] and \
            design_space.components_mappings[
                design_space.components[1]
            ][int(data[1])][0] == v[0]:
            data[9] = int(k)
            break
    return data


def generate_simulation_dataset():
    dataset = load_dataset(
        os.path.join(
            rl_explorer_root,
            configs["dataset"]
        )
    )
    target_dataset = os.path.join(
        rl_explorer_root,
        os.path.dirname(configs["dataset"]),
        os.path.splitext(
            os.path.basename(configs["dataset"])
        )[0] + "-extend.txt"
    )
    design_space = load_design_space()
    # construct pre-generated dataset
    new_dataset = []

    for data in dataset:
        data = adjust_data(design_space, data)
        manager = Gem5Wrapper(configs, design_space, np.int64(data[:-3]), 1)
        perf = manager.evaluate_perf()
        power, area = manager.evaluate_power_and_area()
        new_dataset.append(
            np.insert(
                data,
                len(data),
                values=np.array([perf, power, area * 1e6]),
                axis=0
            )
        )
        _new_dataset = np.array(new_dataset)
        write_txt(
            target_dataset,
            _new_dataset,
            fmt="%f"
        )


def calib_dataset():
    """
        we use xgboost to calibrate the model
    """
    design_space = load_design_space()
    dataset = load_txt(
        os.path.join(
            rl_explorer_root,
            configs["dataset"]
        ),
        fmt=float
    )
    calib_xgboost(design_space, dataset)


def load_calibrate_ppa_models():
    ppa_model_root = os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        configs["ppa-model"],
    )
    perf_root = os.path.join(
        ppa_model_root,
        "boom-perf.pt"
    )
    power_root = os.path.join(
        ppa_model_root,
        "boom-power.pt"
    )
    area_root = os.path.join(
        ppa_model_root,
        "boom-area.pt"
    )
    perf_model = joblib.load(perf_root)
    power_model = joblib.load(power_root)
    area_model = joblib.load(area_root)
    return perf_model, power_model, area_model


def validate():
    design_space = load_design_space()
    dataset = load_txt(
        os.path.join(
            rl_explorer_root,
            configs["dataset"]
        ),
        fmt=float
    )
    all_dataset = Dataset(
        dataset[range(dataset.shape[0])],
        len(design_space.descriptions[configs["design"]].keys())
    )
    lightweight_ppa_models = list(load_calibrate_ppa_models())
    for metric in Dataset.metrics:
        all_feature, all_gt = getattr(
            all_dataset,
            "get_{}_dataset".format(metric)
        )()
        pred = lightweight_ppa_models[Dataset.metrics.index(metric)].predict(all_feature)
        np.savetxt("validate-{}.csv".format(metric), pred)


def main():
    if args.mode == "simulation":
        generate_simulation_dataset()
    elif args.mode == "calib":
        calib_dataset()
    else:
        assert args.mode == "validate", \
            "[ERROR]: {} is not supported.".format(args.mode)
        validate()


if __name__ == '__main__':
    args = parse_args()
    configs = get_configs(args.configs)
    configs["configs"] = args.configs
    rl_explorer_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir
        )
    )
    configs["vis-root"] = os.path.join(
        rl_explorer_root,
        "tools",
        "vis"
    )
    configs["logger"] = None
    main()
