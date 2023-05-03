# Author: baichen318@gmail.com


import os
import sys
import torch
import random
import socket
import argparse
import numpy as np
import torch.nn as nn
import scipy.stats as stats
import torch.utils.data as data
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from collections import OrderedDict
from sklearn.model_selection import KFold
from utils.utils import get_configs, load_txt, \
    write_txt, mkdir, info, assert_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib


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

    def __init__(self, design, dataset, dims_of_state):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.dims_of_state = dims_of_state
        self.design_space = load_design_space()
        self.embedding = self.design_space.embedding_dims
        if "BOOM" in design:
            self.perf_feature = np.concatenate((
                    # embedding
                    dataset[:, :self.embedding],
                    # statistics
                    dataset[:, self.embedding + 3:self.embedding + 3 + 9],
                    # pred. PPA
                    np.expand_dims(dataset[:, -3], axis=1)
                ),
                axis=1
            )
            self.perf_gt = dataset[:, self.embedding]
            self.power_feature = np.concatenate((
                    # embedding
                    dataset[:, :self.embedding],
                    # statistics
                    dataset[:, self.embedding + 3:self.embedding + 3 + 9],
                    # pred. PPA
                    np.expand_dims(dataset[:, -2], axis=1)
                ),
                axis=1
            )
            self.power_gt = dataset[:, self.embedding + 1]
            self.area_feature = np.concatenate((
                    # embedding
                    dataset[:, :self.embedding],
                    # statistics
                    dataset[:, self.embedding + 3:self.embedding + 3 + 9],
                    # pred. PPA
                    np.expand_dims(dataset[:, -1], axis=1)
                ),
                axis=1
            )
            self.area_gt = dataset[:, self.embedding + 2]
        else:
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
    def __init__(self, metric, decode_width=None):
        super(CalibModel, self).__init__()
        self.metric = metric
        self.decode_width = decode_width
        self.model = self.init_xgb()
        self.mae = None
        self.mse = None
        self.mape = None
        self.kendall_tau = None

    def init_xgb(self):
        if self.metric == "perf":
            return XGBRegressor(
                max_depth=6,
                gamma=0.0000001,
                min_child_weight=1,
                subsample=1.0,
                eta=0.1,
                reg_alpha=0,
                reg_lambda=0.1,
                booster="gbtree",
                objective="reg:squaredlogerror",
                eval_metric="mae",
                n_jobs=-1
            )
        elif self.metric == "power":
            return XGBRegressor(
                max_depth=6,
                gamma=0,
                min_child_weight=1,
                subsample=1.0,
                eta=0.11,
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
                max_depth=6,
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
        if self.decode_width is not None:
            output_path = os.path.join(
                rl_explorer_root,
                configs["ppa-model"],
                self.decode_width
            )
        else:
            output_path = os.path.join(
                rl_explorer_root,
                configs["ppa-model"]
            )
        mkdir(output_path)
        if "BOOM" in configs["design"]:
            name = "boom"
        else:
            assert "Rocket" == configs["design"], \
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
            msg = "{} ".format(metric)
            for idx in self.index:
                try:
                    msg += "{}: {} ".format(idx, self.stats[metric][idx][-1])
                except IndexError:
                    pass
            info(msg)

    def summary(self):
        for metric in self.metrics:
            avg = []
            for idx in self.index:
                avg.append(np.average(self.stats[metric][idx]))
            info("summary: {} mae: {}, mse: {}, mape: {}, kendall tau: {}".format(
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
            choices=["simulation", "calib", "ablation-study", "validate"],
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
    global Simulator
    if "BOOM" in configs["algo"]["design"]:
        from dse.env.boom.design_space import parse_design_space
        from simulation.boom.simulation import Gem5Wrapper as BOOMGem5Wrapper
        Simulator = BOOMGem5Wrapper
    else:
        assert configs["algo"]["design"] == "Rocket", \
            assert_error("{} is not supported.".format(configs["design"]))
        from dse.env.rocket.design_space import parse_design_space
        from simulation.rocket.simulation import Gem5Wrapper as RocketGem5Wrapper
        Simulator = RocketGem5Wrapper
    return parse_design_space(configs)


def split_dataset(dataset):
    # NOTICE: we omit the rest of groups
    if configs["decode_width"] is not None:
        dataset = dataset[np.where(dataset[:, 5] == int(configs["decode_width"]))[0]]
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
            configs["design"],
            dataset[train],
            len(design_space.descriptions[configs["design"]].keys())
        )
        test_dataset = Dataset(
            configs["design"],
            dataset[test],
            len(design_space.descriptions[configs["design"]].keys())
        )
        for metric in Dataset.metrics:
            model = CalibModel(
                metric,
                decode_width=configs["decode_width"]
            )
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
            # visualize(
            #     metric,
            #     test_dataset,
            #     model,
            #     save_path=os.path.join(
            #         configs["vis-root"],
            #         "{}-{}.jpg".format(
            #             metric,
            #             fold
            #         )
            #     )
            # )
        fold += 1
    stats.summary()
    if args.save:
        all_dataset = Dataset(
            configs["design"],
            dataset[range(dataset.shape[0])],
            len(design_space.descriptions[configs["design"]].keys())
        )
        for metric in Dataset.metrics:
            model = CalibModel(
                metric,
                decode_width=configs["decode_width"]
            )
            all_feature, all_gt = getattr(
                all_dataset,
                "get_{}_dataset".format(metric)
            )()
            model.fit(all_feature, all_gt)
            model.predict(all_feature, all_gt)
            stats.update(model)
            model.save()
            stats.summary()
            visualize(metric, all_dataset, model)


def ablation_study_calib_xgboost(design_space, dataset):

    # we make a pertubation
    random.seed(2022)
    n_samples = dataset.shape[0]
    idx = random.sample(range(n_samples), k=n_samples)
    dataset = dataset[idx]

    partition = round(0.2 * (n_samples))
    train_dataset = dataset[partition:, :]
    test_dataset = dataset[:partition, :]
    n_samples = train_dataset.shape[0]

    ratio = [i for i in np.arange(1, 10, 0.5)]
    for _ratio in ratio:
        _ratio = _ratio / 10
        info("current ratio: {}".format(_ratio))
        stats = Stats(Dataset.metrics)
        _train_dataset = Dataset(
            configs["design"],
            train_dataset[:round(n_samples * _ratio), :],
            len(design_space.descriptions[configs["design"]].keys())
        )
        _test_dataset = Dataset(
            configs["design"],
            test_dataset,
            len(design_space.descriptions[configs["design"]].keys())
        )
        for metric in Dataset.metrics:
            model = CalibModel(metric)
            train_feature, train_gt = getattr(
                _train_dataset,
                "get_{}_dataset".format(metric)
            )()
            test_feature, test_gt = getattr(
                _test_dataset,
                "get_{}_dataset".format(metric)
            )()
            model.fit(train_feature, train_gt)
            model.predict(test_feature, test_gt)
            stats.update(model)
        stats.show_current_status()


def adjust_boom_data(design_space, data):
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


def adjust_rocket_data(design_space, data):
    """
        Re-cycle from DAC to ICCAD
        BTB: data[0],
        R. I-Cache: data[1],
        FPU: data[2],
        mulDiv: data[3],
        useVM: data[4],
        R. D-Cache: data[5],
    """
    data[0] += 1
    data[1] += 1
    data[2] += 1
    data[3] += 1
    data[4] += 1
    data[5] += 1
    return data


def adjust_data(design, design_space, data, choice=True):
    if choice:
        if "BOOM" in design:
            return adjust_boom_data(design_space, data)
        else:
            assert "Rocket" in design
            return adjust_rocket_data(design_space, data)
    else:
        return data


def generate_simulation_dataset():
    dataset = load_dataset(
        os.path.join(configs["env"]["calib"]["dataset"])
    )
    target_dataset = os.path.join(
        os.path.dirname(configs["env"]["calib"]["dataset"]),
        os.path.splitext(
            os.path.basename(configs["env"]["calib"]["dataset"])
        )[0] + "-{}.txt".format(socket.gethostname())
    )
    design_space = load_design_space()
    # construct pre-generated dataset
    new_dataset = []

    for data in dataset:
        # data = adjust_data(configs["design"], design_space, data, choice=True)
        manager = Simulator(configs, design_space, np.int64(data[:-3]), 1)
        perf, stats = manager.evaluate_perf()
        power, area = manager.evaluate_power_and_area()
        _stats = []
        for k, v in stats.items():
            _stats.append(v)
        new_dataset.append(
            np.insert(
                data,
                len(data),
                values=np.array(_stats + [perf, power, area * 1e6]),
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


def load_calibrate_ppa_models(design, decode_width=None):
    if decode_width is not None:
        ppa_model_root = os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            configs["ppa-model"],
            decode_width
        )
    else:
        ppa_model_root = os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            configs["ppa-model"]
        )
    if "BOOM" in design:
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
    else:
        assert "Rocket" == design, \
            "[ERROR]: {} is unsupported.".format(design)
        perf_root = os.path.join(
            ppa_model_root,
            "rocket-perf.pt"
        )
        power_root = os.path.join(
            ppa_model_root,
            "rocket-power.pt"
        )
        area_root = os.path.join(
            ppa_model_root,
            "rocket-area.pt"
        )
    perf_model = joblib.load(perf_root)
    power_model = joblib.load(power_root)
    area_model = joblib.load(area_root)
    return perf_model, power_model, area_model


def abalation_study():
    design_space = load_design_space()
    dataset = load_txt(
        os.path.join(
            rl_explorer_root,
            configs["dataset"]
        ),
        fmt=float
    )
    ablation_study_calib_xgboost(design_space, dataset)


def validate():
    design_space = load_design_space()
    dataset = load_txt(
        os.path.join(
            rl_explorer_root,
            configs["dataset"]
        ),
        fmt=float
    )
    if configs["decode_width"] is not None:
        dataset = dataset[np.where(dataset[:, 5] == int(configs["decode_width"]))[0]]
    all_dataset = Dataset(
        configs["design"],
        dataset[range(dataset.shape[0])],
        len(design_space.descriptions[configs["design"]].keys())
    )
    lightweight_ppa_models = list(
        load_calibrate_ppa_models(configs["design"], configs["decode_width"])
    )
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
    elif args.mode == "ablation-study":
        abalation_study()
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
    # a tricy to implement `Gem5Wrapper`
    Simulator = None
    # if "BOOM" in configs["design"]:
    #     decode_width = configs["design"].split(' ')[0].split('-')[0]
    #     configs["decode_width"] = decode_width
    # else:
    #     configs["decode_width"] = None
    configs["decode_width"] = None
    main()
