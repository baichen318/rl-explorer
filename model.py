# Author: baichen318@gmail.com

import sys
import os
import numpy as np
from xgboost import XGBRegressor
import sklearn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib

from collections import OrderedDict
from multiprocessing import Process, Queue
from space import parse_design_space
from search import sa_search
from util import parse_args, get_configs, read_csv, read_csv_v2, if_exist, \
    calc_mape, point2knob, knob2point, create_logger, is_pow2, mkdir, \
    execute, mse, r2, mape, write_csv
from vis import handle_v2
from exception import UnDefinedException
from vlsi.vlsi import offline_vlsi_flow_v2

methods = [
    "lr",
    "lasso",
    "ridge",
    "svr",
    "lsvr",
    "xgb",
    "rf",
    "ab",
    "gb",
    "bg",
    "dnn-gp"
]

def split_dataset(dataset):
    # split dataset into x label & y label
    # dataset: `np.array`
    x = []
    y = []
    for data in dataset:
        x.append(data[0:-2])
        y.append(data[-2:])

    return np.array(x), np.array(y)

def validate(dataset):
    data = []
    for item in dataset:
        _data = []
        for i in item[0].split(' '):
            _data.append(int(i))
        for i in item[-2:]:
            _data.append(float(i))
        data.append(_data)
    data = np.array(data)

    return data

def kFold():
    return KFold(n_splits=10)

def create_model(method):
    if method == "lr":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(n_jobs=-1)
    elif method == "lasso":
        from sklearn.linear_model import Lasso
        model = Lasso()
    elif method == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge()
    elif method == "svr":
        from sklearn.svm import SVR
        model = MultiOutputRegressor(
            SVR()
        )
    elif method == "lsvr":
        from sklearn.svm import LinearSVR
        model = MultiOutputRegressor(
            LinearSVR()
        )
    elif method == "xgb":
        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100
            )
        )
    elif method == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = MultiOutputRegressor(
            RandomForestRegressor()
        )
    elif method == "ab":
        from sklearn.ensemble import AdaBoostRegressor
        model = MultiOutputRegressor(
            AdaBoostRegressor()
        )
    elif method == "gb":
        from sklearn.ensemble import GradientBoostingRegressor
        model = MultiOutputRegressor(
            GradientBoostingRegressor()
        )
    elif method == "bg":
        from sklearn.ensemble import BaggingRegressor
        model = MultiOutputRegressor(
            BaggingRegressor()
        )
    elif method == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, \
            ExpSineSquared , DotProduct, ConstantKernel
        # kernel = ConstantKernel(1.0, (1e-3, 1000)) * \
        #     RationalQuadratic(1.0, 1.0, (1e-5, 1e5), (1e-5, 1e5)) + WhiteKernel(0, (1e-3, 1000))
        kernel = ConstantKernel(1.0, (1e-3, 1000)) * \
            DotProduct(1.0,(1e-5, 1e5)) + WhiteKernel(0.1, (1e-3, 1000))
        model = MultiOutputRegressor(
            GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b')
        )
    elif method == "br":
        from sklearn.linear_model import BayesianRidge
        model = MultiOutputRegressor(
            BayesianRidge()
        )
    elif method == "dnn-gp":
        from bayes_gp import BayesianOptimization
        model = BayesianOptimization(configs)
    else:
        raise UnDefinedException("%s not supported" % method)
    return model

def regression(method, dataset, index):
    def analysis(model, gt, predict):
        # coefficients
        if hasattr(model, "coef_"):
            print("[INFO]: coefficients of LinearRegression:\n", model.coef_)
        elif hasattr(model, "feature_importances_"):
            print("[INFO]: significance of features:\n", model.feature_importances_)

        mse_l = mse(gt[:, 0], predict[:, 0])
        mse_p = mse(gt[:, 1], predict[:, 1])
        r2_l = r2(gt[:, 0], predict[:, 0])
        r2_p = r2(gt[:, 1], predict[:, 1])
        mape_l = mape(gt[:, 0], predict[:, 0])
        mape_p = mape(gt[:, 1], predict[:, 1])
        logger.info("[INFO]: MSE (latency): %.8f, MSE (power): %.8f" % (mse_l, mse_p))
        logger.info("[INFO]: R2 (latency): %.8f, R2 (power): %.8f" % (r2_l, r2_p))
        logger.info("[INFO]: MAPE (latency): %.8f, MAPE (power): %.8f" % (mape_l, mape_p))

        return mse_l, mse_p, r2_l, r2_p, mape_l, mape_p

    model = create_model(method)
    # MSE, R2, MAPE for latency and power
    avg_metrics = [0 for i in range(6)]
    perf = float('inf')
    cnt = 0
    for train_index, test_index in index:
        logger.info("train:\n%s" % str(train_index))
        logger.info("test:\n%s" % str(test_index))
        x_train, y_train = split_dataset(dataset[train_index])
        x_test, y_test = split_dataset(dataset[test_index])
        # train
        model.fit(x_train, y_train)
        # predict
        predict = model.predict(x_test)
        metrics = analysis(model, y_test, predict)
        for i in range(len(avg_metrics)):
            avg_metrics[i] += metrics[i]
        cnt += 1
        if (0.5 * metrics[4] + 0.5 * metrics[5]) < perf:
            perf = 0.5 * metrics[4] + 0.5 * metrics[5]
            # achieve the average minimum in one round
            min_mape_l, min_mape_p = metrics[4], metrics[5]
            joblib.dump(
                model,
                os.path.join(
                    configs["model-output-path"],
                    configs["model"] + ".mdl"
                )
            )
    msg = "[INFO]: achieve the best performance: MAPE (latency): %.8f " %  min_mape_l + \
        "MAPE (power): %.8f in one round. " % min_mape_p + \
        "Average MAPE (latency): %.8f, " % avg_metrics[4] + \
        "average MAPE (power): %.8f, " % avg_metrics[5] + \
        "average R2 (latency): %.8f, " % avg_metrics[2] + \
        "average R2 (power): %.8f" % avg_metrics[3]
    logger.info(msg)

    # test
    try:
        model = joblib.load(
            os.path.join(
                configs["model-output-path"],
                configs["model"] + ".mdl"
            )
        )
    except Exception:
        pass
    heap = sa_search(model, design_space, logger, top_k=50,
        n_iter=5000, early_stop=3000, parallel_size=128, log_interval=50)
    # saving results
    mkdir(configs["rpt-output-path"])
    write_csv(
        os.path.join(
            configs["rpt-output-path"],
            configs["model"] + '-prediction.rpt'
        ),
        heap,
        mode='a'
    )
    # get the metris
    perf = []
    for (hv, p) in heap:
        if hv != -1:
            perf.append(model.predict(p.reshape(1, -1)).ravel())
    assert len(perf) > 0, "[ERROR]: SA cannot find a good point"
    # visualize
    handle_v2(
        perf,
        configs["model"] + '-prediction',
        configs
    )

    return heap

def verify(heap):
    """
        `heap`: <list> (<tuple> in <list>), specifically,
        <tuple> is (<int>, <list>) or (hv, configurations)
        We only extract the first 3 points to verify
    """
    dataset = []
    for (hv, p) in heap:
        if hv != -1:
            dataset.append(p)
    dataset = np.array(dataset)[:3]
    offline_vlsi_flow_v2(dataset, configs)

def handle():
    dataset, title = read_csv_v2(configs["dataset-output-path"])
    dataset = validate(dataset)
    kf = kFold()
    index = kf.split(dataset)

    if configs["model"] in methods:
        heap = regression(configs["model"], dataset, index)
    else:
        raise UnDefinedException("%s undefined" % configs["model"])

    # verify(heap)

if __name__ == "__main__":
    argv = parse_args()
    configs = get_configs(argv.configs)
    design_space = parse_design_space(
        configs["design-space"]
    )
    logger = create_logger("logs", "model")
    handle()
