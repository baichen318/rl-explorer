
import os
import numpy as np
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from space import parse_design_space
from sample import RandomSampler
from util import load_dataset, split_dataset, rmse, strflush
from vis import plot_predictions_with_gt
from exception import UnDefinedException

class SurrogateModel(object):
    """
        SurrogateModel: 12 traditional basic ML models
    """
    def __init__(self, method):
        super(SurrogateModel, self).__init__()
        self.method = method
        self.model = self.init()

    def init(self):
        try:
            model = getattr(self, "init_" + self.method)()
        except AttributeError as e:
            raise UnDefinedException(self.method)
        return model

    def init_lr(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(n_jobs=-1)

    def init_lasso(self):
        from sklearn.linear_model import Lasso
        return Lasso(
            alpha=0.1,
            precompute=True,
            max_iter=1000000,
            tol=1e-5
        )

    def init_ridge(self):
        from sklearn.linear_model import Ridge
        return Ridge(tol=1e-6)

    def init_elastic(self):
        from sklearn.linear_model import ElasticNet
        return ElasticNet(
            alpha=0.1,
            precompute=True,
            max_iter=1000000,
            tol=1e-5
        )

    def init_svr(self):
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(
            SVR(
                C=1,
                kernel="linear",
                tol=1e-6
            )
        )

    def init_xgb(self):
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(
            XGBRegressor(
                max_depth=3,
                gamma=0.0001,
                min_child_weight=1,
                subsample=1.0,
                eta=0.3,
                reg_lambda=1.00,
                alpha=0,
                objective='reg:linear',
                n_jobs=-1
            )
        )

    def init_rf(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor 
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=3,
                criterion="mse",
                n_jobs=-1
            )
        )

    def init_ab(self):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(
            AdaBoostRegressor(
                n_estimators=3,
                learning_rate=0.1,
                loss='linear'
            )
        )

    def init_gb(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(
            GradientBoostingRegressor(
                loss='huber',
                learning_rate=0.01,
                n_estimators=100,
            )
        )

    def init_bg(self):
        from sklearn.ensemble import BaggingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(
            BaggingRegressor(
                n_estimators=10,
                bootstrap_features=False
            )
        )

    def init_gp(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, \
            ExpSineSquared , DotProduct, ConstantKernel, Matern
        from sklearn.multioutput import MultiOutputRegressor
        # kernel = ConstantKernel(1.0, (1e-3, 1000)) * \
            # RationalQuadratic(1.0, 1.0, (1e-5, 1e5), (1e-5, 1e5)) + WhiteKernel(0, (1e-3, 1000))
        kernel = ConstantKernel(1.0, (1e-3, 1000)) * \
            DotProduct(1.0,(1e-5, 1e5)) + WhiteKernel(0.1, (1e-3, 1000))
        # kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        return MultiOutputRegressor(
            GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b')
        )

    def init_br(self):
        from sklearn.linear_model import BayesianRidge
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(
            BayesianRidge(
                n_iter=300,
                tol=1e-6,
            )
        )

    def fit(self, x, y):
        """
            x: <numpy.ndarray>
            y: <numpy.ndarray>
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
            x: <numpy.ndarray>
        """
        return self.model.predict(x)


class BayesianOptimization(object):
    """docstring for BayesianOptimization"""
    def __init__(self, configs):
        super(BayesianOptimization, self).__init__()
        self.configs = configs
        # build model
        self.model = SurrogateModel(self.configs["model"])
        self.space = parse_design_space(self.configs["design-space"])
        self.sampler = RandomSampler(self.configs)
        self.dataset = load_dataset(configs["dataset-output-path"])
        self.unsampled = None

    def sample(self, dataset):
        return self.sampler.sample(dataset)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def run(self):
        x, y = [], []
        dataset = self.dataset.copy()
        for i in range(self.configs["max-bo-steps"]):
            dataset, sample = self.sample(dataset)
            _x, _y = split_dataset(sample)
            # add `_x` & `_y` to `x` & `y` respectively
            if len(x) == 0:
                for j in _x:
                    x.append(j)
                x = np.array(x)
                for j in _y:
                    y.append(j)
                y = np.array(y)
            else:
                for j in _x:
                    x = np.insert(x, len(x), j, axis=0)
                for j in _y:
                    y = np.insert(y, len(y), j, axis=0)
            self.fit(x, y)

            __y = self.predict(x)
            msg = "[INFO]: Training Iter %d: RMSE of c.c.: %.8f, " % ((i + 1), rmse(y[:, 0], __y[:, 0])) + \
                "RMSE of power: %.8f on %d train data" % (rmse(y[:, 1], __y[:, 1]), len(x))
            strflush(msg)
            # validate
            __x, __y = split_dataset(dataset)
            ___y = self.predict(__x)
            msg = "[INFO]: Testing Iter %d: RMSE of c.c.: %.8f, " % ((i + 1), rmse(__y[:, 0], ___y[:, 0])) + \
                "RMSE of power: %.8f on %d test data" % (rmse(__y[:, 1], ___y[:, 1]), len(__x))
            strflush(msg)

        self.unsampled = dataset

    def validate(self, logger=None):
        x, y = split_dataset(self.unsampled)
        _y = self.predict(x)
        msg = "[INFO]: RMSE of c.c.: %.8f, " % rmse(y[:, 0], _y[:, 0]) + \
            "RMSE of power: %.8f on %d test data" % (rmse(y[:, 1], _y[:, 1]), len(self.unsampled))
        strflush(msg)

        # visualize
        plot_predictions_with_gt(
            y,
            _y,
            title=self.configs["model"],
            output=self.configs["fig-output-path"]
        )

    def save(self):
        output = os.path.join(
            self.configs["model-output-path"],
            self.configs["model"] + ".mdl"
        )
        joblib.dump(
            self.model,
            output
        )
        msg = "[INFO]: saving model to %s" % output
        strflush(msg)

    def load(self):
        output = os.path.join(
            self.configs["model-output-path"],
            self.configs["model"] + ".mdl"
        )
        self.model = joblib.load(output)
        msg = "[INFO]: loading model from %s" % output
        strflush(msg)
