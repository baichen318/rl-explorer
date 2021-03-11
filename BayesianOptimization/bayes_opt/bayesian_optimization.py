import warnings
from math import ceil, floor
import numpy as np
from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, acq_max, ensure_rng

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
#sys.path.append('./Platypus/platypus')
#from platypus import NSGAII, MOEAD, Problem, Real, SPEA2, NSGAIII, Solution, InjectedPopulation, Archive

#from sklearn.externals import joblib # for scikit-learn 0.21-
import joblib  # for scikit-learn 0.21+


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None, job_reuse=False):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        
        self._key = []
        self._key_bounds = {}
        for key in pbounds:
            if isinstance(pbounds[key], int) == True:
                self._key_bounds[key] = pbounds[key]
                pbounds[key] = (-0.5, pbounds[key] - 0.5)
                self._key.append(key)
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        if job_reuse == True:
            # TODO: load pretrained model
            # self._gp = joblib.load('./adder/synthesis/model/gp.pkl')
            pass
        else:
        	self._gp = GaussianProcessRegressor(
            	kernel=Matern(nu=2.5),
            	alpha=1e-6,
            	normalize_y=True,
            	n_restarts_optimizer=5,
            	random_state=self._random_state,
        	)

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self.sol_processing_v2(self._space.array_to_params(self._space.random_sample()), utility_function)
        
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            dim=self._space.dim,
            random_state=self._random_state
        )

        return self.sol_processing_v2(self._space.array_to_params(suggestion), utility_function)

    def savegp(self,):
        # TODO: `savegp`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
        if not os.path.exists('./adder/synthesis/model/'):
            os.makedirs('./adder/synthesis/model/')
        joblib.dump(self._gp,'./adder/synthesis/model/gp.pkl')

    def predict(self,test_params):
        """Just for testing the GP prediction without std of one point"""
        """Notice! The self._space.params at least has one data point"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
        x_array = self._space._as_array(test_params)   
        return self._gp.predict(x_array.reshape(1, -1))

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1

            self.probe(x_probe, lazy=False)

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)

    # This is the v1 version to process the solution to mixed integer non-linear programming
    # The main principle is to compare the nearest integers in the neighborhood of the original float solution
    def sol_processing(self, suggestion, utility_function):
        equal_key = {}
        sol_candidate = []
        target_candidate = []
        temp = {}
        for key in self._key:
            if abs(floor(suggestion[key])-suggestion[key]) > abs(ceil(suggestion[key])-suggestion[key]):
                suggestion[key] = ceil(suggestion[key])
            elif abs(floor(suggestion[key])-suggestion[key]) < abs(ceil(suggestion[key])-suggestion[key]):
                suggestion[key] = floor(suggestion[key])
            else:
                equal_key[key] = (floor(suggestion[key]), ceil(suggestion[key]))
        if len(equal_key) != 0 :
            for key in equal_key:
                temp = suggestion
                temp[key] = equal_key[key][0]
                sol_candidate.append(temp)
                temp[key] = equal_key[key][1]
                sol_candidate.append(temp)
            for candidate in sol_candidate:
                can_array = self._space._as_array(candidate)
                acc_value = utility_function.utility(can_array.reshape((1,-1)), gp=self._gp, y_max=self._space.target.max())
                target_candidate.append(acc_value[0])
            index = target_candidate.index(max(target_candidate))
            suggestion = sol_candidate[index]
        return suggestion

    # This is the robust v1 version to process the solution to mixed integer non-linear programming
    # The main principle is to compare the nearest integers in the neighborhood of the original float solution
    # More boundary cases are considered in this version, which are not contemplated in rough v1 version.
    def sol_processing(self, suggestion, utility_function):
        if len(self._space) == 0: 
            for key in self._key:
                suggestion[key] = np.random.randint(0, self._key_bounds[key], 1)[0]
            return suggestion
        equal_key = {}
        sol_candidate = []
        target_candidate = []
        temp = {}
        for key in self._key:
            if suggestion[key] == -0.5:
                suggestion[key] = 0
                continue
            if suggestion[key] == self._key_bounds[key]-0.5:
                suggestion[key] = self._key_bounds[key]-1
                continue
            if abs(floor(suggestion[key])-suggestion[key]) == 0.0 or abs(ceil(suggestion[key])-suggestion[key]) == 0.0:
                continue
            if abs(floor(suggestion[key])-suggestion[key]) > abs(ceil(suggestion[key])-suggestion[key]):
                suggestion[key] = ceil(suggestion[key])
            elif abs(floor(suggestion[key])-suggestion[key]) < abs(ceil(suggestion[key])-suggestion[key]):
                suggestion[key] = floor(suggestion[key])
            else:
                equal_key[key] = (floor(suggestion[key]), ceil(suggestion[key]))
        if len(equal_key) != 0:
            for key in equal_key:
                temp = suggestion
                temp[key] = equal_key[key][0]
                sol_candidate.append(temp)
                temp[key] = equal_key[key][1]
                sol_candidate.append(temp)
            for candidate in sol_candidate:
                can_array = self._space._as_array(candidate)
                acc_value = utility_function.utility(can_array.reshape((1,-1)), gp=self._gp, y_max=self._space.target.max())
                target_candidate.append(acc_value[0])
            index = target_candidate.index(max(target_candidate))
            suggestion = sol_candidate[index]
        return suggestion

    # This is the v2 version to process the solution to mixed integer non-linear programming
    # The main principle is to enumerate all possible integer solutions and use the acquisition function to find the best one
    def sol_processing_v2(self, suggestion, utility_function):
        if len(self._space) == 0: 
            for key in self._key:
                suggestion[key] = np.random.randint(0, self._key_bounds[key], 1)[0]
            return suggestion
        sol_candidate = []
        target_candidate = []
        temp = {}
        for key in self._key:
            temp = suggestion
            for i in range(self._key_bounds[key]-1):
                temp[key] = i
                sol_candidate.append(temp)
        for candidate in sol_candidate:
            can_array = self._space._as_array(candidate)
            acc_value = utility_function.utility(can_array.reshape((1,-1)), gp=self._gp, y_max=self._space.target.max())
            target_candidate.append(acc_value[0])
            index = target_candidate.index(max(target_candidate))
            suggestion = sol_candidate[index]
        return suggestion
