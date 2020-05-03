import os
import torch
import botorch
import numpy as np


class UniformProblem:
    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim

        self.real_lb = problem.lb
        self.real_ub = problem.ub

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        if problem.xopt is not None:
            self.xopt = (problem.xopt - problem.lb) / (problem.ub - problem.lb)
        else:
            self.xopt = problem.xopt

        self.yopt = problem.yopt

        self.real_cf = problem.cf
        self.set_cf()

    def __call__(self, x):
        x = np.atleast_2d(x)

        # map x back to original space
        x = x * (self.real_ub - self.real_lb) + self.real_lb

        return self.problem(x)

    def set_cf(self):
        if self.real_cf is None:
            self.cf = None
            return

        def cf_wrapper(x):
            x = np.atleast_2d(x)

            # map x back to original space
            x = x * (self.real_ub - self.real_lb) + self.real_lb

            return self.real_cf(x)

        self.cf = cf_wrapper


class TorchProblem:
    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim

        self.lb = torch.from_numpy(problem.lb)
        self.ub = torch.from_numpy(problem.ub)

        if problem.xopt is not None:
            self.xopt = torch.from_numpy(problem.xopt)
        else:
            self.xopt = problem.xopt

        self.yopt = torch.from_numpy(problem.yopt)

        if self.problem.cf is not None:
            def cf(x):
                if not isinstance(x, np.ndarray):
                    x = x.numpy()
                return self.problem.cf(x)
            self.cf = cf
        else:
            self.cf = None

    def __call__(self, x):
        return torch.from_numpy(self.problem(x.numpy()))


class LowerConfidenceBound(botorch.acquisition.UpperConfidenceBound):
    def __init__(self, model, beta, objective=None, maximize=True):
        super().__init__(model=model, beta=beta, objective=objective,
                         maximize=maximize)

    def forward(self, X):
        return -super().forward(X)


class PosteriorMeanMinimize(botorch.acquisition.PosteriorMean):
    def __init__(self, model, objective=None):
        super().__init__(model=model, objective=objective)

    def forward(self, X):
        return -super().forward(X)


def acq_func_getter(name, model, train_y, problem_bounds):
    acq_params = {'maximize': False}

    if name == 'EI':
        acq_params['best_f'] = train_y.min()
        acq_func = botorch.acquisition.ExpectedImprovement

    elif name == 'UCB':
        acq_func = LowerConfidenceBound

        (train_x, *_) = model.train_inputs
        t = train_y.numel()
        delta = 0.01
        D = train_x.shape[1]

        acq_params['beta'] = 2 * np.log(D * t**2 * np.pi**2 / (6 * delta))

    elif name == 'PI':
        acq_func = botorch.acquisition.ProbabilityOfImprovement
        acq_params['best_f'] = train_y.min()

    elif name == 'mean':
        acq_func = PosteriorMeanMinimize
        acq_params = {}

    return acq_func(model, **acq_params)


def generate_save_filename(problem_name, mean_name, acq_name, run_no,
                           problem_params={}, results_dir='results'):
    # append dim if different from default
    if 'd' in problem_params:
        pn = f'{problem_name:s}{problem_params["d"]:d}'
    else:
        pn = problem_name

    fname = f'{pn:s}_{mean_name:s}_{acq_name:s}_{run_no:03d}.pt'

    return os.path.join(results_dir, fname)


def generate_data_filename(problem_name, run_no, problem_params={},
                           data_dir='data'):
    # append dim if different from default
    if 'd' in problem_params:
        pn = f'{problem_name:s}{problem_params["d"]:d}'
    else:
        pn = problem_name

    fname = f'{pn:s}_{run_no:03d}.pt'

    return os.path.join(data_dir, fname)
