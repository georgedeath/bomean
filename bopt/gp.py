import torch
import botorch
import gpytorch

from pyDOE2.doe_lhs import lhs
from .mean_functions import MeanZero
from .transforms import Transform_Standardize
from .util import acq_func_getter


class GP_FixedNoise_CustomMean(gpytorch.models.ExactGP,
                               botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_x, train_y,
                 ls_constraint, out_constraint,
                 mean_function=None,
                 noise_size=1e-6):
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            torch.full_like(train_y, noise_size)
        )
        super(GP_FixedNoise_CustomMean, self).__init__(train_x, train_y,
                                                       likelihood)
        base_kernel = gpytorch.kernels.MaternKernel(
            lengthscale_constraint=ls_constraint,
            nu=5 / 2
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_constraint=out_constraint
        )
        if mean_function is None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_function

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_model_restarts(mll, n_restarts=10, verbose=False):
    # only optimise things with bounds
    bounds = []
    for param_name, _, constraint in mll.named_parameters_and_constraints():
        if (constraint is not None) and constraint.enforced:
            lb = constraint.lower_bound
            ub = torch.min(constraint.upper_bound, torch.tensor(1e5))
            bounds.append([param_name, lb, ub, constraint])

    n_dim = len(bounds)

    # Latin hypercube sample
    H = lhs(n_dim, samples=n_restarts, criterion='maximin')
    H = torch.from_numpy(H)

    for i in range(n_dim):
        # rescale
        _, lb, ub, constraint = bounds[i]
        H[:, i] = H[:, i] * (ub - lb) + lb

        # transform the bounds into the 'raw' bound interval that gpytorch uses
        H[:, i] = constraint.inverse_transform(H[:, i])

    # Set the model and likelihood into training model
    mll.train()

    # results storage
    res_loss = torch.zeros(n_restarts)
    res_endpoints = [[[] for _ in range(n_dim)] for _ in range(n_restarts)]

    for res_idx, h in enumerate(H):
        # set the hyperparameters to start the optimisation from
        for i in range(n_dim):
            for param_name, param in mll.named_parameters():
                if param_name == bounds[i][0]:
                    param.data.fill_(h[i])
                    break

        _, resd = botorch.optim.fit.fit_gpytorch_scipy(mll,
                                                       method='L-BFGS-B',
                                                       options={'disp': False},
                                                       track_iterations=False)

        # store the loss and corresponding hyperparams
        res_loss[res_idx] = resd['fopt']
        for i in range(n_dim):
            for param_name, param in mll.named_parameters():
                if param_name == bounds[i][0]:
                    res_endpoints[res_idx][i] = param.data
                    break

        if verbose:
            print(f'Iter {res_idx}: {res_loss[res_idx]:g}')

    # finally, set the best hyperparameters
    argmin = res_loss.argmin()
    for i in range(n_dim):
        for param_name, param in mll.named_parameters():
            if param_name == bounds[i][0]:
                try:
                    param.data[:] = res_endpoints[argmin][i]
                except IndexError:
                    param.data.fill_(res_endpoints[argmin][i])
                break

    return mll


def create_and_fit_GP(train_x, train_y, ls_bounds, out_bounds,
                      mean_func, n_restarts):

    # Kernel lengthscale and variance (output scale) constraints
    ls_constraint = gpytorch.constraints.Interval(*torch.tensor(ls_bounds))
    out_constraint = gpytorch.constraints.Interval(*torch.tensor(out_bounds))

    # instantiate model
    model = GP_FixedNoise_CustomMean(train_x, train_y,
                                     ls_constraint, out_constraint,
                                     mean_function=mean_func)
    likelihood = model.likelihood

    # train it
    model.train()
    model.likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    train_model_restarts(mll, n_restarts=n_restarts)

    return model, likelihood


def perform_BO_iteration(Xtr, Ytr, problem_bounds,
                         ls_bounds=[1e-6, 1e6], out_bounds=[1e-6, 1e6],
                         acq_func_name='EI',
                         mean_func=MeanZero,
                         output_transform=Transform_Standardize,
                         cf=None
                         ):
    # scale Y
    T_out = output_transform(Ytr)
    train_y = T_out.scale_mean(Ytr)
    train_x = Xtr

    # create/train mean function and train gp
    mf = mean_func(train_x, train_y)
    while True:
        try:
            model, likelihood = create_and_fit_GP(train_x, train_y, ls_bounds,
                                                  out_bounds, mf, n_restarts=10)

            # define an acquisition function based on the model
            acq = acq_func_getter(acq_func_name, model, train_y, problem_bounds)

            # optimise the acquisition function
            budget = 1000 * train_x.shape[1]
            if cf is None:
                train_xnew, acq_f = botorch.optim.optimize_acqf(
                    acq_function=acq,
                    bounds=problem_bounds,
                    q=1,
                    num_restarts=10,
                    raw_samples=budget,
                )

            else:
                train_xnew, acq_f = optimize_acqf_cmaes_cf(acq, cf,
                                                           problem_bounds,
                                                           budget)
            break
        except RuntimeError as e:
            print(e)
            ls_bounds[0] *= 10
            print('New ls bounds:', ls_bounds)

    return train_xnew, acq_f, model, acq


def optimize_acqf_cmaes_cf(acq, cf, problem_bounds, budget):
    import numpy as np
    import cma
    import warnings

    def inital_point_generator(cf, lb, ub):
        def wrapper():
            while True:
                x = np.random.uniform(lb, ub)
                if np.all(x >= lb) and np.all(x <= ub) and cf(x):
                    return x
        return wrapper

    def acq_wrapper(acq, cf):
        got_cf = cf is not None

        def func(X):
            if isinstance(X, list):
                X = np.reshape(np.array(X), (len(X), -1))

            # convert to torch
            Xt = torch.from_numpy(X).unsqueeze(-2)

            # evaluate and negate because CMA-ES minimise
            Y = -acq(Xt)  # minus as CMA-ES minimises

            # convert back to numpy and then to list (for CMA-ES)
            y = Y.double().numpy().ravel().tolist()

            # evaluate constraint function for each decision vector
            if got_cf:
                for i, x in enumerate(X):
                    if not cf(x):
                        y[i] = np.inf

            # CMA-ES expects a float if it gives one decision vector
            if len(y) == 1:
                return y[0]

            return y
        return func

    lb = problem_bounds[0].numpy()
    ub = problem_bounds[1].numpy()

    cma_options = {'bounds': [list(lb), list(ub)],
                   'tolfun': 1e-7,
                   'maxfevals': budget,
                   'verb_disp': 0,
                   'verb_log': 0,
                   'verbose': -1,
                   'CMA_stds': np.abs(ub - lb),
                   }

    x0 = inital_point_generator(cf, lb, ub)
    f = acq_wrapper(acq, cf)

    # ignore warnings about flat fitness (i.e starting in a flat EI location)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xopt, es = cma.fmin2(objective_function=None,
                             parallel_objective=f,
                             x0=x0, sigma0=0.25, options=cma_options,
                             bipop=True, restarts=9)
        warnings.resetwarnings()

    train_xnew = torch.from_numpy(es.best.x).unsqueeze(0)
    acq_f = -torch.from_numpy(np.array(es.best.f))

    return train_xnew, acq_f
