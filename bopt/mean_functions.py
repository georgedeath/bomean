import torch
import gpytorch
import numpy as np

from gpytorch.utils.broadcasting import _mul_broadcast_shape
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, GridSearchCV

# cross-validation parameters
_LAMBDAS = np.power(10.0, [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
_LAMBDAS = torch.from_numpy(_LAMBDAS)

_SIGMAS = np.power(10.0, [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
                          0.0, 0.5, 1.0, 1.5, 2.0])
_SIGMAS = torch.from_numpy(_SIGMAS)


class MeanConstant(gpytorch.means.Mean):
    def __init__(self, value, batch_shape=torch.Size(), **kwargs):
        super().__init__()
        self.batch_shape = batch_shape

        self.constant = torch.full((*batch_shape, 1), value)

    def forward(self, input):
        if input.shape[:-2] == self.batch_shape:
            return self.constant.expand(input.shape[:-1])

        return self.constant.expand(_mul_broadcast_shape(input.shape[:-1],
                                                         self.constant.shape))


class MeanZero(MeanConstant):
    def __init__(self, train_x, train_y):
        super().__init__(value=0.)


class MeanAverage(MeanConstant):
    def __init__(self, train_x, train_y):
        value = train_y.flatten().mean()
        super().__init__(value=value)


class MeanMedian(MeanConstant):
    def __init__(self, train_x, train_y):
        # torch median makes difference choices in the case that
        # len(y) is even - it takes the value closest to the mean
        # value (out of the two middle values), instead of halfway
        # between them.

        train_y = train_y.flatten()

        if train_y.numel() % 2 == 0:
            ys, _ = train_y.sort()
            n = train_y.numel() // 2

            value = (train_y[n - 1] + train_y[n]) / 2

        else:
            value = train_y.median()

        super().__init__(value=value)


class MeanMin(MeanConstant):
    def __init__(self, train_x, train_y):
        value = train_y.flatten().min()
        super().__init__(value=value)


class MeanMax(MeanConstant):
    def __init__(self, train_x, train_y):
        value = train_y.flatten().max()
        super().__init__(value=value)


class MeanQuadraticCV(gpytorch.means.Mean):
    def __init__(self, train_x, train_y, kfolds=5):
        super().__init__()
        self.batch_shape = torch.Size()

        # train linear regressor
        train_x = train_x.detach().numpy()
        train_y = train_y.detach().numpy()

        # get the quadratic features creator
        self.pf = PolynomialFeatures(2)

        # extract p=2 order polynomial features
        x_poly = self.pf.fit_transform(train_x)

        # check if we have enough points to perform k-fold
        # cross validation
        kfolds = np.minimum(kfolds, train_y.size // 2)

        if kfolds > 1:
            param_grid = {'alpha': _LAMBDAS.numpy()}
            cv = GridSearchCV(Ridge(), param_grid,
                              cv=kfolds)
            cv.fit(x_poly, train_y)

            # best_estimator_ contains a fitted model to the best params
            self.model = cv.best_estimator_

        else:
            errmsg = 'Not enough points to perform cross validation:'
            errmsg += f' {train_y.size} (minimum = 4).'
            errmsg += ' Proceeding with alpha=0.01'
            print(errmsg)

            self.model = Ridge(alpha=0.01)
            self.model.fit(x_poly, train_y)

    def forward(self, input):
        # no batch; input is shape [n, d]
        if input.shape[:-2] == self.batch_shape:
            _temp = self.pf.fit_transform(input)
            return torch.tensor(self.model.predict(_temp))

        # else we have a shape of [b1, b2, ..., bk, n, d]
        else:
            X = input.reshape(-1, input.shape[-1]).detach().numpy()
            _temp = self.pf.fit_transform(X)
            ret = torch.tensor(self.model.predict(_temp))
            ret = ret.reshape(input.shape[:-1])
            return ret


class MeanLinearCV(gpytorch.means.Mean):
    def __init__(self, train_x, train_y, kfolds=5):
        super().__init__()
        self.batch_shape = torch.Size()

        # train linear regressor
        train_x = train_x.detach().numpy()
        train_y = train_y.detach().numpy()

        # check if we have enough points to perform k-fold
        # cross validation
        kfolds = int(min(kfolds, train_y.size // 2))

        if kfolds > 1:
            param_grid = {'alpha': _LAMBDAS.numpy()}
            cv = GridSearchCV(Ridge(), param_grid,
                              cv=kfolds)
            cv.fit(train_x, train_y)

            # best_estimator_ contains a fitted model to the best params
            self.model = cv.best_estimator_

        else:
            errmsg = 'Not enough points to perform cross validation:'
            errmsg += f' {train_y.size} (minimum = 4).'
            errmsg += ' Proceeding with alpha=0.01'
            print(errmsg)

            self.model = Ridge(alpha=0.01)
            self.model.fit(train_x, train_y)

    def forward(self, input):
        # if we have the default N by d shape
        if input.shape[:-2] == self.batch_shape:
            return torch.tensor(self.model.predict(input))

        # else we have N by q by d
        else:
            X = input.reshape(-1, input.shape[-1]).detach().numpy()
            ret = torch.tensor(self.model.predict(X))
            ret = ret.reshape(input.shape[:-1])
            return ret


class MeanRandomForrest(gpytorch.means.Mean):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.batch_shape = torch.Size()

        self.N_TREES = 100
        self.et = ExtraTreesRegressor(n_estimators=self.N_TREES,
                                      bootstrap=True)
        self.et.fit(train_x.detach().numpy(), train_y.detach().numpy())

    def forward(self, input):
        input = input.detach().numpy()
        # no batch; input is shape [n, d]
        if input.shape[:-2] == self.batch_shape:
            return torch.tensor(self.et.predict(input))

        # else we have a shape of [b1, b2, ..., bk, n, d]
        else:
            # reshape to (N, D) to batch predict, then reshape back
            X = input.reshape(-1, input.shape[-1])
            ret = torch.tensor(self.et.predict(X))
            ret = ret.reshape(input.shape[:-1])
            return ret


class MeanRBFCV(gpytorch.means.Mean):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.batch_shape = torch.Size()
        self.model = RBFNetCV(train_x, train_y)

        # max evaluations to carry out in parallel - this is used to avoid
        # huge amounts of memory being used
        self.max_N = 20000

    def forward(self, input):
        with torch.no_grad():
            # no batch; input is shape [n, d]
            if input.shape[:-2] == self.batch_shape:
                return self.model.forward(input)

            # else we have a shape of [b1, b2, ..., bk, n, d]
            else:
                # reshape to (N, D) to batch predict, then reshape back
                X = input.reshape(-1, input.shape[-1])
                N = X.shape[0]

                if N > self.max_N:
                    ret = torch.zeros(N)

                    start = 0
                    for _ in range(N // self.max_N):
                        end = start + self.max_N
                        ret[start:end] = self.model.forward(X[start:end, :])
                        start = end

                    if X[start:-1].shape[0] > 0:
                        ret[start:] = self.model.forward(X[start:, :])

                else:
                    ret = self.model.forward(X)

                ret = ret.reshape(input.shape[:-1])
                return ret


class RBFNetCV(torch.nn.Module):
    def __init__(self, X, Y):
        super().__init__()

        # number of centres and output classes
        self.K = X.shape[0]
        self.outdim = 1

        # one rbf on each datapoint
        self.centers = X.detach().clone()

        # same lengthscale for all
        self.sigma = torch.tensor(self.K, dtype=torch.float64)

        # weight of each kernel
        self.w = None

        # caching
        self._I = torch.eye(self.K)
        self.loss_func = torch.nn.MSELoss()

        # fit network
        self.fit(X, Y)

    def activations(self, x):
        x = x[:, None, :]
        c = self.centers[None, :, :]

        # calculate the distance between each centre and the input points
        # d = (x - c).pow(2).sum(-1).pow(0.5) * sigma
        d = torch.norm(x - c, p=2, dim=-1)

        # rbf activation
        phi = torch.exp(-self.sigma * d.pow(2))

        return phi

    def forward(self, x, w=None):
        # get the activations
        phi = self.activations(x)

        # combine the weighted outputs
        w = self.w if w is None else w

        return (w * phi).sum(1)

    def _calc_weights(self, X, Y, _lambda):
        # sets the rbf network weights based on the regularised
        # least squares solution; for full detail see section 6:
        # https://www.cc.gatech.edu/~isbell/tutorials/rbf-intro.pdf
        H = self.activations(X)
        A = H.T @ H + self._I * _lambda

        # try to use the torch SVD to perform the pseudoinverse (gessd)
        try:
            Ainv = torch.pinverse(A)

        # the faster torch pseudoinverse uses the 'gessd' method which handles
        # poorly-conditioned matrices badly. so, if this fails, switch to the
        # scipy version which used the more robust, but SLOWER, 'gesvd' method
        except:
            print('torch.pinverse() failed, using scipy')
            import scipy
            U, s, Vh = scipy.linalg.svd(A.detach().numpy(),
                                        lapack_driver="gesvd",
                                        full_matrices=False)
            mask = s != 0.
            s[mask] = 1 / s[mask]
            sT = np.diag(s).T
            Ainv = Vh.T @ sT @ U.T
            Ainv = torch.from_numpy(Ainv)

        # weights
        return Ainv @ H.T @ Y

    def fit(self, X, Y, kfolds=5):
        with torch.no_grad():
            _lambdas = _LAMBDAS
            _sigmas = _SIGMAS

            # perform 5-fold cv
            kfolds = int(min(kfolds, Y.numel() // 2))

            if kfolds > 1:
                res = torch.zeros(_lambdas.numel(), _sigmas.numel(), kfolds)

                kf = KFold(n_splits=kfolds)

                # perform cross validation for each of the lambdas
                for i, _lambda in enumerate(_lambdas):
                    for j, _sigma in enumerate(_sigmas):
                        for k, (train_idx, val_idx) in enumerate(kf.split(X)):
                            self.sigma = _sigma
                            w = self._calc_weights(X[train_idx],
                                                   Y[train_idx],
                                                   _lambda)

                            # predictions
                            pred = self.forward(X[val_idx, :], w)

                            # MSE
                            res[i, j, k] = self.loss_func(pred, Y[val_idx])

                # identify the best pair of parameters by taking the average MSE
                resmeans = torch.mean(res, axis=-1)
                am = torch.argmin(resmeans)
                l_ind, s_ind = np.unravel_index(am, (_lambdas.numel(),
                                                     _sigmas.numel()))

                # find the best performing
                best_lambda = _lambdas[l_ind]
                self.sigma = _sigmas[s_ind]

            else:
                best_lambda = 0.01

            # calculate the weights for all the data using it
            self.w = self._calc_weights(X, Y, best_lambda)
