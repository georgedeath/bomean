import numpy as np


class Ackley:
    """
    Ackley Function
    http://www.sfu.ca/~ssurjano/ackley.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -32.768)
        self.ub = np.full(self.dim, 32.768)

        # function parameters
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi

        # optimum value(s)
        self.xopt = np.full(self.dim, 0.0)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = -self.a * np.exp(-self.b * np.sqrt(np.mean(np.square(x), axis=1)))
        term2 = -np.exp(np.mean(np.cos(self.c * x), axis=1))

        val = term1 + term2 + self.a + np.exp(1.)

        return val.ravel()


class Bukin6:
    """
    Bukin Function N. 6
    http://www.sfu.ca/~ssurjano/bukin6.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.array([-15., -3.])
        self.ub = np.array([-5., 3.])

        # optimum value(s)
        self.xopt = np.array([-10., 1.])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = 100. * np.sqrt(np.abs(x[:, 1] - 0.01 * x[:, 0]**2))
        term2 = 0.01 * np.abs(x[:, 0] + 10.)

        val = term1 + term2

        return val.ravel()


class CrossInTray:
    """
    Cross-in-Tray Function
    http://www.sfu.ca/~ssurjano/crossit.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -10.)
        self.ub = np.full(self.dim, 10.)

        # optimum value(s)
        self.xopt = np.array([[1.3491, -1.3491], [1.3491, 1.3491],
                              [-1.3491, 1.3491], [-1.3491, -1.3491]])
        self.yopt = self.__call__(self.xopt[0, :])

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = np.sin(x[:, 0]) * np.sin(x[:, 1])
        term2 = np.exp(np.abs(100. - np.sqrt(x[:, 0]**2 + x[:, 1]**2) / np.pi))

        val = - 0.0001 * (np.abs(term1 * term2) + 1)**0.1

        return val.ravel()


class DropWave:
    """
    Drop-Wave Function
    http://www.sfu.ca/~ssurjano/drop.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -5.12)
        self.ub = np.full(self.dim, 5.12)

        # optimum value(s)
        self.xopt = np.array([0., 0.])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = 1. + np.cos(12. * np.sqrt(x[:, 0]**2 + x[:, 1]**2))
        term2 = 0.5 * (x[:, 0]**2 + x[:, 1]**2) + 2.

        val = -(term1 / term2)

        return val.ravel()


class Eggholder:
    """
    Eggholder Function
    http://www.sfu.ca/~ssurjano/egg.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -512.)
        self.ub = np.full(self.dim, 512.)

        # optimum value(s)
        self.xopt = np.array([512., 404.2319])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = -(x[:, 1] + 47.) * np.sin(np.sqrt(np.abs(x[:, 0] + x[:, 1] / 2. + 47.)))
        term2 = - x[:, 1] * np.sin(np.sqrt(np.abs(x[:, 1] - (x[:, 1] + 47.))))

        val = term1 + term2

        return val.ravel()


class GramacyLee:
    """
    Gramacy & Lee (2012) Function
    http://www.sfu.ca/~ssurjano/grlee12.html
    """

    def __init__(self):
        self.dim = 1
        self.lb = np.full(self.dim, 0.5)
        self.ub = np.full(self.dim, 2.5)

        # optimum value(s)
        self.xopt = np.array([0.54856368])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = np.sin(10. * np.pi * x[:, 0]) / (2. * x[:, 0])
        term2 = (x[:, 0] - 1)**4

        val = term1 + term2

        return val.ravel()


class Griewank:
    """
    Griewank Function
    http://www.sfu.ca/~ssurjano/griewank.html
    Note: we use the [-5, 5]^D by default, however the domain stated above
          is [-600, 600]^D
    """

    def __init__(self, d=2, lb=-5., ub=5):
        self.dim = d
        self.lb = np.full(self.dim, lb, dtype='float')
        self.ub = np.full(self.dim, ub, dtype='float')

        # function parameters
        self.cosdiv = np.sqrt(np.arange(1, d + 1)).reshape(1, d)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = np.sum(np.square(x) / 4000., axis=1)
        term2 = np.prod(np.cos(x / self.cosdiv), axis=1)

        val = term1 - term2 + 1

        return val.ravel()


class HolderTable:
    """
    Holder Table Function
    http://www.sfu.ca/~ssurjano/holder.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -10.)
        self.ub = np.full(self.dim, 10.)

        # optimum value(s)
        self.xopt = np.array([[8.05502, 9.66459],
                              [8.05502, -9.66459],
                              [-8.05502, 9.66459],
                              [-8.05502, -9.66459]])
        self.yopt = self.__call__(self.xopt[0, :])

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = np.sin(x[:, 0]) * np.cos(x[:, 1])
        term2 = np.exp(np.abs(1. - np.linalg.norm(x, axis=1) / np.pi))

        val = -np.abs(term1 * term2)

        return val.ravel()


class Levy:
    """
    Levy Function
    http://www.sfu.ca/~ssurjano/levy.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -10.)
        self.ub = np.full(self.dim, 10.)

        # optimum value(s)
        self.xopt = np.ones(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        w = 1. + (x - 1.) / 4.
        term1 = np.sin(np.pi * w[:, 0])**2
        term2 = np.sum((w[:, :-1] - 1.)**2
                       * (1. + 10. * np.sin(np.pi * w[:, :-1] + 1.)**2),
                       axis=1)
        term3 = (w[:, -1] - 1.)**2 * (1. + np.sin(2. * np.pi * w[:, -1])**2)

        val = term1 + term2 + term3

        return val.ravel()


class Levy13:
    """
    Levy Function N. 13
    http://www.sfu.ca/~ssurjano/levy13.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -10.)
        self.ub = np.full(self.dim, 10.)

        # optimum value(s)
        self.xopt = np.ones(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = np.sin(3. * np.pi * x[:, 0])**2
        term2 = (x[:, 0] - 1.)**2 * (1. + np.sin(3. * np.pi * x[:, 1])**2)
        term3 = (x[:, 1] - 1.)**2 * (1. + np.sin(2. * np.pi * x[:, 1])**2)

        val = term1 + term2 + term3

        return val.ravel()


class Rastrigin:
    """
    Rastrigin Function
    http://www.sfu.ca/~ssurjano/rastr.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -5.12)
        self.ub = np.full(self.dim, 5.12)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = 10. * self.dim
        term2 = np.sum(np.square(x) - 10. * np.cos(2. * np.pi * x), axis=1)

        val = term1 + term2

        return val.ravel()


class Schaffer2:
    """
    Schaffer Function N. 2
    http://www.sfu.ca/~ssurjano/schaffer2.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -100.)
        self.ub = np.full(self.dim, 100.)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        xsqr = np.square(x)

        term1 = np.sin(xsqr[:, 1] - xsqr[:, 0])**2 - 0.5
        term2 = 1. + 0.0001 * np.sum(xsqr, axis=1)**2

        val = 0.5 + term1 / term2

        return val.ravel()


class Schaffer4:
    """
    Schaffer Function N. 4
    http://www.sfu.ca/~ssurjano/schaffer4.html
    Note: the forumla in the above url is missing the a squared on the cos.
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -100.)
        self.ub = np.full(self.dim, 100.)

        # optimum value(s)
        self.xopt = np.array([[0, 1.25313], [0, -1.25313]])
        self.yopt = self.__call__(self.xopt[0, :])

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        xsqr = np.square(x)

        term1 = np.cos(np.sin(np.abs(xsqr[:, 0] - xsqr[:, 1])))**2 - 0.5
        term2 = 1. + 0.0001 * np.sum(xsqr, axis=1)**2

        val = 0.5 + term1 / term2

        return val.ravel()


class Schwefel:
    """
    Schwefel Function
    http://www.sfu.ca/~ssurjano/schwef.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -500.)
        self.ub = np.full(self.dim, 500.)

        # optimum value(s)
        self.xopt = np.full(self.dim, 420.9687)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = 418.9829 * self.dim
        term2 = np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

        val = term1 - term2

        return val.ravel()


class Schubert:
    """
    Schubert Function
    http://www.sfu.ca/~ssurjano/shubert.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -5.12)
        self.ub = np.full(self.dim, 5.12)

        # function parameters
        self.i = np.arange(1, 6).reshape(1, 5)

        # optimum value(s)
        self.xopt = np.array([-1.425128, -0.800273])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        x0 = x[:, 0][:, np.newaxis]
        x1 = x[:, 1][:, np.newaxis]

        term1 = np.sum(self.i * np.cos((self.i + 1) * x0 + self.i), axis=1)
        term2 = np.sum(self.i * np.cos((self.i + 1) * x1 + self.i), axis=1)

        val = term1 * term2

        return val.ravel()


class Bohachevsky1:
    """
    Bohachevsky Function N. 1
    http://www.sfu.ca/~ssurjano/boha.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -100.)
        self.ub = np.full(self.dim, 100.)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = x[:, 0]**2 + 2. * x[:, 1]**2
        term2 = 0.3 * np.cos(3. * np.pi * x[:, 0])
        term3 = 0.4 * np.cos(4. * np.pi * x[:, 1])

        val = term1 - term2 - term3 + 0.7

        return val.ravel()


class Perm0:
    """
    Perm Function 0, d, beta
    http://www.sfu.ca/~ssurjano/perm0db.html
    """

    def __init__(self, d=2, beta=10.):
        self.dim = d
        self.lb = np.full(self.dim, -self.dim)
        self.ub = np.full(self.dim, self.dim)

        # function parameters
        self.beta = beta
        self.J = np.arange(1, self.dim + 1)
        self.betaconst = np.tile(self.J + self.beta, reps=(self.dim, 1))
        self.minusconst = np.tile(1 / self.J, reps=(self.dim, 1))
        self.minusconst = self.minusconst**self.J[np.newaxis, :]

        # optimum value(s)
        self.xopt = 1 / np.arange(1, self.dim + 1)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = self.betaconst
        term2 = (x**self.J)[:, np.newaxis, :] - self.minusconst

        val = np.sum(np.sum(term1 * term2, axis=1)**2, axis=1)

        return val.ravel()


class RotatedHyperEllipsiod:
    """
    Rotated Hyper-Ellipsoid Function
    http://www.sfu.ca/~ssurjano/rothyp.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -65.536)
        self.ub = np.full(self.dim, 65.536)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        coef = np.arange(1, self.dim + 1)[::-1]
        val = np.sum(np.square(x) * coef[np.newaxis, :], axis=1)

        return val.ravel()


class Sphere:
    """
    Sphere Function
    http://www.sfu.ca/~ssurjano/spheref.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -5.12)
        self.ub = np.full(self.dim, 5.12)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        val = np.sum(np.square(x), axis=1)

        return val.ravel()


class SumDifferentPowers:
    """
    Sum of Different Powers Function
    http://www.sfu.ca/~ssurjano/sumpow.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -1.)
        self.ub = np.full(self.dim, 1.)

        # function parameters
        self.powers = np.arange(1, self.dim + 1).reshape(1, self.dim)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        val = np.sum(np.abs(x) ** self.powers, axis=1)

        return val.ravel()


class SumSquares:
    """
    Sum Squares Function (Axis Parallel Hyper-Ellipsoid function)
    http://www.sfu.ca/~ssurjano/sumsqu.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -5.12)
        self.ub = np.full(self.dim, 5.12)

        # function parameters
        self.coef = np.arange(1, self.dim + 1).reshape(1, self.dim)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        val = np.sum(self.coef * np.square(x), axis=1)

        return val.ravel()


class Trid:
    """
    Trid Function
    http://www.sfu.ca/~ssurjano/trid.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -self.dim**2)
        self.ub = np.full(self.dim, self.dim**2)

        # optimum value(s)
        self.xopt = np.arange(1, d + 1) * (d + 1. - np.arange(1, d + 1))
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = np.sum(np.square(x - 1), axis=1)
        term2 = np.sum(x[:, 1:] * x[:, :-1], axis=1)

        val = term1 - term2

        return val.ravel()


class Booth:
    """
    Booth Function
    http://www.sfu.ca/~ssurjano/booth.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -10.)
        self.ub = np.full(self.dim, 10.)

        # optimum value(s)
        self.xopt = np.array([1., 3.])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = (x[:, 0] + 2. * x[:, 1] - 7.)**2
        term2 = (2. * x[:, 0] + x[:, 1] - 5.)**2

        val = term1 + term2

        return val.ravel()


class Matyas:
    """
    Matyas Function
    http://www.sfu.ca/~ssurjano/matya.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -10.)
        self.ub = np.full(self.dim, 10.)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = 0.26 * np.sum(np.square(x), axis=1)
        term2 = 0.48 * np.prod(x, axis=1)

        val = term1 - term2

        return val.ravel()


class McCormick:
    """
    McCormick Function
    http://www.sfu.ca/~ssurjano/mccorm.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.array([-1.5, -3.])
        self.ub = np.array([4., 4.])

        # optimum value(s)
        self.xopt = np.array([-0.54719, -1.54719])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = np.sin(np.sum(x, axis=1))
        term2 = np.square(x[:, 0] - x[:, 1])
        term3 = - 1.5 * x[:, 0] + 2.5 * x[:, 1] + 1.

        val = term1 + term2 + term3

        return val.ravel()


class PowerSum:
    """
    Power Sum Function
    http://www.sfu.ca/~ssurjano/powersum.html
    Note: the coefficient matrix b is wrong in both the url above and the
    original source of the function. Assuming the global minimum is at
    x = [1,..,d] then b has to have elements:
    b[i] = sum x^(i+1); e.g d=4, b[0] = [1^1 + 2^1 + 3^1 + 4^1]
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.zeros(self.dim)
        self.ub = np.arange(1, self.dim + 1)

        # function parameters
        self.i = np.arange(1, self.dim + 1, dtype='float').reshape(1, self.dim, 1)
        self.b = np.sum(self.i ** self.i.ravel(), axis=1).reshape(1, self.dim)

        # optimum value(s)
        self.xopt = np.arange(1, self.dim + 1, dtype='float')
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        inner = np.sum(x[:, np.newaxis, :]**self.i, axis=2) - self.b
        term1 = np.sum(inner**2, axis=1)

        val = term1

        return val.ravel()


class Zakharov:
    """
    Zakharov Function
    http://www.sfu.ca/~ssurjano/zakharov.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -5.)
        self.ub = np.full(self.dim, 10.)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        _temp = np.sum(self.coef * x, axis=1)

        term1 = np.sum(np.square(x), axis=1)
        term2 = np.square(_temp)
        term3 = np.power(_temp, 4)

        val = term1 + term2 + term3

        return val.ravel()


class ThreeHumpCamel:
    """
    Three-Hump Camel Function
    http://www.sfu.ca/~ssurjano/camel3.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -5.)
        self.ub = np.full(self.dim, 5.)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = 2. * x[:, 0]**2 - 1.05 * x[:, 0]**4
        term2 = x[:, 0]**6 / 6. + np.prod(x, axis=1) + x[:, 1]**2

        val = term1 + term2

        return val.ravel()


class SixHumpCamel:
    """
    Six-Hump Camel Function
    http://www.sfu.ca/~ssurjano/camel6.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.array([-3., -2.])
        self.ub = np.array([3., 2.])

        # optimum value(s)
        self.xopt = np.array([[0.0898, -0.7126], [-0.0898, 0.7126]])
        self.yopt = self.__call__(self.xopt[0, :])

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        xsqr = np.square(x)

        term1 = (4. - 2.1 * xsqr[:, 0] + x[:, 0]**4 / 3.) * xsqr[:, 0]
        term2 = np.prod(x, axis=1) + (-4. + 4. * xsqr[:, 1]) * xsqr[:, 1]

        val = term1 + term2

        return val.ravel()


class DixonPrice:
    """
    Dixon-Price Function
    http://www.sfu.ca/~ssurjano/dixonpr.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -10.)
        self.ub = np.full(self.dim, 10.)

        # function parameters
        self.i = np.arange(1, self.dim + 1, dtype='float').reshape(1, self.dim)

        # optimum value(s)
        temp = np.arange(1, self.dim + 1)
        self.xopt = 2.**(-((2.**temp - 2.) / 2.**temp))
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        xsqr = np.square(x)

        term1 = (x[:, 0] - 1)**2
        term2 = np.sum(self.i * (2. * xsqr[:, 1:] - x[:, :-1])**2, axis=1)

        val = term1 + term2

        return val.ravel()


class Rosenbrock:
    """
    Rosenbrock Function
    http://www.sfu.ca/~ssurjano/rosen.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -5.)
        self.ub = np.full(self.dim, 10.)

        # optimum value(s)
        self.xopt = np.ones(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = 100. * (x[:, 1:] - x[:, :-1]**2)**2
        term2 = (x[:, :-1] - 1)**2
        term3 = np.sum(term1 + term2, axis=1)

        val = term3

        return val.ravel()


class DeJong5:
    """
    De Jong Function N. 5
    http://www.sfu.ca/~ssurjano/dejong5.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -65.536)
        self.ub = np.full(self.dim, 65.536)

        # function parameters
        self.A = np.zeros((2, 25))
        self.A[0, :] = np.tile([-32, -16, 0, 16, 32], (1, 5))
        self.A[1, :] = np.tile([-32, -16, 0, 16, 32], (5, 1)).T.ravel()
        self.i = np.arange(1, 26, dtype='float').reshape(1, 25)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = self.i + (x[:, 0][:, np.newaxis] - self.A[0, :][np.newaxis, :])**6
        term2 = (x[:, 1][:, np.newaxis] - self.A[1, :][np.newaxis, :])**6
        term3 = np.sum(1 / (term1 + term2), axis=1)
        val = (0.002 + term3)**(-1.)

        return val.ravel()


class Easom:
    """
    Easom Function
    http://www.sfu.ca/~ssurjano/easom.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -100.)
        self.ub = np.full(self.dim, 100.)

        # optimum value(s)
        self.xopt = np.array([np.pi, np.pi])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = -np.cos(x[:, 0]) * np.cos(x[:, 1])
        term2 = np.exp(-(x[:, 0] - np.pi)**2 - (x[:, 1] - np.pi)**2)
        val = term1 * term2

        return val.ravel()


class Michalewicz:
    """
    Michalewicz Function - 2D
    http://www.sfu.ca/~ssurjano/michal.html
    Note for d=5 f(x*) = -4.687658
             d=10 f(x*) = -9.66015
             (unknown x* for both)
    """

    def __init__(self, d=2, m=10):
        self.dim = d
        self.lb = np.full(self.dim, 0.)
        self.ub = np.full(self.dim, np.pi)

        # function parameters
        self.i = np.arange(1, self.dim + 1, dtype='float')
        self.m = m

        # optimum value(s)
        if self.dim == 2:
            self.xopt = np.array([2.2, 1.57])
            self.yopt = np.array([-1.8013])

        elif self.dim == 5:
            self.xopt = None
            self.yopt = np.array([-4.687658])

        elif self.dim == 10:
            self.xopt = None
            self.yopt = np.array([-4.687658])

        else:
            self.xopt = None
            self.yopt = None

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = -np.sum(np.sin(x) * np.sin((self.i * x**2) / np.pi)**(2. * self.m), axis=1)
        val = term1

        return val.ravel()


class Beale:
    """
    Beale Function
    http://www.sfu.ca/~ssurjano/beale.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -4.5)
        self.ub = np.full(self.dim, 4.5)

        # optimum value(s)
        self.xopt = np.array([3., 0.5])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = (1.5 - x[:, 0] + np.prod(x, axis=1))**2
        term2 = (2.25 - x[:, 0] + x[:, 0] * x[:, 1]**2)**2
        term3 = (2.625 - x[:, 0] + x[:, 0] * x[:, 1]**3)**2
        val = term1 + term2 + term3

        return val.ravel()


class Branin:
    """
    Branin Function
    http://www.sfu.ca/~ssurjano/branin.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.array([-5., 0.])
        self.ub = np.array([10., 15.])

        # function parameters
        self.a = 1.
        self.b = 5.1 / (4. * np.pi**2)
        self.c = 5. / np.pi
        self.r = 6.
        self.s = 10.
        self.t = 1. / (8. * np.pi)

        # optimum value(s)
        self.xopt = np.array([[-np.pi, 12.275],
                              [np.pi, 2.275],
                              [9.42478, 2.475]])
        self.yopt = self.__call__(self.xopt[0, :])

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = self.a * (x[:, 1] - self.b * x[:, 0]**2
                          + self.c * x[:, 0] - self.r)**2
        term2 = self.s * (1. - self.t) * np.cos(x[:, 0]) + self.s

        val = term1 + term2

        return val.ravel()


class Colville:
    """
    Colville Function
    http://www.sfu.ca/~ssurjano/colville.html
    """

    def __init__(self):
        self.dim = 4
        self.lb = np.full(self.dim, -10.)
        self.ub = np.full(self.dim, 10.)

        # optimum value(s)
        self.xopt = np.ones(4)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = 100. * (x[:, 0]**2 - x[:, 1]) + (x[:, 0] - 1.)**2
        term2 = (x[:, 2] - 1.)**2 + 90. * (x[:, 2]**2 - x[:, 3])**2
        term3 = 10.1 * ((x[:, 1] - 1.)**2 + (x[:, 3] - 1)**2)
        term4 = 19.8 * (x[:, 1] - 1) * (x[:, 3] - 1)

        val = term1 + term2 + term3 + term4

        return val.ravel()


class Forrester:
    """
    Forrester et al. (2008) Function
    http://www.sfu.ca/~ssurjano/forretal08.html
    """

    def __init__(self):
        self.dim = 1
        self.lb = np.full(self.dim, 0.)
        self.ub = np.full(self.dim, 1.)

        # optimum value(s)
        self.xopt = np.array([0.75724768])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = (6. * x - 2.)**2 * np.sin(12. * x - 4)

        val = term1

        return val.ravel()


class GoldsteinPrice:
    """
    Goldstein-Price Function
    http://www.sfu.ca/~ssurjano/goldpr.html
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -2.)
        self.ub = np.full(self.dim, 2.)

        # optimum value(s)
        self.xopt = np.array([0., -1.])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = (1. + (x[:, 0] + x[:, 1] + 1)**2
                 * (19. - 14. * x[:, 0] + 3. * x[:, 0]**2
                    - 14. * x[:, 1] + 6. * x[:, 0] * x[:, 1]
                    + 3. * x[:, 1]**2))
        term2 = (30 + (2. * x[:, 0] - 3. * x[:, 1])**2
                 * (18. - 32. * x[:, 0] + 12. * x[:, 0]**2
                    + 48. * x[:, 1] - 36. * x[:, 0] * x[:, 1]
                    + 27. * x[:, 1]**2))
        val = term1 * term2

        return val.ravel()


class Hartmann3:
    """
    Hartmann 3D Function
    http://www.sfu.ca/~ssurjano/hart3.html
    """

    def __init__(self):
        self.dim = 3
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        # function parameters
        self.alpha = np.array([1., 1.2, 3., 3.2])
        self.A = np.array([[3., 10., 30.],
                           [0.1, 10., 35.],
                           [3., 10., 30.],
                           [0.1, 10., 35.]])
        self.P = np.array([[0.3689, 0.1170, 0.2673],
                           [0.4699, 0.4389, 0.7470],
                           [0.1091, 0.8732, 0.5547],
                           [0.0381, 0.5743, 0.8828]])

        # optimum value(s)
        self.xopt = np.array([0.114614, 0.555649, 0.852547])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = np.sum(self.A[np.newaxis, :, :]
                       * (x[:, np.newaxis, :] - self.P[np.newaxis, :, :])**2,
                       axis=2)
        term2 = -np.sum(self.alpha[np.newaxis, :] * np.exp(-term1), axis=1)

        val = term2

        return val.ravel()


class Hartmann6:
    """
    Hartmann 6D Function
    http://www.sfu.ca/~ssurjano/hart6.html
    """

    def __init__(self):
        self.dim = 6
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        # function parameters
        self.alpha = np.array([1., 1.2, 3., 3.2])
        self.A = np.array([[10., 3., 17., 3.5, 1.7, 8.],
                           [0.05, 10., 17., 0.1, 8., 14.],
                           [3., 3.5, 1.7, 10., 17., 8.],
                           [17., 8., 0.05, 10., 0.1, 14.]])

        self.P = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                           [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                           [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                           [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

        # optimum value(s)
        self.xopt = np.array([[0.201690, 0.150011, 0.476874,
                               0.275332, 0.311652, 0.657300]])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = np.sum(self.A[np.newaxis, :, :]
                       * (x[:, np.newaxis, :] - self.P[np.newaxis, :, :])**2,
                       axis=2)
        term2 = -np.sum(self.alpha[np.newaxis, :] * np.exp(-term1), axis=1)

        val = term2

        return val.ravel()


class Perm:
    """
    Perm Function d, beta
    http://www.sfu.ca/~ssurjano/permdb.html
    """

    def __init__(self, d=2, beta=0.5):
        self.dim = d
        self.lb = np.full(self.dim, -self.dim)
        self.ub = np.full(self.dim, self.dim)

        # function parameters
        self.beta = beta
        self.J = np.arange(1, self.dim + 1, dtype='float')

        self.betaconst = np.tile(self.J, reps=(self.dim, 1))
        self.betaconst = self.betaconst**self.J[:, np.newaxis]
        self.betaconst += self.beta

        # optimum value(s)
        self.xopt = np.arange(1, self.dim + 1)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = self.betaconst
        term2 = (x[:, np.newaxis, :] / self.J)**self.J - 1.

        val = np.sum(np.sum(term1 * term2, axis=1)**2, axis=1)

        return val.ravel()


class Powell:
    """
    Powell Function
    http://www.sfu.ca/~ssurjano/powell.html
    """

    def __init__(self, d=4):
        self.dim = d
        self.lb = np.full(self.dim, -4.)
        self.ub = np.full(self.dim, 5.)

        if (d % 4) != 0:
            s = f'Problem dimension must be a multiple of 4: {d}'
            raise ValueError(s)

        # function parameters
        self.a = np.arange(0, self.dim - 3, 4)
        self.b = np.arange(1, self.dim - 2, 4)
        self.c = np.arange(2, self.dim - 1, 4)
        self.d = np.arange(3, self.dim, 4)

        # optimum value(s)
        self.xopt = np.zeros(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = (x[:, self.a] + 10. * x[:, self.b])**2
        term2 = 5. * (x[:, self.c] - x[:, self.d])**2
        term3 = (x[:, self.b] - 2. * x[:, self.c])**4
        term4 = 10. * (x[:, self.a] - x[:, self.d])**4

        val = np.sum(term1 + term2 + term3 + term4, axis=1)

        return val.ravel()


class Shekel:
    """
    Shekel Function
    http://www.sfu.ca/~ssurjano/shekel.html
    """

    def __init__(self, m=10):
        self.dim = 4
        self.lb = np.full(self.dim, 0.)
        self.ub = np.full(self.dim, 10.)

        if m < 1 or m > 10:
            s = f'Problem only defined for m \\in [1, 10]: {m}'
            raise ValueError(s)

        # function parameters
        self.m = m
        self.beta = np.array([0.1, 0.2, 0.2, 0.4, 0.4,
                              0.6, 0.3, 0.7, 0.5, 0.5])
        self.C = np.array([[4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                           [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6],
                           [4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                           [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6]])

        # optimum value(s)
        self.xopt = np.array([4., 4., 4., 4.])
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = (x[:, :, np.newaxis] - self.C[np.newaxis, :, :self.m])**2
        term2 = np.sum(term1, axis=1) + self.beta[np.newaxis, :self.m]
        term3 = -np.sum(1. / term2, axis=1)

        val = term3

        return val.ravel()


class StyblinskiTang:
    """
    Styblinski-Tang Function
    http://www.sfu.ca/~ssurjano/stybtang.html
    """

    def __init__(self, d=2):
        self.dim = d
        self.lb = np.full(self.dim, -5.)
        self.ub = np.full(self.dim, 5.)

        # optimum value(s)
        self.xopt = np.full(self.dim, -2.903534)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = x**4 - 16. * x**2 + 5. * x
        term2 = 0.5 * np.sum(term1, axis=1)

        val = term2

        return val.ravel()
