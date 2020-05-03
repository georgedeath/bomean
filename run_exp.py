"""
Command-line parser for running individual experiments.


"""

import argparse

from bopt.optim import perform_optimisation
from bopt.util import generate_save_filename

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''
optimisation using EI and different mean functions: experimental evaluation
--------------------------------------------
Examples:
    Using EI with the RBF mean function on the Branin test function and a
    budget of 200 function evaluations (including 2*D training points),
    with a run number of 1 (must have corresponding training data):
    > python run_exp.py -p Branin -b 200 -r 1 -a EI -m MeanRBFCV

    Using UCB with the linear mean function on the Ackley function and a budget
    of 200 function evaluation (including 2*D training points), with a run
    number of 2 and a problem dimensionality of 5. Note that the default
    dimensionality of the Ackley function is 2 and, therefore, we need to
    specify it; the corresponding training data should be in
    "data/Ackley5_002.pt":
    > python run_exp.py -p Branin -b 200 -r 2 -a UCB -m MeanLinearCV -d 5
'''
)

parser.add_argument('-p',
                    dest='problem_name',
                    type=str,
                    help='Test problem name. e.g. Branin, Shekel',
                    required=True)

parser.add_argument('-r',
                    dest='run_no',
                    type=int,
                    help='Optimisation run number. Data file must exist in "data".',
                    required=True)

parser.add_argument('-b',
                    dest='budget',
                    type=int,
                    help='Total evaluation budget, including the 2*d LHS samples.',
                    required=True)

parser.add_argument('-a',
                    dest='acq_name',
                    choices=['EI', 'UCB'],
                    type=str,
                    help='Acquisition function name.',
                    required=True)

parser.add_argument('-m',
                    dest='mean_name',
                    type=str,
                    help='Mean function name. Available mean functions:'
                         + ' MeanZero (arithmetic), MeanMedian, MeanMin,'
                         + ' MeanMax, MeanLinearCV MeanQuadraticCV,'
                         + ' MeanRandomForrest and MeanRBFCV.',
                    required=True)

parser.add_argument('-d',
                    dest='problem_dim',
                    type=int,
                    help='Problem dimension (if different from original)',
                    required=False)

a = parser.parse_args()

problem_params = {}
if a.problem_dim is not None:
    problem_params['d'] = a.problem_dim

save_path = generate_save_filename(a.problem_name, a.mean_name, a.acq_name,
                                   a.run_no, problem_params=problem_params,
                                   results_dir='results')

perform_optimisation(a.problem_name, problem_params, a.run_no, a.budget,
                     a.mean_name, a.acq_name, save_path)
