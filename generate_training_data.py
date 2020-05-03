import os
import bopt
import torch

from pyDOE2.doe_lhs import lhs


torch.set_default_dtype(torch.float64)

data_dir = 'data'


selected_problems = {
    'Branin': {},
    'Eggholder': {},
    'GoldsteinPrice': {},
    'SixHumpCamel': {},
    'Shekel': {},
    'Ackley': {'d': 5},
    'Hartmann6': {},
    'Michalewicz': {'d': 10},
    'Rosenbrock': {'d': 10},
    'StyblinskiTang': {'d': 10}
}

n_runs = 51

for problem_name, problem_params in selected_problems.items():
    # get the test problem class
    f_class = getattr(bopt.test_problems, problem_name)

    # instantiate a torch version of the problem rescaled in [0, 1]^d
    f = bopt.util.TorchProblem(
        bopt.util.UniformProblem(f_class(**problem_params))
    )

    for run_no in range(1, n_runs + 1):
        fpath = bopt.util.generate_data_filename(problem_name, run_no,
                                                 problem_params,
                                                 data_dir=data_dir)
        if os.path.exists(fpath):
            s = f'File already exists, skipping: {fpath:s}'
            print(s)
            continue

        # generate 2 * d HLS samples in [0, 1]^d
        Xtr = lhs(f.dim, samples=2 * f.dim)
        Xtr = torch.from_numpy(Xtr)

        # "expensively" evaluate
        Ytr = f(Xtr)

        # save training data
        torch.save(obj={'Xtr': Xtr, 'Ytr': Ytr}, f=fpath)

        print(f'Saved file: {fpath:s}')
