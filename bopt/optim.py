import os
import torch
from . import gp, mean_functions, test_problems, transforms, util


def perform_optimisation(problem_name, problem_params, run_no, budget,
                         mean_name, acq_name, save_path, save_every=1):
    ls_bounds = [1e-6, 1e6]
    out_bounds = [1e-6, 1e6]
    transform_name = 'Transform_Standardize'

    data_path = util.generate_data_filename(problem_name, run_no,
                                            problem_params, data_dir='data')
    # check if we're resuming a saved run
    if os.path.exists(save_path):
        load_path = save_path
    else:
        load_path = data_path

    # load the training data
    data = torch.load(load_path)
    Xtr = data['Xtr']
    Ytr = data['Ytr']

    # if it has additional arguments add them to the dictionary passed to f
    if 'problem_params' in data:
        problem_params.update(data['problem_params'])

    print(f'Training data shape: {Xtr.shape}')

    # load the problem instance
    f = getattr(test_problems, problem_name)(**problem_params)

    # wrap the problem for torch and so that it resides in [0, 1]^d
    f = util.TorchProblem(util.UniformProblem(f))
    problem_bounds = torch.stack((f.lb, f.ub))

    # set up the rest of the classes needed (mean function etc)
    transform_function = getattr(transforms, transform_name)
    mean_func = getattr(mean_functions, mean_name)

    # begin/continue the bayesopt loop
    while Xtr.shape[0] < budget:
        Xnew, acq_f, model, acq = gp.perform_BO_iteration(
            Xtr, Ytr, problem_bounds, ls_bounds=ls_bounds, out_bounds=out_bounds,
            acq_func_name=acq_name, mean_func=mean_func,
            output_transform=transform_function,
            cf=f.cf
        )

        # expensively evaluate function - note that in the PitzDaily CFD test
        # problem the mesh can fail to converge even though the solution passes
        # the constraint function. if this occurs we generate a random valid
        # solution and use that in replacement of the newly selected solution
        while True:
            try:
                # attempt to evaluate the solution
                Ynew = f(Xnew).view(1)
                break

            # TypeError: 'NoneType' object is not subscriptable (PitzDaily)
            except TypeError:
                if problem_name != 'PitzDaily':
                    raise

                s = 'PitzDaily: CFD mesh failed to converge'
                s += ' generating random solution'
                print(s)
                # generate a random valid solution
                while True:
                    Xnew = torch.rand((1, f.dim)) * (f.ub - f.lb) + f.lb
                    if (f.cf is None) or f.cf(Xnew):
                        break

        # augment the training data
        Xtr = torch.cat((Xtr, Xnew))
        Ytr = torch.cat((Ytr, Ynew))

        s = f'Iteration {Xtr.shape[0]:> 3d} ({abs(f.yopt - Ytr.min()).item():g})'
        s += f'\n\tLocation selected: {Xnew.flatten().numpy()}'
        s += f'\n\tFitness: {Ynew.item():g}\n'
        print(s)

        # save the run
        if (Xtr.shape[0] % save_every == 0) or (Xtr.shape[0] == budget):
            save_dict = {'Xtr': Xtr, 'Ytr': Ytr,
                         'problem_params': problem_params}
            torch.save(obj=save_dict, f=save_path)
