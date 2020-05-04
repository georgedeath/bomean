# What do you Mean? <br/> The Role of the Mean Function in Bayesian Optimisation

This repository contains the Python3 code for the mean function experiments presented in:
> George De Ath, Jonathan E. Fieldsend, and Richard M. Everson. 2020. What do you Mean? The Role of the
Mean Function in Bayesian Optimisation. In Genetic and Evolutionary Computation Conference Companion (GECCO ’20 Companion), July 8–12, 2020, Cancún, Mexico. ACM, New York, NY, USA, 9 pages.</br>
> **Paper**: https://doi.org/10.1145/3377929.3398118 </br>
> **Preprint**: <https://arxiv.org/abs/2004.08349>

The repository also contains all training data used for the initialisation of each of the 51 optimisation runs carried to evaluate each mean function, the optimisation results of each of the runs on each of the test problems evaluated, and a jupyter notebook to generate the results figures and tables in the paper.

The remainder of this document details:

- The steps needed to install the package and related python modules on your system:  [installation](#installation)
- The format of the [training data](#training-data) and [saved runs](#optimisation-results).
- How to [repeat the experiments](#reproduction-of-experiments).
- How to [reproduce the figures in the paper](#reproduction-of-figures-and-tables-in-the-paper).

## Citation

If you use any part of this code in your work, please cite:

```bibtex
@inproceedings{death:bomean,
    title={What do you Mean? The Role of the Mean Function in Bayesian Optimisation},
    author = {George {De Ath} and Jonathan E. Fieldsend and Richard M. Everson},
    year = {2020},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3377929.3398118},
    doi = {10.1145/3377929.3398118},
    booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
}
```

## Installation

Manual installation is straight-forward for the optimisation library apart from the configuration of the PitzDaily test problem due to the installation and compilation of [OpenFOAM®](http://www.openfoam.com). Note that if you do not wish to use the PitzDaily test problem then the library will work fine without the optional instructions included at the end of this section. The following instructions will assume that [Anaconda3](https://docs.anaconda.com/anaconda/install/) has been installed and that you are running the following commands from the command prompt/console:

```bash
> # clone git repository
> git clone https://github.com/georgedeath/bomean /bomean
> cd /bomean
> # install python packages via new conda environment
> conda env create -f environment.yml
> # activate the new environment
> conda activate bomean
> # test it out by trying to run a completed experiment
(bomean) > python run_exp.py -p Branin -b 200 -r 1 -a EI -m MeanMin
Training data shape: torch.Size([200, 2])
```

Windows only: it may be necessary to also install [Visual C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### Optional instructions for the PitzDaily test problem
Now follow the linked instructions to [install OpenFOAM5](https://openfoamwiki.net/index.php/Installation/Linux/OpenFOAM-5.x/Ubuntu) (this will take 30min - 3hours to install). Note that this has only been tested with the Ubuntu 12.04 and 18.04 instructions. Once this has been successfully installed, the command `of5x` has to be ran before the PitzDaily test problem can be evaluated.

Finally, compile the pressure calculation function and check that the test problem works correctly:

```bash
> of5x
> cd /bomean/bopt/test_problems/Exeter_CFD_Problems/data/PitzDaily/solvers/
> wmake calcPressureDifference
> # test the PitzDaily solver
> cd /bomean
> python -m bopt.test_problems.pitzdaily
PitzDaily successfully instantiated..
Generated valid solution, evaluating..
Fitness value: [0.24748876]
```

Please ignore errors like `Getting LinuxMem: [Errno 2] No such file or directory: '/proc/621/status` as these are from OpenFOAM and do not impact the optimisation process.

## Training data

The initial training locations for each of the 51 sets of [Latin hypercube](https://www.jstor.org/stable/1268522) samples are located in the `data` directory in this repository with the filename structure `ProblemName_number`, e.g. the first set of training locations for the Branin problem is stored in `Branin_001.npz`. Each of these files is a compressed numpy file created with [torch.save](https://pytorch.org/docs/stable/torch.html#torch.save). It has two [torch.tensor](https://pytorch.org/docs/stable/torch.html#torch.tensor) arrays containing the 2*D initial locations and their corresponding fitness values. Note that for problems that have a non-default dimensionality (e.g. Ackley with d=5), then the data files have the dimensionality appended, e.g. `Ackley5_001.pt`; see the suite of [available test problems](bopt/test_problems/synthetic_problems.py). To load and inspect the training data, use the following instructions:

```python
> cd /bomean
> python
>>> import torch
>>> data = torch.load('data/Ackley5_001.pt')
>>> Xtr = data['Xtr']
>>> Ytr = data['Ytr']
>>> Xtr.shape, Ytr.shape
(torch.Size([10, 5]), torch.Size([10]))
```

The robot pushing test problems (push4 and push8) have a third array `'problem_params'`  that contains their instance-specific parameters:

```python
> cd /bomean
> python
>>> import torch
>>> data = torch.load('data/push4_001.pt')
>>> problem_params = data['problem_params']
>>> problem_params
{'t1_x': -3.4711672068892483, 't1_y': -0.5245311776466455}
```

these are automatically passed to the problem function when it is instantiated to create a specific problem instance.

## Optimisation results

The results of all optimisation runs can be found in the `results` directory. The filenames have the following structure: `ProblemName_MeanFunction_AcquisitionFunction_Run.pt`.

 Similar to the training data, these are also contain [torch.tensor](https://pytorch.org/docs/stable/torch.html#torch.tensor) arrays, `Xtr` and `Ytr`, corresponding to the evaluated locations in the optimisation run and their function evaluations, as well as a dictionary `problem_params` containing any optional parameters (i.e. the dimensionality of the problem and/or the robot pushing instance parameters). Note that the evaluations and their function values will also include the initial 2*D training locations at the beginning of the arrays.

The following example loads the first optimisation run on the Ackley (d=5) test problem with the linear mean function and using the EI acquisition function:

```python
> cd /bomean
> python
>>> import torch
>>> data = torch.load('results/Ackley5_MeanLinearCV_EI_001.pt')
>>> Xtr = data['Xtr']
>>> Ytr = data['Ytr']
>>> Xtr.shape, Ytr.shape
(torch.Size([200, 5]), torch.Size([200]))
>>> problem_params
{'d': 5}
```

## Reproduction of experiments

The python file `batch_simulation_script.py` provides a convenient way to reproduce an individual experimental evaluation carried out the paper. It has the following syntax:

```script
> python run_exp.py -h
usage: run_exp.py [-h] -p PROBLEM_NAME -r RUN_NO -b BUDGET -a {EI,UCB} -m
                  MEAN_NAME [-d PROBLEM_DIM]

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
    > python run_exp.py -p Ackley -b 200 -r 2 -a UCB -m MeanLinearCV -d 5

optional arguments:
  -h, --help       show this help message and exit
  -p PROBLEM_NAME  Test problem name. e.g. Branin, Shekel
  -r RUN_NO        Optimisation run number. Data file must exist in "data".
  -b BUDGET        Total evaluation budget, including the 2*d LHS samples.
  -a {EI,UCB}      Acquisition function name.
  -m MEAN_NAME     Mean function name. Available mean functions: MeanZero
                   (arithmetic), MeanMedian, MeanMin, MeanMax, MeanLinearCV
                   MeanQuadraticCV, MeanRandomForrest and MeanRBFCV.
  -d PROBLEM_DIM   Problem dimension (if different from original)
```

## Reproduction of figures and tables in the paper

The jupyter notebook [Mean_function_results.ipynb](Mean_function_results.ipynb) contains the code to load and process the optimisation results (stored in the `results` directory) as well as the code to produce all results figures and tables used in the paper and supplementary material.

## Model fitting experimental results

In the supplementary material we investigated the ability of the GP, with a given mean function, to model the given test functions. Results of this are in [Mean_function_results_RMSE.ipynb](Mean_function_results_RMSE.ipynb)
