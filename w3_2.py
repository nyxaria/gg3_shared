import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import scipy.special
import os

import w3_utils
from models_hmm import RampModelHMM



if __name__ == "__main__":
    K = 25
    T_MS = 100
    RH = 50
    M_GRID = 15
    X0 = 0.2

    param_specs = OD([
        ('beta', np.linspace(0, 7, M_GRID)),
        ('sigma', np.exp(np.linspace(np.log(0.04), np.log(4), M_GRID))),
        ('x0', X0),
        ('K', K),
        ('T', T_MS),
        ('Rh', RH)
    ])

    params_grid = w3_utils.make_params_grid(param_specs)

    uni_post = w3_utils.uniform_posterior(params_grid)

    gauss_post = w3_utils.gaussian_posterior(params_grid, mu={
        'beta': 1.0, 'sigma': 1.0
    }, cov={
        ('beta', 'beta'): 2,
        ('sigma', 'sigma'): 3
    }, log=True)