import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import scipy.special
import os
from scipy.interpolate import griddata
import w3_2


import w3_utils
from models_hmm import RampModelHMM

var = lambda a, b, frac: ((b-a) * frac) ** 2
mean = lambda a, b: (b+a)/2

if __name__ == "__main__":
    K = 25
    T_MS = 100
    RH = 50
    M_GRID = 15
    X0 = 0.2
    # N_DATASETS = 50
    # N_TRIALS = 5

    # STD_FRACTION = 0.25

    # TODO changeme
    BETA_RANGE = (0, 4)
    SIGMA_RANGE = (0.04, 4)

    M_RANGE = (T_MS * 0.25, T_MS * 0.75)
    R_RANGE = (1, 6)

    ramp_param_specs = OD([
        ('beta', np.linspace(*BETA_RANGE, M_GRID)),
        ('sigma', np.exp(np.linspace(np.log(SIGMA_RANGE[0]),
                                     np.log(SIGMA_RANGE[1]),
                                     M_GRID))),
        ('x0', X0),
        ('K', K),
        ('T', T_MS),
        ('Rh', RH)
    ])

    step_param_specs = OD([('m', list(M_RANGE) + [M_GRID]),
                ('r', list(R_RANGE) + [6]),
                ('x0', X0),
                ('K', K),
                ('T', T_MS),
                ('Rh', RH)])

    ramp_params_grid = w3_utils.make_params_grid(ramp_param_specs)
    step_params_grid = w3_utils.make_params_grid(step_param_specs)

    uniform_ramp_posterior = w3_utils.uniform_prior(ramp_params_grid)
    uniform_step_posterior = w3_utils.uniform_prior(step_params_grid)



    N_DATASETS = 72
    N_TRIALS = 50

    # 4_11

    # BASELINE

    fn = "./results/UU_D" + str(N_DATASETS) + "_shape1_T" + str(N_TRIALS) + ".csv"

    w3_2.model_selection(
        ramp_params_grid, step_params_grid,
        uniform_ramp_posterior, uniform_step_posterior,  # generating
        uniform_ramp_posterior, uniform_step_posterior,  # inference
        N_DATASETS=N_DATASETS, N_TRIALS=N_TRIALS,
        ramp_gamma_shape=1, step_gamma_shape=1,
        # format: generating: G/U; inference: G/U; n. datasets, n. trials
        # if G, append std_fraction on front
        save_to=os.path.join(os.getcwd(), fn)
    )

    w3_2.plot_heatmap(fn, 'Uniform prior, shape=1, ' + str(N_TRIALS) + ' trials/dataset')
    w3_2.plot_confusion_matrix(fn, 'Uniform prior, shape=1, ' + str(N_TRIALS) + ' trials/dataset',
                               save_name=fn[:-4] + '.png')

    # TEST 1

    fn = "./results/UU_D" + str(N_DATASETS) + "_shape3_T" + str(N_TRIALS) + ".csv"

    w3_2.model_selection(
        ramp_params_grid, step_params_grid,
        uniform_ramp_posterior, uniform_step_posterior,  # generating
        uniform_ramp_posterior, uniform_step_posterior,  # inference
        N_DATASETS=N_DATASETS, N_TRIALS=N_TRIALS,
        ramp_gamma_shape=5, step_gamma_shape=5,
        # format: generating: G/U; inference: G/U; n. datasets, n. trials
        # if G, append std_fraction on front
        save_to=os.path.join(os.getcwd(), fn)
    )

    w3_2.plot_heatmap(fn, 'Uniform prior, shape=3, ' + str(N_TRIALS) + ' trials/dataset')
    w3_2.plot_confusion_matrix(fn, 'Uniform prior, shape=3, ' + str(N_TRIALS) + ' trials/dataset',
                               save_name=fn[:-4] + '.png')

    # TEST 2

    fn = "./results/UU_D" + str(N_DATASETS) + "_shape5_T" + str(N_TRIALS) + ".csv"

    w3_2.model_selection(
        ramp_params_grid, step_params_grid,
        uniform_ramp_posterior, uniform_step_posterior,  # generating
        uniform_ramp_posterior, uniform_step_posterior,  # inference
        N_DATASETS=N_DATASETS, N_TRIALS=N_TRIALS,
        ramp_gamma_shape=9, step_gamma_shape=9,
        # format: generating: G/U; inference: G/U; n. datasets, n. trials
        # if G, append std_fraction on front
        save_to=os.path.join(os.getcwd(), fn)
    )

    w3_2.plot_heatmap(fn, 'Uniform prior, shape=5, ' + str(N_TRIALS) + ' trials/dataset')
    w3_2.plot_confusion_matrix(fn, 'Uniform prior, shape=5, ' + str(N_TRIALS) + ' trials/dataset',
                               save_name=fn[:-4] + '.png')