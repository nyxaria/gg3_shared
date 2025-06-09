import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import scipy.special
import os
import pickle
import concurrent.futures
from tqdm import tqdm
import argparse

import w3_utils
from models_hmm import RampModelHMM, StepModelHMM
# make text larger
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})


def task_3_1_1_visualize_posterior_2d(model_type, true_params, n_trials, param_specs, params_grid, K=50, T=100, Rh=50,
                                      show=False):
    """
    Visualizes the 2D posterior probability for a given model with x0 fixed.
    """
    if model_type == 'ramp':
        Model, llh_func = RampModelHMM, w3_utils.ramp_LLH
        param1_name, param2_name = 'beta', 'sigma'
        xlabel, ylabel = r'$\beta$', r'$\sigma$'
    else:  # step
        Model, llh_func = StepModelHMM, w3_utils.step_LLH
        param1_name, param2_name = 'm', 'r'
        xlabel, ylabel = 'm', 'r'

    cache_dir = os.path.join("plots", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    true_param_str = "-".join(
        [f"{k}_{v:.2f}" for k, v in sorted(true_params.items()) if k in [param1_name, param2_name, 'x0']])
    grid_size = len(param_specs[param1_name])
    cache_filename = os.path.join(cache_dir,
                                  f"posterior_2d_{model_type}_{true_param_str}_N{n_trials}_M{grid_size}_K{K}.pickle")

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            norm_post_grid = pickle.load(f)
        print(f"Loaded from cache: {cache_filename}")
    else:
        if model_type == 'ramp':
            model = Model(beta=true_params['beta'], sigma=true_params['sigma'], x0=true_params['x0'], K=K, Rh=Rh)
            data, _, _ = model.simulate(Ntrials=n_trials, T=T)
        else:  # step
            model = Model(m=true_params['m'], r=true_params['r'], x0=true_params['x0'], Rh=Rh)
            data, _, _ = model.simulate_exact(Ntrials=n_trials, T=T, delay_compensation=True)

        llh_grid = llh_func(data, params_grid)
        prior_grid = w3_utils.uniform_prior(params_grid, log=True)
        norm_post_grid = w3_utils.norm_posterior(llh_grid, prior_grid, log=True)
        with open(cache_filename, 'wb') as f:
            pickle.dump(norm_post_grid, f)
        print(f"Saved to cache: {cache_filename}")

    param1_vals = param_specs[param1_name]
    param2_vals = param_specs[param2_name]

    delta1 = param1_vals[1] - param1_vals[0]
    delta2 = param2_vals[1] - param2_vals[0]
    extent = [
        param1_vals[0] - delta1 / 2, param1_vals[-1] + delta1 / 2,
        param2_vals[0] - delta2 / 2, param2_vals[-1] + delta2 / 2
    ]

    plt.figure(figsize=(8, 6))
    plt.imshow(np.exp(norm_post_grid).T, origin='lower', aspect='auto',
               extent=extent, cmap='viridis')
    plt.colorbar(label='Posterior Probability')
    plt.scatter(true_params[param1_name], true_params[param2_name], color='red', marker='x', s=100,
                label='True Parameters')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{model_type.capitalize()} Model\n2D Posterior (x0={true_params["x0"]:.2f}, N_trials={n_trials})')
    plt.legend()

    filename = f"plots/task_3_1_1_{model_type}_posterior_N{n_trials}.png"
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    T = 100
    Rh = 50
    N_TRIALS = 10

    # colors = ['blue', 'green', 'purple', 'orange']

    '''specs = OD([('m', [25, 75, 15]),
                ('r', [1, 6, 6]),
                ('T', T),
                ('Rh', Rh)])

    params_grid = w3_utils.make_params_grid(specs)

    step_true_params = [{'m': 35, 'r': 2, 'x0': 0.2},
                        {'m': 65, 'r': 2, 'x0': 0.2},
                        {'m': 35, 'r': 5, 'x0': 0.2},
                        {'m': 65, 'r': 5, 'x0': 0.2}]

    step_true_params = [{'m': 50, 'r': 2, 'x0': 0.2}]'''

    specs = OD([
        ('beta', np.linspace(0, 4, 10)),
        ('sigma', np.exp(np.linspace(np.log(0.04), np.log(4), 10))),
        ('x0', 0.2)
    ])

    params_grid = w3_utils.make_params_grid(specs)
    '''ramp_true_params = [{'beta': 0.75, 'sigma': 0.75, 'x0': 0.2},
                        {'beta': 2, 'sigma': 0.75, 'x0': 0.2},
                        {'beta': 3.25, 'sigma': 0.75, 'x0': 0.2},
                        {'beta': 0.75, 'sigma': 2, 'x0': 0.2},
                        {'beta': 2, 'sigma': 2, 'x0': 0.2},
                        {'beta': 3.25, 'sigma': 2, 'x0': 0.2},
                        {'beta': 0.75, 'sigma': 3.75, 'x0': 0.2},
                        {'beta': 2, 'sigma': 3.75, 'x0': 0.2},
                        {'beta': 3.25, 'sigma': 3.75, 'x0': 0.2}
                        ]'''
    '''ramp_true_params = [{'beta': 0.75, 'sigma': 0.5, 'x0': 0.2},
                        {'beta': 3, 'sigma': 0.5, 'x0': 0.2},
                        {'beta': 0.75, 'sigma': 2, 'x0': 0.2},
                        {'beta': 3, 'sigma': 2, 'x0': 0.2}]'''

    ramp_true_params = [
        params_grid[2, 2],
        params_grid[2, 7],
        params_grid[7, 7],
        params_grid[7, 2]
    ]

    posteriors = []
    expectations = []
    for tp in ramp_true_params: #
        print(tp)
        #
        data, _, _ = RampModelHMM(beta=tp['beta'], sigma=tp['sigma'], Rh=Rh).simulate(Ntrials=N_TRIALS,
                                                                                                          T=T)
        LLH_probs_grid = w3_utils.ramp_LLH(data, params_grid) #
        prior_probs_grid = w3_utils.uniform_prior(params_grid)

        npost = w3_utils.norm_posterior(LLH_probs_grid, prior_probs_grid)

        posteriors.append(npost)
        expectations.append(w3_utils.expectation(npost, params_grid))


    # TODO change
    m_values = w3_utils.get_param_values(params_grid, 'beta')
    r_values = w3_utils.get_param_values(params_grid, 'sigma')


    plt.figure(figsize=(10 * 0.75, 8 * 0.75))

    for idx, npost in enumerate(posteriors):
        cs = plt.contour(m_values, r_values, np.exp(npost), levels=5, cmap='Reds')

        # plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(label='Posterior Probability')

    plt.xlabel('beta')
    plt.ylabel('sigma')
    plt.title('Posterior probabilities - Ramp model, Trials = ' + str(N_TRIALS))
    for i, exp in enumerate(expectations):
        true_point = (ramp_true_params[i]['beta'], ramp_true_params[i]['sigma'])
        exp_point = (exp['beta'], exp['sigma'])

        # Plot points
        plt.scatter(true_point[0], true_point[1],
                    color='red', marker='x', s=100,
                    label='True Parameters' if i == 0 else None)

        plt.scatter(exp_point[0], exp_point[1],
                    color='blue', marker='o', s=100,
                    label='Expected Parameters' if i == 0 else None)

        # Calculate Euclidean distance
        euc_dist = np.sqrt((true_point[0] - exp_point[0]) ** 2 +
                           (true_point[1] - exp_point[1]) ** 2)

        # Draw line between points
        plt.plot([true_point[0], exp_point[0]],
                 [true_point[1], exp_point[1]],
                 'k--', alpha=0.5)

        # Add distance label at the middle of the line
        mid_x = (true_point[0] + exp_point[0]) / 2
        mid_y = (true_point[1] + exp_point[1]) / 2
        plt.annotate(f'{euc_dist:.2f}',
                     (mid_x, mid_y),
                     xytext=(5, 5),
                     textcoords='offset points')

plt.legend()

plt.show()
