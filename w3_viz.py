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

def task_3_1_1_visualize_posterior_2d(model_type, true_params, n_trials, param_specs, params_grid, K=50, T=100, Rh=50, show=False):
    """
    Visualizes the 2D posterior probability for a given model with x0 fixed.
    """
    if model_type == 'ramp':
        Model, llh_func = RampModelHMM, w3_utils.ramp_LLH
        param1_name, param2_name = 'beta', 'sigma'
        xlabel, ylabel = r'$\beta$', r'$\sigma$'
    else: # step
        Model, llh_func = StepModelHMM, w3_utils.step_LLH
        param1_name, param2_name = 'm', 'r'
        xlabel, ylabel = 'm', 'r'

    cache_dir = os.path.join("plots", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    true_param_str = "-".join([f"{k}_{v:.2f}" for k, v in sorted(true_params.items()) if k in [param1_name, param2_name, 'x0']])
    grid_size = len(param_specs[param1_name])
    cache_filename = os.path.join(cache_dir, f"posterior_2d_{model_type}_{true_param_str}_N{n_trials}_M{grid_size}_K{K}.pickle")

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            norm_post_grid = pickle.load(f)
        print(f"Loaded from cache: {cache_filename}")
    else:
        if model_type == 'ramp':
            model = Model(beta=true_params['beta'], sigma=true_params['sigma'], x0=true_params['x0'], K=K, Rh=Rh)
            data, _, _ = model.simulate(Ntrials=n_trials, T=T)
        else: # step
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
    plt.scatter(true_params[param1_name], true_params[param2_name], color='red', marker='x', s=100, label='True Parameters')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{model_type.capitalize()} Model\n2D Posterior (x0={true_params["x0"]:.2f}, N_trials={n_trials})')
    plt.legend()
    
    filename = f"plots/task_3_1_1_{model_type}_posterior_N{n_trials}.png"
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()

def _calculate_map_error_single(args):
    """
    Helper function for parallel execution of MAP error calculation for a single parameter set.
    """
    model_type, true_params, inference_params_grid, cache_dir, K, T, Rh, n_trials = args

    if model_type == 'ramp':
        param_names = ['beta', 'sigma']
    else: # step
        param_names = ['m', 'r']

    param_str = "-".join([f"{k}_{v:.2f}" for k, v in sorted(true_params.items()) if k in param_names + ['x0']])
    cache_filename = os.path.join(cache_dir, f"{param_str}.pickle")

    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'rb') as f:
                map_error, expectation_error = pickle.load(f)
            return map_error, expectation_error
        except (ValueError, EOFError):
            pass

    if model_type == 'ramp':
        model = RampModelHMM(beta=true_params['beta'], sigma=true_params['sigma'], x0=true_params['x0'], K=K, Rh=Rh)
        data, _, _ = model.simulate(Ntrials=n_trials, T=T)
        llh_grid = w3_utils.ramp_LLH(data, inference_params_grid)
    else:
        model = StepModelHMM(m=true_params['m'], r=true_params['r'], x0=true_params['x0'], Rh=Rh)
        data, _, _ = model.simulate_exact(Ntrials=n_trials, T=T, delay_compensation=True)
        llh_grid = w3_utils.step_LLH(data, inference_params_grid)

    prior_grid = w3_utils.uniform_prior(inference_params_grid, log=True)
    norm_post_grid = w3_utils.norm_posterior(llh_grid, prior_grid, log=True)

    # MAP error
    map_indices = np.unravel_index(np.argmax(norm_post_grid), norm_post_grid.shape)
    map_params = inference_params_grid[map_indices]
    map_error = np.sqrt(sum((true_params[p] - map_params[p])**2 for p in param_names))

    # expectation error
    posterior_means = w3_utils.expectation(norm_post_grid, inference_params_grid, log=True)
    expectation_error = np.sqrt(sum((true_params[p] - posterior_means[p])**2 for p in param_names))

    with open(cache_filename, 'wb') as f:
        pickle.dump((map_error, expectation_error), f)

    return map_error, expectation_error

def task_3_1_1_visualize_map_error(model_type, true_param_specs, inference_param_specs, n_trials, K=50, T=100, Rh=50, show=False):
    """
    For a grid of true parameters, simulates data, finds the MAP estimate,
    and visualizes the error between the true parameters and the MAP estimate.
    Caches results to avoid re-computation and runs in parallel.
    """
    if model_type == 'ramp':
        param1_name, param2_name = 'beta', 'sigma'
        xlabel, ylabel = r'$\beta$', r'$\sigma$'
    else: # step
        param1_name, param2_name = 'm', 'r'
        xlabel, ylabel = 'm', 'r'

    true_params_grid = w3_utils.make_params_grid(true_param_specs)
    inference_params_grid = w3_utils.make_params_grid(inference_param_specs)

    M_inference = len(inference_param_specs[param1_name])
    cache_dir = os.path.join("plots", "cache", f"map_error_{model_type}_{true_param_specs['x0']}_{M_inference}_{K}_{n_trials}_{T}_{Rh}")
    os.makedirs(cache_dir, exist_ok=True)
    
    tasks = [
        (model_type, param_dict.item(), inference_params_grid, cache_dir, K, T, Rh, n_trials)
        for param_dict in np.nditer(true_params_grid, flags=['refs_ok'])
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(_calculate_map_error_single, tasks), total=len(tasks)))
    
    map_errors, expectation_errors = zip(*results)

    map_error_grid = np.array(map_errors).reshape(true_params_grid.shape)
    expectation_error_grid = np.array(expectation_errors).reshape(true_params_grid.shape)

    true_param1_vals = true_param_specs[param1_name]
    true_param2_vals = true_param_specs[param2_name]

    delta_param1 = true_param1_vals[1] - true_param1_vals[0]
    delta_param2 = true_param2_vals[1] - true_param2_vals[0]
    extent = [
        true_param1_vals[0] - delta_param1 / 2, true_param1_vals[-1] + delta_param1 / 2,
        true_param2_vals[0] - delta_param2 / 2, true_param2_vals[-1] + delta_param2 / 2
    ]
    
    # MAP error
    plt.figure(figsize=(8, 6))
    im = plt.imshow(map_error_grid.T, origin='lower', aspect='auto', extent=extent, cmap='magma')
    plt.colorbar(im, label='MAP Estimation Error (Euclidean Distance)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{model_type.capitalize()} Model\nMAP Estimation Error (N_trials={n_trials})')
    
    filename = f"plots/task_3_1_1_{model_type}_map_error_x{true_param_specs['x0']}_N{n_trials}.png"
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()

    # expectation error
    plt.figure(figsize=(8, 6))
    im = plt.imshow(expectation_error_grid.T, origin='lower', aspect='auto', extent=extent, cmap='magma')
    plt.colorbar(im, label='Expectation Error (Euclidean Distance)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{model_type.capitalize()} Model\nPosterior Mean Estimation Error (N_trials={n_trials})')
    
    filename = f"plots/task_3_1_1_{model_type}_expectation_error_N{n_trials}.png"
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()
    return expectation_error_grid


def _analyze_estimation_worker(args):
    model_type, true_params, n_trials, params_grid, param_names, K, T, Rh = args

    if model_type == 'ramp':
        Model, llh_func = RampModelHMM, w3_utils.ramp_LLH
    else:  # step
        Model, llh_func = StepModelHMM, w3_utils.step_LLH

    if model_type == 'ramp':
        model = Model(beta=true_params['beta'], sigma=true_params['sigma'], x0=true_params['x0'], K=K, Rh=Rh)
        data, _, _ = model.simulate(Ntrials=n_trials, T=T)
    else:
        model = Model(m=true_params['m'], r=true_params['r'], x0=true_params['x0'], Rh=Rh)
        data, _, _ = model.simulate_exact(Ntrials=n_trials, T=T, delay_compensation=True)

    llh_grid = llh_func(data, params_grid)
    prior_grid = w3_utils.uniform_prior(params_grid, log=True)
    norm_post_grid = w3_utils.norm_posterior(llh_grid, prior_grid, log=True)

    posterior_means = w3_utils.expectation(norm_post_grid, params_grid, log=True)
    posterior_std_devs = w3_utils.posterior_std_dev(norm_post_grid, params_grid, posterior_means, log=True)

    errors = {}
    stds = {}
    for p in param_names:
        errors[p] = np.abs(posterior_means[p] - true_params[p])
        stds[p] = posterior_std_devs.get(p, np.nan)

    return errors, stds

def task_3_1_2_analyze_estimation_2d(model_type, true_params, n_trials_list, params_grid, n_datasets=1, K=50, T=100, Rh=50, show=False):
    if model_type == 'ramp':
        param_names = ['beta', 'sigma']
        Model, llh_func = RampModelHMM, w3_utils.ramp_LLH
    else: # step
        param_names = ['m', 'r']
        Model, llh_func = StepModelHMM, w3_utils.step_LLH

    cache_dir = os.path.join("plots", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    true_param_str = "-".join([f"{k}_{v:.2f}" for k, v in sorted(true_params.items()) if k in param_names + ['x0']])
    grid_shape_str = "_".join(map(str, params_grid.shape))
    ntrials_str = "_".join(map(str, n_trials_list))
    cache_filename = os.path.join(cache_dir, f"analyze_2d_{model_type}_{true_param_str}_G{grid_shape_str}_N{ntrials_str}_D{n_datasets}_K{K}.pickle")

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            estimation_errors, posterior_stds = pickle.load(f)
        print(f"Loaded from cache: {cache_filename}")
    else:
        estimation_errors = {p: [] for p in param_names}
        posterior_stds = {p: [] for p in param_names}
    
        for n_trials in n_trials_list:
            print(f"Processing n_trials = {n_trials}")
            tasks = [(model_type, true_params, n_trials, params_grid, param_names, K, T, Rh) for _ in range(n_datasets)]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(tqdm(executor.map(_analyze_estimation_worker, tasks), total=len(tasks), desc=f"Averaging over datasets"))

            errors_for_n_trials = {p: [] for p in param_names}
            stds_for_n_trials = {p: [] for p in param_names}

            for errors, stds in results:
                for p in param_names:
                    errors_for_n_trials[p].append(errors[p])
                    stds_for_n_trials[p].append(stds[p])
            
            for p in param_names:
                estimation_errors[p].append(np.mean(errors_for_n_trials[p]))
                posterior_stds[p].append(np.mean(stds_for_n_trials[p]))
        
        with open(cache_filename, 'wb') as f:
            pickle.dump((estimation_errors, posterior_stds), f)
        print(f"Saved to cache: {cache_filename}")

    plt.figure(figsize=(10, 7))
    colors = ['#1f77b4', '#ff7f0e']
    param_labels = {'beta': r'$\beta$', 'sigma': r'$\sigma$', 'm': 'm', 'r': 'r'}

    for i, p in enumerate(param_names):
        error = np.array(estimation_errors[p])
        std_dev = np.array(posterior_stds[p])
        
        plt.plot(n_trials_list, error, 'o-', color=colors[i], label=f'Parameter {param_labels.get(p, p)}')
        plt.fill_between(n_trials_list, error - std_dev, error + std_dev, color=colors[i], alpha=0.2)

    plt.xlabel('Number of Trials')
    plt.ylabel('Error / Std Dev')
    plt.suptitle(f'{model_type.capitalize()} Model\nEstimation Quality vs. Number of Trials (N Datasets={n_datasets})')
    plt.legend()
    plt.xscale('log')
    # plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    
    filepath = f"plots/task_3_1_2_{model_type}_x{true_params['x0']}_ND{n_datasets}_Nerror_vs_ntrials.png"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return estimation_errors

def task_3_1_3_visualize_posterior_marginal(model_type, true_params, n_trials, param_specs, params_grid, K=50, T=100, Rh=50, show=False):
    if model_type == 'ramp':
        Model, llh_func = RampModelHMM, w3_utils.ramp_LLH
        param_names = ['beta', 'sigma', 'x0']
        labels = [r'$\beta$', r'$\sigma$', 'x0']
    else: # step
        Model, llh_func = StepModelHMM, w3_utils.step_LLH
        param_names = ['m', 'r', 'x0']
        labels = ['m', 'r', 'x0']

    cache_dir = os.path.join("plots", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    true_param_str = "-".join([f"{k}_{v:.2f}" for k, v in sorted(true_params.items()) if k in param_names])
    grid_size = len(param_specs[param_names[0]])
    cache_filename = os.path.join(cache_dir, f"posterior_3d_{model_type}_{true_param_str}_N{n_trials}_M{grid_size}_K{K}.pickle")

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            norm_post_grid = pickle.load(f)
        print(f"Loaded from cache: {cache_filename}")
    else:
        if model_type == 'ramp':
            model = Model(beta=true_params['beta'], sigma=true_params['sigma'], x0=true_params['x0'], K=K, Rh=Rh)
            data, _, _ = model.simulate(Ntrials=n_trials, T=T)
        else:
            model = Model(m=true_params['m'], r=true_params['r'], x0=true_params['x0'], Rh=Rh)
            data, _, _ = model.simulate_exact(Ntrials=n_trials, T=T, delay_compensation=True)

        llh_grid = llh_func(data, params_grid)
        prior_grid = w3_utils.uniform_prior(params_grid, log=True)
        norm_post_grid = w3_utils.norm_posterior(llh_grid, prior_grid, log=True)
        with open(cache_filename, 'wb') as f:
            pickle.dump(norm_post_grid, f)
        print(f"Saved to cache: {cache_filename}")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    marginal_pairs = [(0, 1), (0, 2), (1, 2)]
    for i, (p1_idx, p2_idx) in enumerate(marginal_pairs):
        other_idx = 3 - p1_idx - p2_idx
        marginal_post = scipy.special.logsumexp(norm_post_grid, axis=other_idx)
        
        p1_vals = param_specs[param_names[p1_idx]]
        p2_vals = param_specs[param_names[p2_idx]]
        
        delta1 = p1_vals[1] - p1_vals[0]
        delta2 = p2_vals[1] - p2_vals[0]
        extent = [p1_vals[0] - delta1/2, p1_vals[-1] + delta1/2, 
                  p2_vals[0] - delta2/2, p2_vals[-1] + delta2/2]

        im = axs[i].imshow(np.exp(marginal_post).T, origin='lower', aspect='auto', extent=extent, cmap='viridis')
        axs[i].scatter(true_params[param_names[p1_idx]], true_params[param_names[p2_idx]], c='r', marker='x')
        axs[i].set_xlabel(labels[p1_idx])
        axs[i].set_ylabel(labels[p2_idx])
        axs[i].set_title(f'P({param_names[p1_idx]}, {param_names[p2_idx]})')
        fig.colorbar(im, ax=axs[i], orientation='vertical')

    fig.suptitle(f'{model_type.capitalize()} Model\n2D Marginal Posteriors (N_trials={n_trials})', fontsize=16)
    filepath = f"plots/task_3_1_3_{model_type}_marginals_N{n_trials}.png"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def task_3_1_3_analyze_estimation_3d(model_type, true_params, n_trials_list, params_grid, n_datasets=1, K=50, T=100, Rh=50, show=False):
    if model_type == 'ramp':
        param_names = ['beta', 'sigma', 'x0']
        Model, llh_func = RampModelHMM, w3_utils.ramp_LLH
    else: # step
        param_names = ['m', 'r', 'x0']
        Model, llh_func = StepModelHMM, w3_utils.step_LLH

    cache_dir = os.path.join("plots", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    true_param_str = "-".join([f"{k}_{v:.2f}" for k, v in sorted(true_params.items()) if k in param_names])
    grid_shape_str = "_".join(map(str, params_grid.shape))
    ntrials_str = "_".join(map(str, n_trials_list))
    cache_filename = os.path.join(cache_dir, f"analyze_3d_{model_type}_{true_param_str}_G{grid_shape_str}_N{ntrials_str}_D{n_datasets}_K{K}.pickle")

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            estimation_errors, posterior_stds = pickle.load(f)
        print(f"Loaded from cache: {cache_filename}")
    else:
        estimation_errors = {p: [] for p in param_names}
        posterior_stds = {p: [] for p in param_names}

        for n_trials in n_trials_list:
            print(f"Processing 3D n_trials = {n_trials}")
            tasks = [(model_type, true_params, n_trials, params_grid, param_names, K, T, Rh) for _ in range(n_datasets)]

            with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(tqdm(executor.map(_analyze_estimation_worker, tasks), total=len(tasks), desc=f"Averaging over datasets"))
            
            errors_for_n_trials = {p: [] for p in param_names}
            stds_for_n_trials = {p: [] for p in param_names}

            for errors, stds in results:
                for p in param_names:
                    errors_for_n_trials[p].append(errors[p])
                    stds_for_n_trials[p].append(stds[p])

            for p in param_names:
                estimation_errors[p].append(np.mean(errors_for_n_trials[p]))
                posterior_stds[p].append(np.mean(stds_for_n_trials[p]))

        with open(cache_filename, 'wb') as f:
            pickle.dump((estimation_errors, posterior_stds), f)
        print(f"Saved to cache: {cache_filename}")

    plt.figure(figsize=(18, 5))
    for i, p in enumerate(param_names):
        plt.subplot(1, 3, i + 1)
        plt.plot(n_trials_list, estimation_errors[p], 'o-', label='Estimation Error')
        plt.plot(n_trials_list, posterior_stds[p], 's--', label='Posterior Std Dev')
        plt.xlabel('Number of Trials')
        plt.ylabel('Error / Std Dev')
        plt.title(f'Parameter: {p}')
        plt.legend()
        plt.xscale('log')
    plt.suptitle(f'{model_type.capitalize()} Model\nEstimation Quality vs. Number of Trials (3D, N Datasets={n_datasets})')
    filepath = f"plots/task_3_1_3_{model_type}_error_vs_ntrials_ND{n_datasets}.png"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    K = 25
    T_MS = 100
    RH = 50

    map_x0 = 0.0

    N_MAP_TRIALS = 400
    N_DATASETS_AVG = 10

    M_2D_GRID = 31
    M_TRUE_GRID = 11
    M_INFERENCE_GRID = 31


    ramp_param_2d = OD([
        ('beta', np.linspace(0, 4, M_2D_GRID)),
        ('sigma', np.linspace(0.04, 4, M_2D_GRID)),
        ('x0', 0.2),
        ('K', K), ('T', T_MS), ('Rh', RH)
    ])

    step_param_2d = OD([
        ('m', np.linspace(0, T_MS*3/4, M_2D_GRID)),
        ('r', np.linspace(1, 6, 6).astype(int)),
        ('x0', 0.2),
        ('K', K), ('T', T_MS), ('Rh', RH)
    ])
    #
    # RAMP
    #

    print("\n--- RAMP MODEL ANALYSIS ---")
    ramp_true_params = {'beta': ramp_param_2d['beta'][M_2D_GRID//3], 'sigma': ramp_param_2d['sigma'][3], 'x0': 0.2}
    print(f"Ramp True params: {ramp_true_params}")
    # 3.1.1
    
    ramp_params_grid_2d = w3_utils.make_params_grid(ramp_param_2d)
    task_3_1_1_visualize_posterior_2d('ramp', ramp_true_params, n_trials=10, param_specs=ramp_param_2d, params_grid=ramp_params_grid_2d, K=K, T=T_MS, Rh=RH, show=args.show)
    task_3_1_1_visualize_posterior_2d('ramp', ramp_true_params, n_trials=50, param_specs=ramp_param_2d, params_grid=ramp_params_grid_2d, K=K, T=T_MS, Rh=RH, show=args.show)
    task_3_1_1_visualize_posterior_2d('ramp', ramp_true_params, n_trials=100, param_specs=ramp_param_2d, params_grid=ramp_params_grid_2d, K=K, T=T_MS, Rh=RH, show=args.show)
    task_3_1_1_visualize_posterior_2d('ramp', ramp_true_params, n_trials=400, param_specs=ramp_param_2d, params_grid=ramp_params_grid_2d, K=K, T=T_MS, Rh=RH, show=args.show)

    # 3.1.1
    ramp_true_param_specs = OD([
        ('beta', np.linspace(0, 4, M_TRUE_GRID)),
        ('sigma', np.linspace(0.04, 4, M_TRUE_GRID)),
        ('x0', map_x0),
        ('K', K), ('T', T_MS), ('Rh', RH)
    ])
    ramp_inference_param_specs = OD([
        ('beta', np.linspace(0, 4, M_INFERENCE_GRID)),
        ('sigma', np.linspace(0.04, 4, M_INFERENCE_GRID)),
        ('x0', map_x0),
        ('K', K), ('T', T_MS), ('Rh', RH)
    ])
    task_3_1_1_visualize_map_error('ramp', ramp_true_param_specs, ramp_inference_param_specs, n_trials=N_MAP_TRIALS, K=K, T=T_MS, Rh=RH, show=args.show)

    # 3.1.2

    n_trials_list = [1, 5, 10, 20, 50, 100, 200, 400]
    ramp_params_grid_2d = w3_utils.make_params_grid(ramp_inference_param_specs)
    task_3_1_2_analyze_estimation_2d('ramp', {'beta': ramp_param_2d['beta'][M_2D_GRID//2], 'sigma': ramp_param_2d['sigma'][2], 'x0': 0.0}, n_trials_list, ramp_params_grid_2d, n_datasets=N_DATASETS_AVG, K=K, T=T_MS, Rh=RH, show=args.show or True)
    
    
    # 3.1.3
    ramp_params_specs_3d = OD([
        ('beta', np.linspace(0, 4, M_INFERENCE_GRID)),
        ('sigma', np.linspace(0.04, 4, M_INFERENCE_GRID)),
        ('x0', np.linspace(0, 0.5, M_INFERENCE_GRID)),
        ('K', K), ('T', T_MS), ('Rh', RH)
    ])
    ramp_params_grid_3d = w3_utils.make_params_grid(ramp_params_specs_3d)
    # task_3_1_3_visualize_posterior_marginal('ramp', ramp_true_params, n_trials=100, param_specs=ramp_params_specs_3d, params_grid=ramp_params_grid_3d, K=K, T=T_MS, Rh=RH, show=args.show)
    
    ramp_params_specs_3d = OD([
        ('beta', np.linspace(0, 4, M_TRUE_GRID)),
        ('sigma', np.linspace(0.04, 4, M_TRUE_GRID)),
        ('x0', np.linspace(0, 0.5, M_TRUE_GRID)),
        ('K', K), ('T', T_MS), ('Rh', RH)
    ])
    ramp_params_grid_3d = w3_utils.make_params_grid(ramp_params_specs_3d)
    # task_3_1_3_analyze_estimation_3d('ramp', ramp_true_params, n_trials_list=[50, 100, 200], params_grid=ramp_params_grid_3d, n_datasets=N_DATASETS_AVG, K=K, T=T_MS, Rh=RH, show=args.show)

    #
    # STEP
    #

    print("\n--- STEP MODEL ANALYSIS ---")
    step_true_params = {'m': step_param_2d['m'][M_2D_GRID//2], 'r': step_param_2d['r'][2], 'x0': 0.2}
    print(f"Step True params: {step_true_params}")
    # 3.1.1
    
    step_params_grid_2d = w3_utils.make_params_grid(step_param_2d)
    task_3_1_1_visualize_posterior_2d('step', step_true_params, n_trials=10, param_specs=step_param_2d, params_grid=step_params_grid_2d, K=K, T=T_MS, Rh=RH, show=args.show)
    task_3_1_1_visualize_posterior_2d('step', step_true_params, n_trials=50, param_specs=step_param_2d, params_grid=step_params_grid_2d, K=K, T=T_MS, Rh=RH, show=args.show)
    task_3_1_1_visualize_posterior_2d('step', step_true_params, n_trials=100, param_specs=step_param_2d, params_grid=step_params_grid_2d, K=K, T=T_MS, Rh=RH, show=args.show)
    task_3_1_1_visualize_posterior_2d('step', step_true_params, n_trials=400, param_specs=step_param_2d, params_grid=step_params_grid_2d, K=K, T=T_MS, Rh=RH, show=args.show)

    # 3.1.1
    step_true_param_specs = OD([
        ('m', np.linspace(0, T_MS*3/4, M_TRUE_GRID)),
        ('r', np.linspace(1, 6, 6).astype(int)),
        ('x0', map_x0),
        ('K', K), ('T', T_MS), ('Rh', RH)
    ])
    step_inference_param_specs = OD([
        ('m', np.linspace(0, T_MS*3/4, M_INFERENCE_GRID)),
        ('r', np.linspace(1, 6, 6).astype(int)),
        ('x0', map_x0),
        ('K', K), ('T', T_MS), ('Rh', RH)
    ])
    task_3_1_1_visualize_map_error('step', step_true_param_specs, step_inference_param_specs, n_trials=N_MAP_TRIALS, K=K, T=T_MS, Rh=RH, show=args.show)


    # 3.1.2
    step_true_params = {'m': step_param_2d['m'][M_2D_GRID//2], 'r': step_param_2d['r'][2], 'x0': 0.0}

    n_trials_list = [1, 5, 10, 20, 50, 100, 200, 400]
    step_params_grid_2d = w3_utils.make_params_grid(step_inference_param_specs)
    task_3_1_2_analyze_estimation_2d('step',  {'m': step_param_2d['m'][M_2D_GRID//2], 'r': step_param_2d['r'][-2], 'x0': 0.0}, n_trials_list, step_params_grid_2d,
                                      n_datasets=N_DATASETS_AVG*3, K=K, T=T_MS, Rh=RH,
                                      show=args.show or True)

    # 3.1.3
    step_params_specs_3d = OD([
        ('m', np.linspace(0, T_MS*3/4, M_INFERENCE_GRID)),
        ('r', np.linspace(1, 6, 6).astype(int)),
        ('x0', np.linspace(0, 0.5, M_INFERENCE_GRID)),
        ('K', K), ('T', T_MS), ('Rh', RH)
    ])
    step_params_grid_3d = w3_utils.make_params_grid(step_params_specs_3d)
    # task_3_1_3_visualize_posterior_marginal('step', step_true_params, n_trials=100, param_specs=step_params_specs_3d, params_grid=step_params_grid_3d, K=K, T=T_MS, Rh=RH, show=args.show)
    
    step_params_specs_3d = OD([
        ('m', np.linspace(0, T_MS*3/4, M_TRUE_GRID)),
        ('r', np.linspace(1, 6, 6).astype(int)),
        ('x0', np.linspace(0, 0.5, M_TRUE_GRID)),
        ('K', K), ('T', T_MS), ('Rh', RH)
    ])
    step_params_grid_3d = w3_utils.make_params_grid(step_params_specs_3d)
    # task_3_1_3_analyze_estimation_3d('step', step_true_params, n_trials_list=[50, 100, 200], params_grid=step_params_grid_3d, n_datasets=N_DATASETS_AVG, K=K, T=T_MS, Rh=RH, show=args.show)

    # --- Combined Error Plot ---
    print("\n--- Generating Combined Expectation Error vs N_trials Plot ---")
    n_trials_list = [50, 100, 200, 400]
    
    ramp_avg_errors, ramp_std_errors = [], []
    print("Analyzing Ramp Model...")
    for n in n_trials_list:
        error_grid = task_3_1_1_visualize_map_error('ramp', ramp_true_param_specs, ramp_inference_param_specs, n_trials=n, K=K, T=T_MS, Rh=RH, show=args.show)
        ramp_avg_errors.append(np.mean(error_grid))
        ramp_std_errors.append(np.std(error_grid))

    step_avg_errors, step_std_errors = [], []
    print("Analyzing Step Model...")
    for n in n_trials_list:
        error_grid = task_3_1_1_visualize_map_error('step', step_true_param_specs, step_inference_param_specs, n_trials=n, K=K, T=T_MS, Rh=RH, show=args.show)
        step_avg_errors.append(np.mean(error_grid))
        step_std_errors.append(np.std(error_grid))

    plt.figure(figsize=(10, 6))
    
    ramp_avg_errors = np.array(ramp_avg_errors)
    ramp_std_errors = np.array(ramp_std_errors)
    step_avg_errors = np.array(step_avg_errors)
    step_std_errors = np.array(step_std_errors)

    ramp_line, = plt.plot(n_trials_list, ramp_avg_errors, 'o-', label='Ramp Model Avg. Expectation Error')
    plt.fill_between(n_trials_list, ramp_avg_errors - ramp_std_errors, ramp_avg_errors + ramp_std_errors, color=ramp_line.get_color(), alpha=0.2)

    step_line, = plt.plot(n_trials_list, step_avg_errors, 's--', label='Step Model Avg. Expectation Error')
    plt.fill_between(n_trials_list, step_avg_errors - step_std_errors, step_avg_errors + step_std_errors, color=step_line.get_color(), alpha=0.2)

    plt.xlabel('Number of Trials')
    plt.ylabel('Avg. Expectation Error (Euclidean Norm)')
    plt.title('Model Expectation Error vs. Number of Trials')
    plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    
    combined_filepath = f"plots/task_3_1_1_combined_avg_expectation_error_vs_ntrials.png"
    plt.savefig(combined_filepath, dpi=300)
    # if args.show:
    plt.show()
    plt.close()
