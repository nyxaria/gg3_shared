import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import os
import w3_2
import w3_utils

var = lambda a, b, frac: ((b-a) * frac) ** 2
mean = lambda a, b: (b+a)/2

def run_or_load_selection(filename, ramp_grid, step_grid, gen_ramp_post, gen_step_post, inf_ramp_post, inf_step_post, n_datasets, n_trials, ramp_shape=1, step_shape=1):
    if not os.path.exists(filename):
        print(f"Running model selection, saving to {filename}...")
        w3_2.model_selection(
            ramp_grid, step_grid,
            gen_ramp_post, gen_step_post, # generating
            inf_ramp_post, inf_step_post, # inference
            N_DATASETS=n_datasets, N_TRIALS=n_trials,
            ramp_gamma_shape=ramp_shape, step_gamma_shape=step_shape,
            save_to=filename
        )
    else:
        print(f"Found existing results file: {filename}")

    confmat_savename = f'plots/task_4_1_2_{os.path.basename(filename)[:-4]}_confmat.png'
    plot_title = f'Model Selection, {os.path.basename(filename)[:-4]}'
    
    ramp_accuracy, step_accuracy = w3_2.plot_confusion_matrix(filename, plot_title, save_name=confmat_savename, show=False)
    return ramp_accuracy, step_accuracy

if __name__ == "__main__":
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    K = 25
    T_MS = 100
    RH = 20
    M_GRID = 15
    X0 = 0.5
    N_DATASETS = 50 
    
    N_TRIALS_LIST = [10, 25, 50, 100, 200]
    STD_FRACTION_PRIOR_MISMATCH = 0.25
    SHAPE_LIKELIHOOD_MISMATCH = 3

    BETA_RANGE = (0, 4)
    SIGMA_RANGE = (0.04, 4)
    M_RANGE = (T_MS * 0.0, T_MS * 0.75)
    R_RANGE = (1, 6)

    ramp_param_specs = OD([
        ('beta', np.linspace(*BETA_RANGE, M_GRID)),
        ('sigma', np.exp(np.linspace(np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1]), M_GRID))),
        ('x0', X0), ('K', K), ('T', T_MS), ('Rh', RH)
    ])
    step_param_specs = OD([
        ('m', np.linspace(*M_RANGE, M_GRID)),
        ('r', np.arange(R_RANGE[0], R_RANGE[1] + 1)),
        ('x0', X0), ('K', K), ('T', T_MS), ('Rh', RH)
    ])

    ramp_params_grid = w3_utils.make_params_grid(ramp_param_specs)
    step_params_grid = w3_utils.make_params_grid(step_param_specs)

    uniform_ramp_posterior = w3_utils.uniform_prior(ramp_params_grid)
    uniform_step_posterior = w3_utils.uniform_prior(step_params_grid)

    gauss_ramp_posterior = w3_utils.gaussian_prior(ramp_params_grid,
        mu={"beta": mean(*BETA_RANGE), "sigma": mean(*SIGMA_RANGE)},
        cov={("beta", "beta"): var(*BETA_RANGE, STD_FRACTION_PRIOR_MISMATCH),
             ("sigma", "sigma"): var(*SIGMA_RANGE, STD_FRACTION_PRIOR_MISMATCH)})

    gauss_step_posterior = w3_utils.gaussian_prior(step_params_grid,
        mu={"m": mean(*M_RANGE), "r": 1},
        cov={("m", "m"): var(*M_RANGE, STD_FRACTION_PRIOR_MISMATCH),
             ("r", "r"): var(*R_RANGE, STD_FRACTION_PRIOR_MISMATCH)})

    results = {
        'baseline': {'ramp': [], 'step': []},
        'prior_mismatch': {'ramp': [], 'step': []},
        'likelihood_mismatch': {'ramp': [], 'step': []}
    }

    for n_trials in N_TRIALS_LIST:
        print(f"--- Processing N_TRIALS = {n_trials} ---")
        # Baseline
        print("Running Baseline (Poisson, Uniform Prior)")
        fn_base = f"./results/UU_D{N_DATASETS}_shape1_T{n_trials}.csv"
        r_acc, s_acc = run_or_load_selection(fn_base, ramp_params_grid, step_params_grid, uniform_ramp_posterior, uniform_step_posterior, uniform_ramp_posterior, uniform_step_posterior, N_DATASETS, n_trials, ramp_shape=1, step_shape=1)
        results['baseline']['ramp'].append(r_acc)
        results['baseline']['step'].append(s_acc)

        # Prior Mismatch
        print(f"Running Prior Mismatch (Poisson, Gaussian Prior SF={STD_FRACTION_PRIOR_MISMATCH})")
        fn_prior = f"./results/GU_D{N_DATASETS}_T{n_trials}_SF{STD_FRACTION_PRIOR_MISMATCH}.csv"
        r_acc, s_acc = run_or_load_selection(fn_prior, ramp_params_grid, step_params_grid, uniform_ramp_posterior, uniform_step_posterior, gauss_ramp_posterior, gauss_step_posterior, N_DATASETS, n_trials, ramp_shape=1, step_shape=1)
        results['prior_mismatch']['ramp'].append(r_acc)
        results['prior_mismatch']['step'].append(s_acc)

        # Likelihood Mismatch
        print(f"Running Likelihood Mismatch (Shape={SHAPE_LIKELIHOOD_MISMATCH}, Uniform Prior)")
        fn_like = f"./results/UU_D{N_DATASETS}_shape{SHAPE_LIKELIHOOD_MISMATCH}_T{n_trials}.csv"
        r_acc, s_acc = run_or_load_selection(fn_like, ramp_params_grid, step_params_grid, uniform_ramp_posterior, uniform_step_posterior, uniform_ramp_posterior, uniform_step_posterior, N_DATASETS, n_trials, ramp_shape=SHAPE_LIKELIHOOD_MISMATCH, step_shape=SHAPE_LIKELIHOOD_MISMATCH)
        results['likelihood_mismatch']['ramp'].append(r_acc)
        results['likelihood_mismatch']['step'].append(s_acc)

    # Plotting
    plt.figure(figsize=(12, 8))
    colors = {'baseline': 'k', 'prior_mismatch': 'b', 'likelihood_mismatch': 'r'}
    linestyles = {'ramp': '--', 'step': '-'}
    
    for mismatch_type, accs in results.items():
        label_prefix = mismatch_type.replace('_', ' ').title()
        plt.plot(N_TRIALS_LIST, accs['ramp'], color=colors[mismatch_type], linestyle=linestyles['ramp'], marker='o', label=f'{label_prefix} - Ramp Acc.')
        plt.plot(N_TRIALS_LIST, accs['step'], color=colors[mismatch_type], linestyle=linestyles['step'], marker='x', label=f'{label_prefix} - Step Acc.')

    plt.title('Model Selection Accuracy vs. Number of Trials under Mismatch')
    plt.xlabel('Number of Trials')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.xticks(N_TRIALS_LIST, labels=N_TRIALS_LIST)
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.ylim(0.4, 1.05)
    plt.savefig('plots/task_4_1_2_accuracy_vs_ntrials_mismatch.png')
    plt.show() 