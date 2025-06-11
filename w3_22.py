import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import scipy.special
import os
from scipy.interpolate import griddata
import w3_2
import w3_utils
from models_hmm import RampModelHMM
import pandas as pd


var = lambda a, b, frac: ((b-a) * frac) ** 2
mean = lambda a, b: (b+a)/2

def run_and_plot_selection(filename, ramp_grid, step_grid, gen_ramp_post, gen_step_post, inf_ramp_post, inf_step_post, n_datasets, n_trials, plot_title_prefix, Rh, x0):
    
    if not os.path.exists(filename):
        print(f"Running model selection, saving to {filename}...")
        w3_2.model_selection(
            ramp_grid, step_grid,
            gen_ramp_post, gen_step_post, # generating
            inf_ramp_post, inf_step_post, # inference
            N_DATASETS=n_datasets, N_TRIALS=n_trials,
            save_to=filename
        )
    else:
        print(f"Found existing results file: {filename}")


    heatmap_savename = f'plots/task_3_2_2_{os.path.basename(filename)[:-4]}_heatmap.png'
    confmat_savename = f'plots/task_3_2_2_{os.path.basename(filename)[:-4]}_confmat.png'
    plot_title = f'{plot_title_prefix}, {n_trials} trials/dataset, Rh={Rh}, x0={x0}'
    
    w3_2.plot_heatmap(filename, plot_title, save_name=heatmap_savename, show=False)
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

    step_param_specs = OD([('m', np.linspace(*M_RANGE, M_GRID)),
                           ('r', np.linspace(*R_RANGE, 6).astype(int)),
                           ('x0', X0),
                           ('K', K),
                           ('T', T_MS),
                           ('Rh', RH)])

    ramp_params_grid = w3_utils.make_params_grid(ramp_param_specs)
    step_params_grid = w3_utils.make_params_grid(step_param_specs)

    uniform_ramp_posterior = w3_utils.uniform_prior(ramp_params_grid)
    uniform_step_posterior = w3_utils.uniform_prior(step_params_grid)



    N_DATASETS = 50
    N_TRIALS_LIST = [5, 10, 15, 20, 30, 50]
    sigma_fract_list = [0.125, 0.25, 0.5]
    accuracies = {'Uniform': []}
    for sf in sigma_fract_list:
        accuracies[f'Gaussian, SF={sf}'] = []

    for N_TRIALS in N_TRIALS_LIST:
        print("--------------------------------")
        print("N_TRIALS: ", N_TRIALS)
        print("Running Uniform prior")
        # TEST 1

        fn = "./results/UU_D" + str(N_DATASETS) + "_T" + str(N_TRIALS) + ".csv"
        ramp_accuracy, step_accuracy = run_and_plot_selection(
            filename=fn,
            ramp_grid=ramp_params_grid,
            step_grid=step_params_grid,
            gen_ramp_post=uniform_ramp_posterior,
            gen_step_post=uniform_step_posterior,
            inf_ramp_post=uniform_ramp_posterior,
            inf_step_post=uniform_step_posterior,
            n_datasets=N_DATASETS,
            n_trials=N_TRIALS,
            plot_title_prefix='Uniform prior',
            Rh=RH,
            x0=X0
        )
        accuracies['Uniform'].append((ramp_accuracy+step_accuracy)/2)

        # TEST 2

        for STD_FRACTION in sigma_fract_list:
            print("Running Gaussian prior, STD_FRACTION: ", STD_FRACTION)
            gauss_ramp_posterior = w3_utils.gaussian_prior(
                ramp_params_grid,
                mu={
                    "beta": mean(*BETA_RANGE),
                    "sigma": mean(*SIGMA_RANGE)
                },
                cov={
                    ("beta", "beta"): var(*BETA_RANGE, STD_FRACTION),
                    ("sigma", "sigma"): var(*SIGMA_RANGE, STD_FRACTION)
                })

            gauss_step_posterior = w3_utils.gaussian_prior(
                step_params_grid,
                mu={
                    "m": mean(*M_RANGE),
                    "r": 1
                },
                cov={
                    ("m", "m"): var(*M_RANGE, STD_FRACTION),
                    ("r", "r"): var(*R_RANGE, STD_FRACTION)
                })

            fn = "./results/GU_D" + str(N_DATASETS) + "_T" + str(N_TRIALS) + "_SF" + str(STD_FRACTION) + ".csv"

            ramp_accuracy, step_accuracy = run_and_plot_selection(
                filename=fn,
                ramp_grid=ramp_params_grid,
                step_grid=step_params_grid,
                gen_ramp_post=uniform_ramp_posterior,
                gen_step_post=uniform_step_posterior,
                inf_ramp_post=gauss_ramp_posterior,
                inf_step_post=gauss_step_posterior,
                n_datasets=N_DATASETS,
                n_trials=N_TRIALS,
                plot_title_prefix=r'Gaussian prior, $\sigma_{frac}$=' + str(STD_FRACTION),
                Rh=RH,
                x0=X0
            )
            accuracies[f'Gaussian, SF={STD_FRACTION}'].append((ramp_accuracy+step_accuracy)/2)

    plt.figure(figsize=(10, 6))
    for label, accs in accuracies.items():
        plt.plot(N_TRIALS_LIST, accs, marker='o', linestyle='-', label=label)
    
    plt.xlabel("Number of Trials (N_TRIALS)")
    plt.ylabel("Overall HMM Accuracy")
    plt.title("HMM Accuracy vs. Number of Trials for Different Priors")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.45, 1.05)
    plt.savefig('plots/task_3_2_2_accuracy_vs_n_trials.png')
    plt.show()