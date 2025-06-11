import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import scipy.special
import os
import scipy
from scipy.interpolate import griddata
import w3_2
import w3_utils
from models_hmm import RampModelHMM
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
import multiprocessing
from tqdm import tqdm
import pandas as pd
from ML_models import StepRampClassifier, compute_summary_statistics
from models_hmm import RampModelHMM, StepModelHMM



def _model_selection_worker_with_14(args):
    """Helper function for parallel execution in model_selection."""
    (ramp_params_grid, step_params_grid, gen_ramp, gen_step, ramp_post, step_post,
     N_TRIALS_RAMP, ramp_gamma_shape, step_gamma_shape, N_TRIALS_STEP, model, scaler) = args

    ramp_params = w3_utils.sample_from_grid(gen_ramp, ramp_params_grid)
    ramp_data, _, _ = RampModelHMM(beta=ramp_params['beta'],
                                   sigma=ramp_params['sigma'],
                                   Rh=ramp_params['Rh'],
                                   x0=ramp_params['x0'] if 'x0' in ramp_params else 0.2,
                                   isi_gamma_shape=ramp_gamma_shape
                                   ).simulate(Ntrials=N_TRIALS_RAMP, T=ramp_params['T'])

    ramp_LLH_ramp = w3_utils.ramp_LLH(ramp_data, ramp_params_grid)
    step_LLH_ramp = w3_utils.step_LLH(ramp_data, step_params_grid)

    ramp_bf_ramp = w3_utils.marginal_likelihood(ramp_LLH_ramp, ramp_post)
    step_bf_ramp = w3_utils.marginal_likelihood(step_LLH_ramp, step_post)

    step_params = w3_utils.sample_from_grid(gen_step, step_params_grid)
    step_data, _, _ = StepModelHMM(m=step_params['m'],
                                   r=step_params['r'],
                                   Rh=step_params['Rh'],
                                   x0=step_params['x0'] if 'x0' in step_params else 0.2,
                                   isi_gamma_shape=step_gamma_shape
                                   ).simulate_exact(Ntrials=N_TRIALS_STEP, T=step_params['T'])

    ramp_LLH_step = w3_utils.ramp_LLH(step_data, ramp_params_grid)
    step_LLH_step = w3_utils.step_LLH(step_data, step_params_grid)

    ramp_bf_step = w3_utils.marginal_likelihood(ramp_LLH_step, ramp_post)
    step_bf_step = w3_utils.marginal_likelihood(step_LLH_step, step_post)

    # ML
    batch_size = 20
    if N_TRIALS_RAMP < batch_size:
        batch_size = N_TRIALS_RAMP
        num_batches = 1
    else:
        num_batches = N_TRIALS_RAMP // batch_size

    for i_dataset in range(num_batches):
        # print(step_spikes[i_dataset].shape)
        X_step = compute_summary_statistics(step_data[i_dataset * batch_size:(i_dataset+1) * batch_size], batch_size)
        X_ramp = compute_summary_statistics(ramp_data[i_dataset * batch_size:(i_dataset+1) * batch_size], batch_size)

        X_step = scaler.transform(X_step)
        X_ramp = scaler.transform(X_ramp)

        X_step = torch.FloatTensor(X_step)
        X_ramp = torch.FloatTensor(X_ramp)

        model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            # ramp is 1, so should be positive = ramp prediction, I think
            step_outputs = model(X_step).numpy()

            epsilon = 1e-3  # Small constant (e.g., machine epsilon)
            step_outputs = np.clip(step_outputs, epsilon, 1 - epsilon)
            LLR_step = np.sum(np.log(step_outputs)) - np.sum(np.log(1 - step_outputs))

            ramp_outputs = model(X_ramp).numpy()

            epsilon = 1e-3  # Small constant (e.g., machine epsilon)
            ramp_outputs = np.clip(ramp_outputs, epsilon, 1 - epsilon)
            LLR_ramp = np.sum(np.log(ramp_outputs)) - np.sum(np.log(1 - ramp_outputs))

            print('STEP', LLR_step, 'RAMP', LLR_ramp)

    return {
        'beta': ramp_params['beta'],
        'sigma': ramp_params['sigma'],
        'ramp_data_ramp_bf': ramp_bf_ramp,
        'ramp_data_step_bf': step_bf_ramp,
        'm': step_params['m'],
        'r': step_params['r'],
        'step_data_ramp_bf': ramp_bf_step,
        'step_data_step_bf': step_bf_step,
        'ML_LLR_step': LLR_step,
        'ML_LLR_ramp': LLR_ramp
    }


def model_selection_with_14(ramp_params_grid, step_params_grid, gen_ramp, gen_step, ramp_post, step_post,
                    N_DATASETS=100, N_TRIALS=100, ramp_gamma_shape=None, step_gamma_shape=None, save_to=None):
    """
    Run simulation and save model comparison results
    Also test against 1.4 ad-hoc ML method

    Parameters:
    -----------
    ramp_params_grid: dict
        Grid of parameters for ramp model
    step_params_grid: dict
        Grid of parameters for step model
    gen_ramp: function
        PDF grid for ramp data
    gen_step: function
        PDF grid for step data
    ramp_post: array-like
        Ramp model posterior
    step_post: array-like
        Step model posterior
    N_DATASETS: int
        Number of datasets to generate
    N_TRIALS: int
        Number of trials per dataset
    ramp_gamma_shape: float, optional
        Shape parameter for ramp model ISI gamma distribution
    step_gamma_shape: float, optional
        Shape parameter for step model ISI gamma distribution
    save_to: str, optional
        Path to save results CSV file
    """

    results = {
        'beta': [], 'sigma': [],
        'ramp_data_ramp_bf': [], 'ramp_data_step_bf': [],
        'm': [], 'r': [],
        'step_data_ramp_bf': [], 'step_data_step_bf': [],
        'ML_LLR_step': [], 'ML_LLR_ramp': []
    }




    save_path = './results/StepRampClassifier.pth'
    checkpoint = torch.load(save_path, weights_only=False)
    model = StepRampClassifier(input_dim=60)  # Use the same architecture as before
    model.load_state_dict(checkpoint['model_state_dict'])

    scaler = checkpoint['scaler_state']
    model.eval()

    worker_args = [(ramp_params_grid, step_params_grid, gen_ramp, gen_step, ramp_post, step_post,
                    N_TRIALS, ramp_gamma_shape, step_gamma_shape, N_TRIALS, model, scaler) for _ in range(N_DATASETS)]

    # pool_results = [_model_selection_worker_with_14(worker_args[0])]

    with multiprocessing.Pool() as pool:
        pool_results = list(tqdm(pool.imap_unordered(_model_selection_worker_with_14, worker_args), total=N_DATASETS))

    for res in pool_results:
        for key in results:
            results[key].append(res[key])

    if save_to:
        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        results_df.to_csv(save_to, index=False)

    return results


var = lambda a, b, frac: ((b-a) * frac) ** 2
mean = lambda a, b: (b+a)/2

if __name__ == "__main__":
    K = 25
    T_MS = 100

    M_GRID = 15

    RH = 20
    X0 = 0.5
    N_TRIALS= 3 # new standard
    # 10 is really bad?

    N_DATASETS = 24 # TODO make this big

    # STD_FRACTION = 0.25

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

    STD_FRACTION = 0.25

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






    # TEST 1

    fn = "./results/0.25GGML_D" + str(N_DATASETS) + "_T" + str(N_TRIALS) + ".csv"

    model_selection_with_14(
        ramp_params_grid, step_params_grid,
        gauss_ramp_posterior, gauss_step_posterior, # generating
        gauss_ramp_posterior, gauss_step_posterior, # inference
        N_DATASETS=N_DATASETS, N_TRIALS=N_TRIALS,
        # format: generating: G/U; inference: G/U; n. datasets, n. trials
        # if G, append std_fraction on front
        save_to=os.path.join(os.getcwd(), fn)
    )

    w3_2.plot_heatmap(fn, r'Heatmap (Gaussian sampling + posterior, $\sigma_{frac} = 0.25$)',
                      save_name=os.path.join(os.getcwd(), f'plots/task_3_2_3_{os.path.basename(fn)[:-4]}_heatmap.png'))
    w3_2.plot_confusion_matrix(fn, r'Confusion Matrix (Gaussian sampling + posterior, $\sigma_{frac} = 0.25$)',
                               save_name=os.path.join(os.getcwd(), f'plots/task_3_2_3_{os.path.basename(fn)[:-4]}_confmat.png'))

    # exit()