import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import scipy.special
import pandas as pd
import os
import multiprocessing
from models_hmm import RampModelHMM, StepModelHMM
from scipy.interpolate import griddata
import seaborn as sns
import w3_utils
from models_hmm import RampModelHMM
from tqdm import tqdm


def _model_selection_worker(args):
    """Helper function for parallel execution in model_selection."""
    (ramp_params_grid, step_params_grid, gen_ramp, gen_step, ramp_post, step_post,
     N_TRIALS_RAMP, ramp_gamma_shape, step_gamma_shape, N_TRIALS_STEP) = args

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

    return {
        'beta': ramp_params['beta'],
        'sigma': ramp_params['sigma'],
        'ramp_data_ramp_bf': ramp_bf_ramp,
        'ramp_data_step_bf': step_bf_ramp,
        'm': step_params['m'],
        'r': step_params['r'],
        'step_data_ramp_bf': ramp_bf_step,
        'step_data_step_bf': step_bf_step
    }


def model_selection(ramp_params_grid, step_params_grid, gen_ramp, gen_step, ramp_post, step_post,
                   N_DATASETS=100, N_TRIALS=100, ramp_gamma_shape=None, step_gamma_shape=None, save_to=None):
    """
    Run simulation and save model comparison results
    
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
        'step_data_ramp_bf': [], 'step_data_step_bf': []
    }

    N_TRIALS_STEP = 50

    worker_args = [(ramp_params_grid, step_params_grid, gen_ramp, gen_step, ramp_post, step_post,
                    N_TRIALS, ramp_gamma_shape, step_gamma_shape, N_TRIALS) for _ in range(N_DATASETS)]

    with multiprocessing.Pool() as pool:
        pool_results = list(tqdm(pool.imap_unordered(_model_selection_worker, worker_args), total=N_DATASETS))
    
    for res in pool_results:
        for key in results:
            results[key].append(res[key])

    if save_to:
        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        results_df.to_csv(save_to, index=False)

    return results


def plot_heatmap(results_df, title='Untitled Heatmap'):
    """
    Plot heatmaps comparing model performance on ramp and step data
    
    Parameters:
    -----------
    results_df: pandas.DataFrame or str
        DataFrame containing results or path to CSV file with results
    """

    if isinstance(results_df, str):
        results_df = pd.read_csv(results_df)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Ramp data heatmap
    grid_x1, grid_y1 = np.mgrid[0:4:100j, 0.04:4:100j]
    grid_z1 = griddata((results_df['beta'], results_df['sigma']),
                       np.array(results_df['ramp_data_ramp_bf']) - np.array(results_df['ramp_data_step_bf']),
                       (grid_x1, grid_y1),
                       method='cubic')

    im1 = ax1.pcolormesh(grid_x1, grid_y1, grid_z1, cmap='coolwarm', shading='auto')
    ax1.scatter(results_df['beta'], results_df['sigma'], c='black', s=20, alpha=0.5)
    ax1.set_xlabel('beta')
    ax1.set_ylabel('sigma') 
    ax1.set_title('Ramp Data')
    ax1.set_yscale('log')
    plt.colorbar(im1, ax=ax1)

    # Step data heatmap
    grid_x2, grid_y2 = np.mgrid[25:75:100j, 1:6:100j]
    grid_z2 = griddata((results_df['m'], results_df['r']),
                       np.array(results_df['step_data_step_bf']) - np.array(results_df['step_data_ramp_bf']),
                       (grid_x2, grid_y2),
                       method='cubic')

    im2 = ax2.pcolormesh(grid_x2, grid_y2, grid_z2, cmap='coolwarm', shading='auto')
    ax2.scatter(results_df['m'], results_df['r'], c='black', s=20, alpha=0.5)
    ax2.set_xlabel('m')
    ax2.set_ylabel('r')
    ax2.set_title('Step Data')

    plt.title(title)
    plt.colorbar(im2, ax=ax2)

    plt.show()


def plot_confusion_matrix(csv_path, plot_title, save_name='confmat'):
    """
    Plot confusion matrix from results CSV and return accuracy.

    Args:
        csv_path (str): Path to the CSV file containing results
        plot_title (str): Title for the confusion matrix plot

    Returns:
        float: Classification accuracy
    """
    results_df = pd.read_csv(csv_path)

    # Calculate predictions (1 if ramp, 0 if step)
    ramp_predictions = (results_df['ramp_data_ramp_bf'] - results_df['ramp_data_step_bf'] > 0).astype(int)
    step_predictions = (results_df['step_data_ramp_bf'] - results_df['step_data_step_bf'] > 0).astype(int)

    ramp_true = np.ones(len(ramp_predictions))
    step_true = np.zeros(len(step_predictions))

    y_true = np.concatenate([ramp_true, step_true])
    y_pred = np.concatenate([ramp_predictions, step_predictions])

    conf_matrix = np.zeros((2, 2))
    conf_matrix[0, 0] = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    conf_matrix[0, 1] = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    conf_matrix[1, 0] = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    conf_matrix[1, 1] = np.sum((y_true == 1) & (y_pred == 1))  # True Positives

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Step', 'Ramp'],
                yticklabels=['Step', 'Ramp'])
    plt.title(plot_title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    plt.savefig(save_name)

    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
    print(f'Overall HMM Accuracy: {accuracy:.2%}')

    if 'ML_LLR_step' in results_df and 'ML_LLR_ramp' in results_df:
        # additionally plot conf matrix for ML (1.4)

        ramp_true = np.ones(len(results_df['ML_LLR_ramp']))
        step_true = np.zeros(len(results_df['ML_LLR_step']))

        ramp_predictions = (results_df['ML_LLR_ramp'] > 0).astype(int)
        step_predictions = (results_df['ML_LLR_step'] > 0).astype(int)

        y_true = np.concatenate([ramp_true, step_true])
        y_pred = np.concatenate([ramp_predictions, step_predictions])

        conf_matrix = np.zeros((2, 2))
        conf_matrix[0, 0] = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
        conf_matrix[0, 1] = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
        conf_matrix[1, 0] = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
        conf_matrix[1, 1] = np.sum((y_true == 1) & (y_pred == 1))  # True Positives

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=['Step', 'Ramp'],
                    yticklabels=['Step', 'Ramp'])
        plt.title(plot_title + ' - ML (1.4)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

        plt.savefig(save_name + ' - ML (1.4)')



    return


def _test_worker(args):
    (uniform_ramp_posterior, ramp_params_grid, uniform_step_posterior, 
     step_params_grid, RH, N_TRIALS, T_MS) = args

    dataset_params = w3_utils.sample_from_grid(uniform_ramp_posterior, ramp_params_grid)
    data_ramp, _, _ = RampModelHMM(beta=dataset_params['beta'],
                            sigma=dataset_params['sigma'], Rh=RH).simulate(Ntrials=N_TRIALS,
                                                                        T=T_MS)

    dataset_params = w3_utils.sample_from_grid(uniform_step_posterior, step_params_grid)

    data_step, _, _ = StepModelHMM(m=dataset_params['m'],
                              r=dataset_params['r'], Rh=RH).simulate_exact(Ntrials=N_TRIALS,
                                                                             T=T_MS)

    ramp_LLH_pgrid = w3_utils.ramp_LLH(data_ramp, ramp_params_grid)
    ramp_bayes = w3_utils.marginal_likelihood(ramp_LLH_pgrid, uniform_ramp_posterior)

    step_LLH_pgrid = w3_utils.step_LLH(data_step, step_params_grid)
    step_bayes = w3_utils.marginal_likelihood(step_LLH_pgrid, uniform_step_posterior)

    return ramp_bayes, step_bayes, dataset_params['m'], dataset_params['r']


if __name__ == "__main__":
    K = 25
    T_MS = 100
    RH = 50
    M_GRID = 10
    X0 = 0.2
    N_DATASETS = 20
    N_TRIALS = 50


    ramp_param_specs = OD([
        ('beta', np.linspace(0, 4, M_GRID)),
        ('sigma', np.exp(np.linspace(np.log(0.04), np.log(4), M_GRID))),
        ('x0', X0),
        ('K', K),
        ('T', T_MS),
        ('Rh', RH)
    ])

    step_param_specs = OD([('m', [T_MS * 0.25, T_MS * 0.75, M_GRID]),
                ('r', [1, 6, 6]),
                ('x0', X0),
                ('K', K),
                ('T', T_MS),
                ('Rh', RH)])

    ramp_params_grid = w3_utils.make_params_grid(ramp_param_specs)
    step_params_grid = w3_utils.make_params_grid(step_param_specs)

    uniform_ramp_posterior = w3_utils.uniform_prior(ramp_params_grid)
    uniform_step_posterior = w3_utils.uniform_prior(step_params_grid)



    iv1 = []
    iv2 = []
    ramp_bayes_factors = []
    step_bayes_factors = []


    worker_args = [(uniform_ramp_posterior, ramp_params_grid, uniform_step_posterior,
                    step_params_grid, RH, N_TRIALS, T_MS) for _ in range(N_DATASETS)]

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap_unordered(_test_worker, worker_args), total=N_DATASETS))

    for ramp_bayes, step_bayes, m_val, r_val in results:
        ramp_bayes_factors.append(ramp_bayes)
        step_bayes_factors.append(step_bayes)
        iv1.append(m_val)
        iv2.append(r_val)


    diff_bayes = np.array(step_bayes_factors) - np.array(ramp_bayes_factors)

    print(diff_bayes)
    # Create a regular grid to interpolate the data
    grid_x, grid_y = np.mgrid[T_MS*0.25:T_MS*0.75:100j, 1:6:100j]
    # grid_x, grid_y = np.mgrid[0:4:100j, 0:4:100j]

    # Interpolate the data onto the regular grid
    grid_z = griddata((iv1, iv2), diff_bayes, (grid_x, grid_y), method='cubic')

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(grid_x, grid_y, grid_z, cmap='coolwarm', shading='auto')
    plt.colorbar(label='Log Likelihood Difference (ramp - step)')

    # Add the original points
    plt.scatter(iv1, iv2, c='black', s=20, alpha=0.5)

    plt.xlabel('m')
    plt.ylabel('r')
    plt.title('Heatmap of Log Likelihood Differences')

    # Use log scale for sigma axis since it was generated using exp(linspace)
    plt.yscale('log')

    plt.show()


'''gauss_post = w3_utils.gaussian_posterior(params_grid, mu={
        'beta': 1.0, 'sigma': 1.0
    }, cov={
        ('beta', 'beta'): 2,
        ('sigma', 'sigma'): 3
    }, log=True)'''