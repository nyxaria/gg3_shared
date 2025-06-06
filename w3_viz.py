import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import scipy.special
import os

import w3_utils
from models_hmm import RampModelHMM

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

def task_3_1_1_visualize_posterior_2d(true_params, n_trials, param_specs, params_grid, K=50, T=100, Rh=50):
    """
    Visualizes the 2D posterior probability for RampModel with x0 fixed.
    """
    model = RampModelHMM(beta=true_params['beta'], sigma=true_params['sigma'], x0=true_params['x0'], K=K, Rh=Rh)
    data, _, _ = model.simulate(Ntrials=n_trials, T=T)

    # calculate likelihood, prior, posterior
    llh_grid = w3_utils.ramp_LLH(data, params_grid)
    prior_grid = w3_utils.uniform_posterior(params_grid, log=True)
    norm_post_grid = w3_utils.norm_posterior(llh_grid, prior_grid, log=True)

    beta_vals = param_specs['beta']
    sigma_vals = param_specs['sigma']

    plt.figure(figsize=(8, 6))

    plt.imshow(np.exp(norm_post_grid).T, origin='lower', aspect='auto',
               extent=[beta_vals[0], beta_vals[-1], sigma_vals[0], sigma_vals[-1]],
               cmap='viridis')
    plt.colorbar(label='Posterior Probability')
    plt.scatter(true_params['beta'], true_params['sigma'], color='red', marker='x', s=100, label='True Parameters')
    
    plt.xlabel(r'Beta ($\beta$)')
    plt.ylabel(r'Sigma ($\sigma$)')
    plt.title(f'2D Posterior (x0={true_params["x0"]:.2f}, N_trials={n_trials})')
    plt.legend()
    
    save_dir = 'plots'
    filename = f"task_3.1.1_posterior_beta_{true_params['beta']:.2f}_sigma_{true_params['sigma']:.2f}_x0_{true_params['x0']:.2f}_ntrials_{n_trials}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

def task_3_1_2_analyze_estimation_2d(true_params, n_trials_list, params_grid, K=50, T=100, Rh=50):
    """
    Analyzes estimation error and posterior uncertainty for RampModel with x0 fixed.
    """
    
    results = {'beta_error': [], 'sigma_error': [], 
               'beta_std': [], 'sigma_std': []}

    for n_trials in n_trials_list:
        model = RampModelHMM(beta=true_params['beta'], sigma=true_params['sigma'], x0=true_params['x0'], K=K, Rh=Rh)
        data, _, _ = model.simulate(Ntrials=n_trials, T=T)

        llh_grid = w3_utils.ramp_LLH(data, params_grid)
        prior_grid = w3_utils.uniform_posterior(params_grid, log=True)
        norm_post_grid = w3_utils.norm_posterior(llh_grid, prior_grid, log=True)

        posterior_means = w3_utils.expectation(norm_post_grid, params_grid, log=True)
        posterior_sds = w3_utils.posterior_std_dev(norm_post_grid, params_grid, posterior_means, log=True)

        results['beta_error'].append(posterior_means.get('beta', np.nan) - true_params['beta'])
        results['sigma_error'].append(posterior_means.get('sigma', np.nan) - true_params['sigma'])
        results['beta_std'].append(posterior_sds.get('beta', np.nan))
        results['sigma_std'].append(posterior_sds.get('sigma', np.nan))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(n_trials_list, np.abs(results['beta_error']), 'o-', label='|Beta Estimation Error|')
    axs[0].plot(n_trials_list, results['beta_std'], 's--', label='Beta Posterior Std Dev')
    axs[0].set_ylabel('Error / Std Dev for Beta')
    axs[0].legend()
    axs[0].set_title(f'Estimation Error & Uncertainty vs. N_trials (x0={true_params["x0"]:.2f})')

    axs[1].plot(n_trials_list, np.abs(results['sigma_error']), 'o-', label='|Sigma Estimation Error|')
    axs[1].plot(n_trials_list, results['sigma_std'], 's--', label='Sigma Posterior Std Dev')
    axs[1].set_ylabel('Error / Std Dev for Sigma')
    axs[1].set_xlabel('Number of Trials (N_trials)')
    axs[1].legend()
    
    plt.tight_layout()
    
    save_dir = 'plots'
    filename = f"task_3.1.2_errors_vs_ntrials_beta_{true_params['beta']:.2f}_sigma_{true_params['sigma']:.2f}_x0_{true_params['x0']:.2f}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

def task_3_1_3_visualize_posterior_marginal(true_params, n_trials, param_specs, params_grid, K=50, T=100, Rh=50):
    """
    Visualizes 2D marginal posteriors for RampModel with x0 also inferred (3D grid).
    """

    model = RampModelHMM(beta=true_params['beta'], sigma=true_params['sigma'], x0=true_params['x0'], K=K, Rh=Rh)
    data, _, _ = model.simulate(Ntrials=n_trials, T=T)

    llh_grid = w3_utils.ramp_LLH(data, params_grid)
    prior_grid = w3_utils.uniform_posterior(params_grid, log=True)
    norm_post_grid_3d = w3_utils.norm_posterior(llh_grid, prior_grid, log=True)

    varying_p_names = [k for k, v in param_specs.items() if isinstance(v, np.ndarray)]
    
    idx_beta = varying_p_names.index('beta')
    idx_sigma = varying_p_names.index('sigma')
    idx_x0 = varying_p_names.index('x0')

    marginal_beta_sigma = scipy.special.logsumexp(norm_post_grid_3d, axis=idx_x0)
    marginal_beta_x0 = scipy.special.logsumexp(norm_post_grid_3d, axis=idx_sigma)
    marginal_sigma_x0 = scipy.special.logsumexp(norm_post_grid_3d, axis=idx_beta)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    beta_vals = param_specs['beta']
    sigma_vals = param_specs['sigma']
    x0_vals = param_specs['x0']

    # beta vs sigma
    im0 = axs[0].imshow(np.exp(marginal_beta_sigma).T, origin='lower', aspect='auto',
                        extent=[beta_vals[0], beta_vals[-1], sigma_vals[0], sigma_vals[-1]], cmap='viridis')
    axs[0].scatter(true_params['beta'], true_params['sigma'], c='r', marker='x')
    axs[0].set_xlabel(r'Beta ($\beta$)')
    axs[0].set_ylabel(r'Sigma ($\sigma$)')
    axs[0].set_title(r'P($\beta$, $\sigma$)')
    fig.colorbar(im0, ax=axs[0], orientation='vertical')

    # beta vs x0
    im1 = axs[1].imshow(np.exp(marginal_beta_x0).T, origin='lower', aspect='auto',
                        extent=[beta_vals[0], beta_vals[-1], x0_vals[0], x0_vals[-1]], cmap='viridis')
    axs[1].scatter(true_params['beta'], true_params['x0'], c='r', marker='x')
    axs[1].set_xlabel(r'Beta ($\beta$)')
    axs[1].set_ylabel('x0')
    axs[1].set_title(r'P($\beta$, x0)')
    fig.colorbar(im1, ax=axs[1], orientation='vertical')

    # sigma vs x0
    im2 = axs[2].imshow(np.exp(marginal_sigma_x0).T, origin='lower', aspect='auto',
                        extent=[sigma_vals[0], sigma_vals[-1], x0_vals[0], x0_vals[-1]], cmap='viridis')
    axs[2].scatter(true_params['sigma'], true_params['x0'], c='r', marker='x')
    axs[2].set_xlabel(r'Sigma ($\sigma$)')
    axs[2].set_ylabel('x0')
    axs[2].set_title(r'P($\sigma$, x0)')
    fig.colorbar(im2, ax=axs[2], orientation='vertical')

    fig.suptitle(f'2D Marginal Posteriors (N_trials={n_trials})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_dir = 'plots'
    filename = f"task_3.1.3_marginals_beta_{true_params['beta']:.2f}_sigma_{true_params['sigma']:.2f}_x0_{true_params['x0']:.2f}_ntrials_{n_trials}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

def task_3_1_3_analyze_estimation_3d(true_params, n_trials_list, params_grid, K=50, T=100, Rh=50):
    """
    Analyzes estimation error and posterior uncertainty for RampModel with x0 inferred.
    """
    
    results = {'beta_error': [], 'sigma_error': [], 'x0_error': [],
               'beta_std': [], 'sigma_std': [], 'x0_std': []}

    for n_trials in n_trials_list:
        model = RampModelHMM(beta=true_params['beta'], sigma=true_params['sigma'], x0=true_params['x0'], K=K, Rh=Rh)
        data, _, _ = model.simulate(Ntrials=n_trials, T=T)

        llh_grid = w3_utils.ramp_LLH(data, params_grid)
        prior_grid = w3_utils.uniform_posterior(params_grid, log=True)
        norm_post_grid = w3_utils.norm_posterior(llh_grid, prior_grid, log=True)

        posterior_means = w3_utils.expectation(norm_post_grid, params_grid, log=True)
        posterior_sds = w3_utils.posterior_std_dev(norm_post_grid, params_grid, posterior_means, log=True)
        
        results['beta_error'].append(posterior_means.get('beta',np.nan) - true_params['beta'])
        results['sigma_error'].append(posterior_means.get('sigma',np.nan) - true_params['sigma'])
        results['x0_error'].append(posterior_means.get('x0',np.nan) - true_params['x0'])
        results['beta_std'].append(posterior_sds.get('beta',np.nan))
        results['sigma_std'].append(posterior_sds.get('sigma',np.nan))
        results['x0_std'].append(posterior_sds.get('x0',np.nan))

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    axs[0].plot(n_trials_list, np.abs(results['beta_error']), 'o-', label='|Beta Estimation Error|')
    axs[0].plot(n_trials_list, results['beta_std'], 's--', label='Beta Posterior Std Dev')
    axs[0].set_ylabel('Beta Error/StdDev')
    axs[0].legend()
    axs[0].set_title('Estimation Error & Uncertainty vs. N_trials (3 Parameters Inferred)')

    axs[1].plot(n_trials_list, np.abs(results['sigma_error']), 'o-', label='|Sigma Estimation Error|')
    axs[1].plot(n_trials_list, results['sigma_std'], 's--', label='Sigma Posterior Std Dev')
    axs[1].set_ylabel('Sigma Error/StdDev')
    axs[1].legend()

    axs[2].plot(n_trials_list, np.abs(results['x0_error']), 'o-', label='|x0 Estimation Error|')
    axs[2].plot(n_trials_list, results['x0_std'], 's--', label='x0 Posterior Std Dev')
    axs[2].set_ylabel('x0 Error/StdDev')
    axs[2].set_xlabel('Number of Trials (N_trials)')
    axs[2].legend()
    
    plt.tight_layout()
    
    save_dir = 'plots'
    filename = f"task_3.1.3_errors_vs_ntrials_3d_beta_{true_params['beta']:.2f}_sigma_{true_params['sigma']:.2f}_x0_{true_params['x0']:.2f}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    K = 25
    T_MS = 100
    RH = 50
    M_GRID = 15 

    true_params = {'beta': 1.0, 'sigma': 0.2, 'x0': 0.2}
    
    param_specs = OD([
        ('beta', np.linspace(0, 4, M_GRID)), 
        ('sigma', np.linspace(0.04, 4, M_GRID)),
        ('x0', true_params['x0']),
        ('K', K),
        ('T', T_MS),
        ('Rh', RH)
    ])
    params_grid = w3_utils.make_params_grid(param_specs)
    
    """
    Repeat for different true parameter values (within the ranges of the prior distribution) and for different number of trials (over a range from 1 to 400). 
    Document your observations of systematic dependence of the posterior on the number of trials. Similarly, note any systematic changes in behaviour
    for different values of the true parameters. 
    """

    # todo loop over n_trials and different true_params to get the results above
    # 3.1.1
    task_3_1_1_visualize_posterior_2d(true_params, n_trials=100,
                                      param_specs=param_specs,
                                      params_grid=params_grid,
                                      K=K, T=T_MS, Rh=RH)
    # exit()
    # 3.1.2
    n_trials_list_312 = [1, 5, 10, 20, 50, 100, 200, 400]
    task_3_1_2_analyze_estimation_2d(true_params, n_trials_list_312,
                                     params_grid=params_grid,
                                     K=K, T=T_MS, Rh=RH)

    # exit()
    # 3.1.3
    true_params = {'beta': 1.0, 'sigma': 0.2, 'x0': 0.1}
    M_GRID = 10 # reduce for speed

    param_specs_313 = OD([
        ('beta', np.linspace(0, 4, M_GRID)),
        ('sigma', np.linspace(0.04, 4, M_GRID)),
        ('x0', np.linspace(0, 0.5, M_GRID)),
        ('K', K),
        ('T', T_MS),
        ('Rh', RH)
    ])
    params_grid_313 = w3_utils.make_params_grid(param_specs_313)
    
    task_3_1_3_visualize_posterior_marginal(true_params, n_trials=100,
                                            param_specs=param_specs_313,
                                            params_grid=params_grid_313,
                                            K=K, T=T_MS, Rh=RH)
    
    task_3_1_3_analyze_estimation_3d(true_params, n_trials_list=[50, 100, 200],
                                     params_grid=params_grid_313,
                                     K=K, T=T_MS, Rh=RH)
