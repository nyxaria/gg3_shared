import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

from models import RampModel
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

N_TRIALS = 1000
T_MS = 1000

def plot_compared_trajectories(continuous_model, hmm_model, T_MS):
    _, x_continuous_all, _ = continuous_model.simulate(Ntrials=N_TRIALS, T=T_MS)
    actual_spikes_hmm, x_hmm_all, _ = hmm_model.simulate(Ntrials=N_TRIALS, T=T_MS)
    
    # take mean across all trials
    mean_x_continuous = np.mean(x_continuous_all, axis=0)
    mean_x_hmm = np.mean(x_hmm_all, axis=0)
    
    dt_sim = 1.0 / T_MS
    time_axis = np.arange(T_MS) * dt_sim
    
    plt.figure(figsize=(12, 6))
    plt.title(f"$x_t$ Trajectory (K={hmm_model.K}, T={T_MS}, $N_{{\mathrm{{trials}}}}$={N_TRIALS})")

    plt.plot(time_axis, mean_x_continuous, label="Continuous", linestyle='-')
    plt.plot(time_axis, mean_x_hmm, label="HMM", linestyle='--')
    
    plt.ylabel("$x_t$")
    plt.xlabel("Time (s)")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/task_2_1_trajectories_B{hmm_model.beta}_S{hmm_model.sigma}_K{hmm_model.K}_T{T_MS}_N{N_TRIALS}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_sigma_comparison_trajectories(K, T, x0, beta, sigma_values, n_trials):
    plt.figure(figsize=(12, 6))
    plt.title(f"HMM $x_t$ Trajectory (beta={beta}, x0={x0}, K={K}, T={T}, $N_{{\mathrm{{trials}}}}$={n_trials})")


    dt_sim = 1.0 / T
    time_axis_sim = np.arange(T) * dt_sim

    for sigma_val_current in sigma_values:
        hmm_model_sigma = RampModelHMM(K=K, beta=beta, sigma=sigma_val_current, x0=x0)
        
        _, x_traj_sigma, _ = hmm_model_sigma.simulate(Ntrials=n_trials, T=T)
        
        mean_trajectory_sigma = np.mean(x_traj_sigma, axis=0)
        plt.plot(time_axis_sim, mean_trajectory_sigma, label=f"sigma = {sigma_val_current:.3f}")

        stuck_count = 0
        for i in range(n_trials):
            if np.allclose(x_traj_sigma[i,:], x_traj_sigma[i,0]):
                stuck_count +=1
        print(f"For sigma={sigma_val_current:.3f}, {stuck_count}/{n_trials} trajectories appeared stuck at initial value.")
    
    plt.xlabel("Time (s)")
    plt.ylabel("$x_t$")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/task_2_1_sigma_comparison_K{K}_T{T}_N{n_trials}.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_and_plot_accuracy_heatmaps(beta_vals, sigma_vals, x0, K, T, n_trials):
    """
    Calculates the error between continuous and HMM models across a grid
    of beta and sigma values, and plots it as a heatmap.
    """
    cache_dir = os.path.join("plots", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    beta_len = len(beta_vals)
    sigma_len = len(sigma_vals)
    cache_filename = os.path.join(cache_dir, f"task_2_1_heatmap_errors_signed_x0{x0:.2f}_K{K}_T{T}_N{n_trials}_B{beta_len}_S{sigma_len}.pickle")

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            errors = pickle.load(f)
        print(f"Loaded errors from cache: {cache_filename}")
    else:
        rmse_errors = np.zeros((len(sigma_vals), len(beta_vals)))
        signed_errors = np.zeros((len(sigma_vals), len(beta_vals)))
        
        for i, sigma in enumerate(tqdm(sigma_vals)):
            for j, beta in enumerate(beta_vals):
                continuous_model = RampModel(beta=beta, sigma=sigma, x0=x0)
                hmm_model = RampModelHMM(K=K, beta=beta, sigma=sigma, x0=x0)
                
                _, x_continuous_all, _ = continuous_model.simulate(Ntrials=n_trials, T=T)
                _, x_hmm_all, _ = hmm_model.simulate(Ntrials=n_trials, T=T)
                
                mean_x_continuous = np.mean(x_continuous_all, axis=0)
                mean_x_hmm = np.mean(x_hmm_all, axis=0)
                
                # calculate RMSE
                rmse = np.sqrt(np.mean((mean_x_continuous - mean_x_hmm)**2))
                rmse_errors[i, j] = rmse

                # calculate signed mean error
                signed_mean_error = np.mean(mean_x_hmm - mean_x_continuous)
                signed_errors[i, j] = signed_mean_error

        errors = {'rmse': rmse_errors, 'signed': signed_errors}
        with open(cache_filename, 'wb') as f:
            pickle.dump(errors, f)
        print(f"Saved errors to cache: {cache_filename}")

    delta_beta = beta_vals[1] - beta_vals[0]
    delta_sigma_first = sigma_vals[1] - sigma_vals[0]
    delta_sigma_last = sigma_vals[-1] - sigma_vals[-2]
    extent = [
        beta_vals[0] - delta_beta / 2, beta_vals[-1] + delta_beta / 2,
        sigma_vals[0] - delta_sigma_first / 2, sigma_vals[-1] + delta_sigma_last / 2
    ]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(errors['rmse'], aspect='auto', origin='lower', extent=extent, cmap='viridis')
    plt.colorbar(im, label="RMSE of mean $x_t$")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\sigma$")
    # plt.yscale('log')
    plt.title(f"HMM vs Continuous Model Accuracy (RMSE)\n(x0={x0}, K={K}, T={T}, N_trials={n_trials})")
    filename = f'plots/task_2_1_heatmap_rmse_x0{x0}_K{K}_T{T}_N{n_trials}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 8))
    signed_errors_matrix = errors['signed']
    max_abs_error = np.max(np.abs(signed_errors_matrix))
    im = plt.imshow(signed_errors_matrix, aspect='auto', origin='lower', extent=extent, cmap='coolwarm', vmin=-max_abs_error, vmax=max_abs_error)
    plt.colorbar(im, label="Signed Mean Error (HMM - Continuous)")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\sigma$")
    # plt.yscale('log')
    plt.title(f"HMM vs Continuous Model Accuracy (Signed Error)\n(x0={x0}, K={K}, T={T}, N_trials={n_trials})")
    filename = f'plots/task_2_1_heatmap_signed_error_x0{x0}_K{K}_T{T}_N{n_trials}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    K = 50
    # beta = 1.0
    # sigma = 0.5
    # x0 = 0.0
    # print("Plotting comparison of continous and HHM with T={T_MS}, N_trials={N_TRIALS}")
    # continuous_model = RampModel(beta=beta, sigma=sigma, x0=x0)
    # hmm_model = RampModelHMM(K=K, beta=beta, sigma=sigma, x0=x0)
    # plot_compared_trajectories(continuous_model, hmm_model, T_MS)
 
    # print("Varying Sigma at Beta=0")
    # beta = 0.0
    # x0 = 0.0
    # # calculate approximate critical sigma for HMM
    # sigma_crit = np.sqrt(T_MS) / (K - 1)
    # print(f"critical sigma for T={T_MS}, K={K}: {sigma_crit:.3f}")
    # sigma_values = [sigma_crit * 2, sigma_crit, sigma_crit / 5, sigma_crit / 10]
    # plot_sigma_comparison_trajectories(K=K, 
    #                                    T=T_MS, 
    #                                    x0=x0, 
    #                                    beta=beta, 
    #                                    sigma_values=sigma_values, 
    #                                    n_trials=N_TRIALS)

    print("\n--- HMM Accuracy Analysis ---")
    
    n_grid_points = 15
    beta_vals = np.linspace(0, 4, n_grid_points)
    sigma_vals = np.logspace(np.log10(0.04), np.log10(4), n_grid_points)
    sigma_vals = np.linspace(0.04, 4, n_grid_points)

    x0_heatmap = 0.2
    n_trials_heatmap = 100
    
    analyze_and_plot_accuracy_heatmaps(beta_vals=beta_vals,
                                       sigma_vals=sigma_vals,
                                       x0=x0_heatmap,
                                       K=K,
                                       T=T_MS,
                                       n_trials=n_trials_heatmap)

if __name__ == "__main__":
    main()
