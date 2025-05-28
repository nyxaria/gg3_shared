import numpy as np
import matplotlib.pyplot as plt

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

def main():
    K = 50

    beta = 1.0
    sigma = 0.5
    x0 = 0.0

    print("Plotting comparison of continous and HHM with T={T_MS}, N_trials={N_TRIALS}")
    continuous_model = RampModel(beta=beta, sigma=sigma, x0=x0)
    hmm_model = RampModelHMM(K=K, beta=beta, sigma=sigma, x0=x0)

    plot_compared_trajectories(continuous_model, hmm_model, T_MS)
 
    print("Varying Sigma at Beta=0")
    beta = 0.0
    x0 = 0.0
    K = 50

    # calculate approximate critical sigma for HMM
    sigma_crit = np.sqrt(T_MS) / (K - 1)
    print(f"critical sigma for T={T_MS}, K={K}: {sigma_crit:.3f}")

    sigma_values = [sigma_crit * 2, sigma_crit, sigma_crit / 5, sigma_crit / 10]

    plot_sigma_comparison_trajectories(K=K, 
                                       T=T_MS, 
                                       x0=x0, 
                                       beta=beta, 
                                       sigma_values=sigma_values, 
                                       n_trials=N_TRIALS)

if __name__ == "__main__":
    main()
