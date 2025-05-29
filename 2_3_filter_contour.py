import numpy as np
import models
import scipy
import inference
from models_hmm import RampModelHMM
import os
import matplotlib.pyplot as plt

import task_2_3

np.set_printoptions(legacy='1.25') # don't show np.float; helps with debug

if __name__ == "__main__":
    num_trials = 5
    num_samples = 5 # on the grid
    K = 50
    smooth_save_filename = "./results/f_sCE_5x0x5rh.npy"
    filter_save_filename = "./results/f_fCE_5x0x5rh.npy"

    recompute = False

    # betas = np.linspace(0, 4, num_samples)
    # sigmas = np.linspace(0, 4, num_samples)
    x0s = np.linspace(0, 1, num_samples)
    Rhs = np.linspace(100, 2000, num_samples)

    if os.path.exists(smooth_save_filename) and os.path.exists(filter_save_filename) and not recompute:
        print("loading " + smooth_save_filename + ', ' + filter_save_filename)
        CE_mat = np.load(smooth_save_filename)
        fCE_mat = np.load(filter_save_filename)
    else:

        print("computing")
        CE_mat = np.zeros((num_samples, num_samples))
        fCE_mat = np.zeros((num_samples, num_samples))

        for i1, b in enumerate(x0s):
            for i2, s in enumerate(Rhs):
                sum_CE = 0
                sum_fCE = 0
                for _ in range(num_trials):
                    ex, fex, expected_s, fexpected_s, states = task_2_3.ramp_HMM_inference({
                        "x0": b,
                        "Rh": s,
                        # "beta": b,
                        # "sigma": s,
                        "K": K,
                    }, test_filtering=True)
                    # true_s = (xs * (K - 1)).flatten().astype(int)
                    sum_CE += task_2_3.cross_entropy(ex, states, time_average=True)
                    sum_fCE += task_2_3.cross_entropy(fex, states, time_average=True)
                CE_mat[i1, i2] = sum_CE / num_trials
                fCE_mat[i1, i2] = sum_fCE / num_trials

            print(f'Progress: {i1 + 1}/{num_samples} (beta)')

        np.save(smooth_save_filename, CE_mat)
        np.save(filter_save_filename, fCE_mat)
        print(f"saved to {smooth_save_filename}")

    # matshow


    # tricontour

    CE_diff = fCE_mat - CE_mat

    beta_grid, sigma_grid = np.meshgrid(x0s, Rhs, indexing='ij') #betas, sigmas, indexing='ij')
    beta_flat = beta_grid.flatten()
    sigma_flat = sigma_grid.flatten()
    CE_diff_flat = CE_diff.flatten()

    plt.matshow(CE_diff)
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(beta_flat, sigma_flat, CE_diff_flat, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Difference in Cross-Entropy')
    plt.xlabel('x0')
    plt.ylabel('Rh')
    plt.title(r'$CE_{smoothed} - CE_{filtered}$ for varying $\beta, \sigma$')
    plt.show()