import numpy as np
import models
import scipy
import inference
from models_hmm import RampModelHMM
import os
import matplotlib.pyplot as plt

import task_2_3

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

np.set_printoptions(legacy='1.25')  # don't show np.float; helps with debug

if __name__ == "__main__":
    num_trials = 30
    num_samples = 10  # on the grid
    K = 50
    smooth_save_filename = "./results/ramp_f_sCE_x0s8.npy"
    filter_save_filename = "./results/ramp_f_fCE_x0s8.npy"

    recompute = False

    T = 100
    rs = np.rint(np.linspace(0, 20, num_samples + 1))[1:]
    betas = np.linspace(0, 4, num_samples)
    sigmas = np.linspace(0, 2, num_samples)
    x0s = np.linspace(0, 1, num_samples)
    Rhs = np.linspace(100, 2000, num_samples)

    ivar1 = x0s
    ivar2 = sigmas

    if os.path.exists(smooth_save_filename) and os.path.exists(filter_save_filename) and not recompute:
        print("loading " + smooth_save_filename + ', ' + filter_save_filename)
        CE_mat = np.load(smooth_save_filename)
        fCE_mat = np.load(filter_save_filename)
    else:

        print("computing")
        CE_mat = np.zeros((num_samples, num_samples))
        fCE_mat = np.zeros((num_samples, num_samples))

        for i1, iv1 in enumerate(ivar1):
            for i2, iv2 in enumerate(ivar2):
                sum_CE = 0
                sum_fCE = 0
                for _ in range(num_trials):
                    ex, fex, _, _, states = task_2_3.ramp_HMM_inference({
                        "x0": iv1,
                        "sigma": iv2,
                        # "beta": b,
                        # "sigma": s,
                        "K": K,
                    }, test_filtering=True)
                    # true_s = (xs * (K - 1)).flatten().astype(int)
                    # bex = task_2_3.compress_states(ex)
                    # bstates = (states == iv2).astype(int)  # TODO if r, set == r

                    # fbex = task_2_3.compress_states(fex)
                    # fbstates = (states == iv2).astype(int)  # TODO if r, set == r

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

    ivar1_grid, ivar2_grid = np.meshgrid(ivar1, ivar2, indexing='ij')  # betas, sigmas, indexing='ij')
    ivar1_flat = ivar1_grid.flatten()
    ivar2_flat = ivar2_grid.flatten()
    CE_diff_flat = CE_diff.flatten()

    os.makedirs('plots', exist_ok=True)

    plt.matshow(CE_diff)
    plt.savefig('plots/task_2_3_filter_contour_matshow.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(ivar1_flat, ivar2_flat, CE_diff_flat, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Difference in Cross-Entropy')
    plt.xlabel('x0')
    plt.ylabel('sigma')
    plt.title(r'Ramp: $CE_{smoothed} - CE_{filtered}$ for varying $x_0, sigma$')
    plt.savefig('plots/task_2_3_filter_contour_contour.png')
    plt.show()
