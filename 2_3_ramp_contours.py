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
    num_trials = 20
    num_samples = 15 # on the grid
    K = 50
    save_filename = "./results/CE_15x0x15rh.npy" # TODO
    recompute = False

    # TODO
    # betas = np.linspace(0, 4, num_samples)
    # sigmas = np.linspace(0, 4, num_samples)
    x0s = np.linspace(0, 1, num_samples)
    Rhs = np.linspace(100, 2000, num_samples)

    if os.path.exists(save_filename) and not recompute:
        print("loading " + save_filename)
        CE_mat = np.load(save_filename)
    else:
        print("computing")
        CE_mat = np.zeros((num_samples, num_samples))

        for i1, b in enumerate(x0s): # TODO
            for i2, s in enumerate(Rhs):# TODO
                sum_CE = 0
                for _ in range(num_trials):
                    ex, expected_s, states = task_2_3.ramp_HMM_inference({ # TODO
                        "x0": b,
                        "Rh": s,
                        #"beta": b,
                        #"sigma": s,
                        "K": K,
                    })
                    # true_s = (xs * (K - 1)).flatten().astype(int)
                    sum_CE += task_2_3.cross_entropy(ex, states, time_average=True)
                CE_mat[i1, i2] = sum_CE / num_trials
            print(f'Progress: {i1 + 1}/{num_samples} (beta)')

        np.save(save_filename, CE_mat)
        print(f"saved to {save_filename}")

    '''# matshow
    plt.matshow(CE_mat)
    plt.show()'''

    # tricontour
    beta_grid, sigma_grid = np.meshgrid(x0s, Rhs, indexing='ij') # TODO
    beta_flat = beta_grid.flatten()
    sigma_flat = sigma_grid.flatten()
    CE_flat = CE_mat.flatten()


    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(beta_flat, sigma_flat, CE_flat, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Cross-Entropy (CE)')
    plt.xlabel('x0') # TODO
    plt.ylabel('Rh') # TODO
    plt.title(r'Time-averaged Cross Entropy for varying $x_0, Rh$') # TODO
    plt.show()