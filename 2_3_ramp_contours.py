import numpy as np
import models
import scipy
import inference
from models_hmm import RampModelHMM
import os
import matplotlib.pyplot as plt

import task_2_3

np.set_printoptions(legacy='1.25') # don't show np.float; helps with debug

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})


if __name__ == "__main__":
    num_trials = 600
    num_samples = 15 # on the grid
    K = 50
    save_filename = "./results/CE_step_15x0x15Rh.npy" # TODO
    recompute = True

    # TODO
    T = 100
    ms = np.linspace(T * 0.1, T * 0.75, num_samples)
    rs = np.rint(np.linspace(0, 20, num_samples+1))[1:]
    # betas = np.linspace(0, 4, num_samples)
    # sigmas = np.linspace(0, 4, num_samples)
    x0s = np.linspace(0.2, 1, num_samples)
    Rhs = np.linspace(0, 500, num_samples)

    # TODO changeme
    ivar1 = x0s # x-axis
    ivar2 = Rhs

    # if os.path.exists(save_filename) and not recompute:
    #     print("loading " + save_filename)
    #     CE_mat = np.load(save_filename)
    # else:
    #     print("computing")
    #     CE_mat = np.zeros((num_samples, num_samples))

    #     for i1, iv1 in enumerate(ivar1):
    #         for i2, iv2 in enumerate(ivar2):
    #             sum_CE = 0
    #             for _ in range(num_trials):
    #                 ex, expected_s, states = task_2_3.step_HMM_inference({ # TODO
    #                     #"m": iv1,
    #                     "r": 10,
    #                     'x0': iv1,
    #                     'Rh': iv2,
    #                     #"beta": b,
    #                     #"sigma": s,
    #                     "K": K,
    #                 })
    #                 bex = task_2_3.compress_states(ex)
    #                 bstates = (states == 10).astype(int) # TODO if r, set == r

    #                 # we probably want to plot the TOTAL cross-entropy in this case

    #                 sum_CE += task_2_3.cross_entropy(bex, bstates, time_average=True)
    #             CE_mat[i1, i2] = sum_CE / num_trials
    #         print(f'Progress: {i1 + 1}/{num_samples} (beta)')

    #     np.save(save_filename, CE_mat)
    #     print(f"saved to {save_filename}")

    # '''# matshow
    # plt.matshow(CE_mat)
    # plt.show()'''

    # # tricontour
    # iv1_grid, iv2_grid = np.meshgrid(ivar1, ivar2, indexing='ij')
    # iv1_flat = iv1_grid.flatten()
    # iv2_flat = iv2_grid.flatten()
    # CE_flat = CE_mat.flatten()

    # os.makedirs('plots', exist_ok=True)


    # plt.figure(figsize=(8, 6))
    # contour = plt.tricontourf(iv1_flat, iv2_flat, CE_flat, levels=20, cmap='viridis')
    # plt.colorbar(contour, label='Cross-Entropy (CE)')
    # plt.xlabel('x0') # TODO
    # plt.ylabel('Rh') # TODO
    # plt.title(r'Step model - BCE for varying $x_0, Rh$') # TODO
    # plt.savefig('plots/task_2_3_ramp_contours_contour.png')
    # plt.show()

    # plt.matshow(CE_mat)
    # plt.savefig('plots/task_2_3_ramp_contours_matshow.png')
    # plt.show()


    # Step version (Rh vs x0)

    num_trials = 100
    num_samples = 15
    T = 250
    recompute = False

    x0s = np.linspace(0.0, 1, num_samples)
    Rhs = np.linspace(10, 2000, num_samples)

    # fixed params
    r = 10
    m = 40
    Rh_lim = (min(Rhs), max(Rhs))
    x0_lim = (min(x0s), max(x0s))
    save_filename = f"./results/step_CE_heatmap_Rh{Rh_lim[0]}-{Rh_lim[1]}_x0{x0_lim[0]}-{x0_lim[1]}_r{r}_m{m}.npy"

    if os.path.exists(save_filename) and not recompute:
        CE_mat = np.load(save_filename)
    else:
        CE_mat = np.zeros((num_samples, num_samples))

        for i, rh in enumerate(Rhs):
            for j, x0 in enumerate(x0s):
                sum_mean_CE = 0
                for _ in range(num_trials):
                    ex, _, states = task_2_3.step_HMM_inference({
                        "r": r,
                        "m": m,
                        "T": T,
                        "x0": x0,
                        "Rh": rh,
                    })
                    bex = task_2_3.compress_states(ex)
                    bstates = (states == r).astype(int)
                    ce_series = task_2_3.cross_entropy(bex, bstates)
                    sum_mean_CE += np.mean(ce_series)
                
                CE_mat[i, j] = sum_mean_CE / num_trials
            print(f'Progress: {i + 1}/{num_samples} (Rh)')

        np.save(save_filename, CE_mat)
        print(f"saved to {save_filename}")

    iv1_grid, iv2_grid = np.meshgrid(x0s, Rhs, indexing='ij')
    iv1_flat = iv1_grid.flatten()
    iv2_flat = iv2_grid.flatten()
    CE_flat = CE_mat.T.flatten() 

    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(iv1_flat, iv2_flat, CE_flat, levels=20, cmap='viridis_r')
    plt.colorbar(contour, label='Mean Cross-Entropy')
    plt.xlabel('x0')
    plt.ylabel('Rh')
    plt.title(r'Step model - BCE for varying $x_0, Rh$ (m=40, r=10)') # TODO
    plt.savefig('./plots/task_2_3_step_CE_contour_Rh_x0.png')
    plt.show()