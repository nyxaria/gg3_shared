# generate ridgeline CE plots

import numpy as np
import models
import scipy
import inference
from models_hmm import RampModelHMM
import os
import matplotlib.pyplot as plt
import pandas as pd
import joypy
import matplotlib.cm as cm

import task_2_3

np.set_printoptions(legacy='1.25') # don't show np.float; helps with debug


if __name__ == "__main__":
    num_trials = 15
    num_samples = 20
    K = 50
    T = 100
    save_filename = "./results/ridgeline_15b.npy"
    recompute = False

    betas = np.linspace(0, 4, num_samples)
    # sigmas = np.linspace(0, 4, num_samples)

    if os.path.exists(save_filename) and not recompute:
        print("loading " + save_filename)
        CE_mat = np.load(save_filename)
    else:
        print("computing")
        CE_mat = np.zeros((num_samples, T))

        for i1, s in enumerate(betas):
            sum_CE = 0
            for _ in range(num_trials):
                ex, expected_s, states = task_2_3.ramp_HMM_inference({
                    # "x0": 0.02,
                    "beta": s,
                    # "sigma": s, # for now
                    "K": K
                })
                sum_CE += task_2_3.cross_entropy(ex, states)
            CE_mat[i1] = sum_CE / num_trials

            print(f'Progress: {i1 + 1}/{num_samples} (beta)')

        np.save(save_filename, CE_mat)
        print(f"saved to {save_filename}")

    # smooth and take the negative differential
    dCE = np.diff(CE_mat, axis=1)
    dCE = scipy.ndimage.gaussian_filter1d(dCE, 2, axis=1)

    dCE *= -1
    ymax = np.max(dCE)
    dCE /= ymax

    df = pd.DataFrame(dCE.T)
    df.columns = [f"{s:.2f}" for s in betas]

    # Create ridgeline plot
    plt.figure(figsize=(10, 8))


    xrange = list(range(T))


    fig, axes = joypy.joyplot(df,
                            kind="values",
                            overlap=5/15,
                            fade=True,
                            linecolor='black',
                            linewidth=0.5,
                            colormap=cm.coolwarm,
                            ylim=(0, 1),
                            title='Decrease in Cross-Entropy over time for varying β',
                            x_range=xrange)

    plt.xlabel(r"t", fontsize=12)

    ax = axes[-1]
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("β")
    ax.yaxis.set_visible(True)
    ax.yaxis.set_ticks([])



    plt.savefig('./plots/ridgeline_15b.png')
