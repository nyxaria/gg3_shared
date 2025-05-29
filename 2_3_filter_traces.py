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

red = '#ff9999'
blue = '#99ccff'

if __name__ == "__main__":
    trials = 50
    trials_to_plot = 3
    T = 100
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [4, 1]})

    CE_sum = np.zeros(T)
    fCE_sum = np.zeros(T)

    colors = plt.cm.coolwarm(np.linspace(1, 0, trials_to_plot))

    for trial in range(trials):
        ex, fex, expected_s, fexpected_s, states = task_2_3.ramp_HMM_inference({
            'T': T,
            'beta': 1.5,
            'sigma': 0.35,
            'Rh': 20,
        }, test_filtering=True)

        # Create a figure with two subplots (2 rows, 1 column)

        # true_s = (xs * (K - 1)).flatten().astype(int)
        CE_sum += task_2_3.cross_entropy(ex, states)
        fCE_sum += task_2_3.cross_entropy(fex, states)

        # Main plot (top)
        # ax1.matshow(ex.T)

        if trial < trials_to_plot:
            alpha = 1 if trial == 0 else 0.15
            # color = colors[trial] HORRIBLE CODE

            color = ['#666666', '#666666', '#666666'] if trial == 0 else [red, blue, '#aaaaaa']
            color = [red, blue, '#aaaaaa']

            ax1.plot(np.arange(T), expected_s, color=color[0], linestyle="dashed",
                     label=f'Trial {trial + 1} (Smoothing Pred.)', alpha=alpha)
            ax1.plot(np.arange(T), fexpected_s, color=color[1], linestyle="dotted", label=f'Trial {trial + 1} (Filtered Pred.)', alpha=alpha)

            ax1.plot(np.arange(T), states, color=color[2], label=f'Trial {trial + 1} (True)', alpha=alpha)

            if trial == 0: # plot stdev
                sstd = np.sqrt(np.sum(
                    (np.arange(ex.shape[1]) - np.sum(np.arange(ex.shape[1]) * ex, axis=1, keepdims=True)) ** 2 * ex,
                    axis=1))
                fstd = np.sqrt(np.sum(
                    (np.arange(fex.shape[1]) - np.sum(np.arange(fex.shape[1]) * fex, axis=1, keepdims=True)) ** 2 * fex,
                    axis=1))

                # print(sstd)
                smooth_color = red
                filter_color = blue

                # Plot ±2 standard deviation regions
                ax1.fill_between(np.arange(T),
                                 expected_s - 2 * sstd,
                                 expected_s + 2 * sstd,
                                 color=smooth_color, alpha=0.3, label='Smoothing ±2σ',
                                 edgecolor=smooth_color, linewidth=1)

                ax1.fill_between(np.arange(T),
                                 fexpected_s - 2 * fstd,
                                 fexpected_s + 2 * fstd,
                                 color=filter_color, alpha=0.3, label='Filtering ±2σ',
                                 edgecolor=filter_color, linewidth=1)

                print('fill between')

        print('completed', trial)

    # Cross-entropy plot (bottom)
    ax2.plot(np.arange(T), CE_sum / trials, color=red, linewidth=2)
    ax2.plot(np.arange(T), fCE_sum / trials, color=blue, linewidth=2)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cross-Entropy')
    ax2.set_title('Average Cross-Entropy Over Time (' + str(trials) + ' trials)')

    # Add legend to the top plot (reduce redundancy with a single entry per trial)
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # Remove duplicates
    ax1.set_title("Markov state (s) over time")
    ax1.set_ylabel("Markov state (proportional to rate)")
    ax1.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper left', ncol=2, framealpha=0.5)

    plt.tight_layout()
    plt.savefig('./plots/ramp_filter_traces_highbeta.png')
