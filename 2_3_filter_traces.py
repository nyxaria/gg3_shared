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
    # step implementation

    trials = 500
    trials_to_plot = 1
    T = 100
    '''fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [4, 1]})

    CE_sum = np.zeros(T)
    fCE_sum = np.zeros(T)

    colors = plt.cm.coolwarm(np.linspace(1, 0, trials_to_plot))

    for trial in range(trials):
        ex, fex, bpred_s, fbpred_s, states = task_2_3.step_HMM_inference({
            'T': T,
            'Rh': 20,
        }, test_filtering=True)

        bex = task_2_3.compress_states(ex)
        fbex = task_2_3.compress_states(fex)
        bstates = (states == 10).astype(int) # TODO if r, set == r
        
        CE_sum += task_2_3.cross_entropy(bex, bstates)
        fCE_sum += task_2_3.cross_entropy(fbex, bstates)

        # ax1.matshow(ex.T)

        if trial < trials_to_plot:
            alpha = 1 if trial == 0 else 0.15

            color = ['#111111', '#111111', '#111111'] if trial == 0 else [red, blue, '#aaaaaa']
            color = [red, blue, '#111111']

            # calc expected vals for plotting purposes
            expected_s = ex @ np.arange(ex.shape[1])
            fexpected_s = fex @ np.arange(ex.shape[1])

            ax1.plot(np.arange(T), expected_s, color=color[0], linestyle="dashed",
                     label=f'Trial {trial + 1} (Smoothing Pred.)' if trial == 0 else '', alpha=alpha)
            ax1.plot(np.arange(T), fexpected_s, color=color[1], linestyle="dotted",
                     label=f'Trial {trial + 1} (Filtered Pred.)' if trial == 0 else '', alpha=alpha)

            ax1.plot(np.arange(T), states, color=color[2], label=f'Trial {trial + 1} (True)', alpha=alpha)

            first_pred_index = np.argmax(bpred_s == 1) if np.any(bpred_s == 1) else None
            filter_first_pred_index = np.argmax(fbpred_s == 1) if np.any(fbpred_s == 1) else None



            if trial == 0:
                sstd = np.sqrt(np.sum(
                    (np.arange(ex.shape[1]) - np.sum(np.arange(ex.shape[1]) * ex, axis=1, keepdims=True)) ** 2 * ex,
                    axis=1))
                fstd = np.sqrt(np.sum(
                    (np.arange(fex.shape[1]) - np.sum(np.arange(fex.shape[1]) * fex, axis=1, keepdims=True)) ** 2 * fex,
                    axis=1))

                smooth_color = red
                filter_color = blue

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
                print(first_pred_index, filter_first_pred_index)
                if first_pred_index is not None:
                    ax1.scatter(first_pred_index, states[first_pred_index],
                                color=color[0], marker='d', s=100, alpha=alpha,
                                label=f'Smoothed Jump Time',
                                zorder=3)

                if filter_first_pred_index is not None:
                    ax1.scatter(filter_first_pred_index, states[filter_first_pred_index],
                                color=color[1], marker='d', s=100, alpha=alpha,
                                label=f'Filtered Jump Time',
                                zorder=3)

        print('completed', trial)

    ax2.plot(np.arange(T), CE_sum / trials, color=red, linewidth=2)
    ax2.plot(np.arange(T), fCE_sum / trials, color=blue, linewidth=2)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cross-Entropy')
    ax2.set_title('Average Cross-Entropy Over Time (' + str(trials) + ' trials)')

    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax1.set_title("Markov state (s) over time")
    ax1.set_ylabel("Markov state (proportional to rate)")
    ax1.legend(unique_labels.values(), unique_labels.keys(),
               loc='lower right', ncol=2, framealpha=0.5)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.show()
    plt.savefig('./plots/task_2_3_step_filter_traces_final.png')'''



    # Ramp implementation
    
    trials = 50
    trials_to_plot = 1
    T = 100
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [4, 1]})

    CE_sum = np.zeros(T)
    fCE_sum = np.zeros(T)

    colors = plt.cm.coolwarm(np.linspace(1, 0, trials_to_plot))

    for trial in range(trials):
        ex, fex, expected_s, fexpected_s, states = task_2_3.ramp_HMM_inference({
            'T': T,
            'beta': 1.,
            'sigma': 0.35,
            'Rh': 50,
        }, test_filtering=True)


        CE_sum += task_2_3.cross_entropy(ex, states)
        fCE_sum += task_2_3.cross_entropy(fex, states)

        # ax1.matshow(ex.T)

        if trial < trials_to_plot:
            alpha = 1 if trial == 0 else 0.15

            color = ['#666666', '#666666', '#666666'] if trial == 0 else [red, blue, '#aaaaaa']
            color = [red, blue, '#aaaaaa']

            ax1.plot(np.arange(T), expected_s, color=color[0], linestyle="dashed",
                     label=f'Trial {trial + 1} (Smoothing Pred.)', alpha=alpha)
            ax1.plot(np.arange(T), fexpected_s, color=color[1], linestyle="dotted", label=f'Trial {trial + 1} (Filtered Pred.)', alpha=alpha)

            ax1.plot(np.arange(T), states, color=color[2], label=f'Trial {trial + 1} (True)', alpha=alpha)

            if trial == 0:
                sstd = np.sqrt(np.sum(
                    (np.arange(ex.shape[1]) - np.sum(np.arange(ex.shape[1]) * ex, axis=1, keepdims=True)) ** 2 * ex,
                    axis=1))
                fstd = np.sqrt(np.sum(
                    (np.arange(fex.shape[1]) - np.sum(np.arange(fex.shape[1]) * fex, axis=1, keepdims=True)) ** 2 * fex,
                    axis=1))

                smooth_color = red
                filter_color = blue

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

    ax2.plot(np.arange(T), CE_sum / trials, color=red, linewidth=2)
    ax2.plot(np.arange(T), fCE_sum / trials, color=blue, linewidth=2)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cross-Entropy')
    ax2.set_title('Average Cross-Entropy Over Time (' + str(trials) + ' trials)')

    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax1.set_title("Markov state (s) over time")
    ax1.set_ylabel("Markov state (proportional to rate)")
    ax1.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper left', ncol=2, framealpha=0.5)

    plt.tight_layout()
    plt.show()
    plt.savefig('./plots/task_2_3_ramp_filter_traces_final.png')
