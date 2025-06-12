import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import os
import w3_2
import w3_utils

if __name__ == "__main__":
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    K = 25
    T_MS = 100
    RH = 20
    M_GRID = 15
    X0 = 0.5
    N_DATASETS = 50 
    N_TRIALS = 25

    BETA_RANGE = (0, 4)
    SIGMA_RANGE = (0.04, 4)

    M_RANGE = (T_MS * 0.25, T_MS * 0.75)
    R_RANGE = (1, 6)

    ramp_param_specs = OD([
        ('beta', np.linspace(*BETA_RANGE, M_GRID)),
        ('sigma', np.exp(np.linspace(np.log(SIGMA_RANGE[0]),
                                     np.log(SIGMA_RANGE[1]),
                                     M_GRID))),
        ('x0', X0),
        ('K', K),
        ('T', T_MS),
        ('Rh', RH)
    ])

    step_param_specs = OD([
        ('m', np.linspace(*M_RANGE, M_GRID)),
        ('r', np.arange(R_RANGE[0], R_RANGE[1] + 1)),
        ('x0', X0),
        ('K', K),
        ('T', T_MS),
        ('Rh', RH)
    ])

    ramp_params_grid = w3_utils.make_params_grid(ramp_param_specs)
    step_params_grid = w3_utils.make_params_grid(step_param_specs)

    uniform_ramp_posterior = w3_utils.uniform_prior(ramp_params_grid)
    uniform_step_posterior = w3_utils.uniform_prior(step_params_grid)

    shape_values = [1, 2, 3, 4, 5]
    
    accuracies = {'ramp': [], 'step': []}

    for shape in shape_values:
        print(f"Running model selection for shape = {shape}")
        fn = f"./results/UU_D{N_DATASETS}_shape{shape}_T{N_TRIALS}.csv"
        
        if not os.path.exists(fn):
            w3_2.model_selection(
                ramp_params_grid, step_params_grid,
                uniform_ramp_posterior, uniform_step_posterior,  # generating
                uniform_ramp_posterior, uniform_step_posterior,  # inference
                N_DATASETS=N_DATASETS, N_TRIALS=N_TRIALS,
                ramp_gamma_shape=shape, step_gamma_shape=shape,
                save_to=fn
            )
        else:
            print(f"Results file already exists: {fn}")

        heatmap_savename = f'plots/task_4_1_1_shape{shape}_heatmap.png'
        confmat_savename = f'plots/task_4_1_1_shape{shape}_confmat.png'
        plot_title = f'Uniform prior, shape={shape}, {N_TRIALS} trials/dataset'
        
        w3_2.plot_heatmap(fn, plot_title, save_name=heatmap_savename, show=False)
        ramp_accuracy, step_accuracy = w3_2.plot_confusion_matrix(fn, plot_title, save_name=confmat_savename, show=False)
        accuracies['ramp'].append(ramp_accuracy)
        accuracies['step'].append(step_accuracy)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(shape_values, accuracies['ramp'], 'o-', label='Ramp Model Accuracy')
    plt.plot(shape_values, accuracies['step'], 'x--', label='Step Model Accuracy')
    
    avg_accuracy = (np.array(accuracies['ramp']) + np.array(accuracies['step'])) / 2
    plt.plot(shape_values, avg_accuracy, 's-.', label='Overall Accuracy')

    plt.title(f'Model Selection Accuracy vs. Gamma Shape Parameter (N_trials={N_TRIALS})')
    plt.xlabel('Gamma Shape Parameter')
    plt.ylabel('Accuracy')
    plt.xticks(shape_values)
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.savefig('plots/task_4_1_1_accuracy_vs_shape.png')
    plt.show() 