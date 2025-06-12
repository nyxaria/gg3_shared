import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import os
import w3_2
import w3_utils
from matplotlib.lines import Line2D

var = lambda a, b, frac: ((b-a) * frac) ** 2
mean = lambda a, b: (b+a)/2

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})
N_TRIALS_LIST = [5, 10, 15, 20, 30, 50]
N_DATASETS=96
STD_FRACTION = 0.25
K = 25
T_MS = 100
RH = 20
M_GRID = 15
X0 = 0.2

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

step_param_specs = OD([('m', np.linspace(*M_RANGE, M_GRID)),
                        ('r', np.linspace(*R_RANGE, 6).astype(int)),
                        ('x0', X0),
                        ('K', K),
                        ('T', T_MS),
                        ('Rh', RH)])

ramp_params_grid = w3_utils.make_params_grid(ramp_param_specs)
step_params_grid = w3_utils.make_params_grid(step_param_specs)

uniform_ramp_posterior = w3_utils.uniform_prior(ramp_params_grid)
uniform_step_posterior = w3_utils.uniform_prior(step_params_grid)


gauss_ramp_accs = []
gauss_step_accs = []
s1_ramp_accs = []
s1_step_accs = []
s3_ramp_accs = []
s3_step_accs = []
s5_ramp_accs = []
s5_step_accs = []

for N_TRIALS in N_TRIALS_LIST:
    fn = "./results/UU_D" + str(N_DATASETS) + "_shape1_T" + str(N_TRIALS)
    s1_step_acc, s1_ramp_acc = w3_2.plot_confusion_matrix(fn + '.csv',
                           r'Confusion matrix, shape = 1, ' + str(N_TRIALS) + ' trials/dataset',
                           save_name=fn + '.png',
                           fig_size_factor=0.8,
                           show=False)

    fn = "./results/UU_D" + str(N_DATASETS) + "_shape3_T" + str(N_TRIALS)
    s3_step_acc,s3_ramp_acc = w3_2.plot_confusion_matrix(fn + '.csv',
                               r'Confusion matrix, shape = 3, ' + str(N_TRIALS) + ' trials/dataset',
                               save_name=fn + '.png',
                               fig_size_factor=0.8,
                               show=False)

    fn = "./results/UU_D" + str(N_DATASETS) + "_shape5_T" + str(N_TRIALS)
    s5_step_acc,s5_ramp_acc = w3_2.plot_confusion_matrix(fn + '.csv',
                               r'Confusion matrix, shape = 5, ' + str(N_TRIALS) + ' trials/dataset',
                               save_name=fn + '.png',
                               fig_size_factor=0.8,
                               show=False)

    fn = "./results/GU_D" + str(50) + "_T" + str(N_TRIALS) + "_SF" + str(STD_FRACTION)

    gauss_step_acc,gauss_ramp_acc = w3_2.plot_confusion_matrix(fn + '.csv',
                               r'Confusion matrix, gaussian (sf=' + str(STD_FRACTION) + '), ' + str(N_TRIALS) + ' trials/dataset',
                               save_name=fn + '.png',
                               fig_size_factor=0.8,
                               show=False)
    gauss_ramp_accs.append(gauss_ramp_acc)
    gauss_step_accs.append(gauss_step_acc)

    s1_step_accs.append(s1_step_acc)
    s1_ramp_accs.append(s1_ramp_acc)
    s3_step_accs.append(s3_step_acc)
    s3_ramp_accs.append(s3_ramp_acc)
    s5_step_accs.append(s5_step_acc)
    s5_ramp_accs.append(s5_ramp_acc)
    print(N_TRIALS)

# plot results
plt.figure(figsize=(12, 8))

colors = {
    'poisson': 'k',
    'gaussian': 'blue',
    'shape3': 'green',
    'shape5': 'red'
}
labels = {
    'poisson': 'Baseline (Poisson, Uniform Prior)',
    'gaussian': 'Prior Mismatch (Gaussian Prior)',
    'shape3': 'Likelihood Mismatch (Shape=3)',
    'shape5': 'Likelihood Mismatch (Shape=5)'
}


plt.plot(N_TRIALS_LIST, s1_ramp_accs, color=colors['poisson'], linestyle='--', marker='o')
plt.plot(N_TRIALS_LIST, s1_step_accs, color=colors['poisson'], linestyle='-', marker='x')

plt.plot(N_TRIALS_LIST, gauss_ramp_accs, color=colors['gaussian'], linestyle='--', marker='o')
plt.plot(N_TRIALS_LIST, gauss_step_accs, color=colors['gaussian'], linestyle='-', marker='x')

plt.plot(N_TRIALS_LIST, s3_ramp_accs, color=colors['shape3'], linestyle='--', marker='o')
plt.plot(N_TRIALS_LIST, s3_step_accs, color=colors['shape3'], linestyle='-', marker='x')

plt.plot(N_TRIALS_LIST, s5_ramp_accs, color=colors['shape5'], linestyle='--', marker='o')
plt.plot(N_TRIALS_LIST, s5_step_accs, color=colors['shape5'], linestyle='-', marker='x')

plt.title('Model Selection Accuracy vs. Number of Trials')
plt.xlabel('Number of Trials')
plt.ylabel('Accuracy')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(0.4, 1.05)


legend_elements_conditions = [
    Line2D([0], [0], color=colors['poisson'], lw=2, label=labels['poisson']),
    Line2D([0], [0], color=colors['gaussian'], lw=2, label=labels['gaussian']),
    Line2D([0], [0], color=colors['shape3'], lw=2, label=labels['shape3']),
    Line2D([0], [0], color=colors['shape5'], lw=2, label=labels['shape5'])
]

legend_elements_models = [
    Line2D([0], [0], color='gray', linestyle='--', marker='o', label='Ramp Model'),
    Line2D([0], [0], color='gray', linestyle='-', marker='x', label='Step Model')
]

ax = plt.gca()
leg1 = ax.legend(handles=legend_elements_conditions, title='Conditions', loc='lower right')
ax.add_artist(leg1)
leg2 = ax.legend(handles=legend_elements_models, title='Model Type', loc='center right')


plt.tight_layout()
plt.show()