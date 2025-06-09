import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
import sklearn

plt.style.use('ggplot')
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

save_path = './results/StepRamp.pth'
checkpoint = torch.load(save_path, weights_only=False)
model = StepRampClassifier(input_dim=300)  # Use the same architecture as before
model.load_state_dict(checkpoint['model_state_dict'])

scaler = checkpoint['scaler_state']
# Set model to evaluation mode
model.eval()



# do worse-case discrimination using the step approximation from before

rng = 69420

# generate data
T = 100
num_testpoints = 22000

grid_samples = 3 # take 10 samples along the range of each parameter

# parameter ranges for the train data
beta_range = [0, 4]
log_sigma_range = np.log([0.04, 4])
x0_range = [0, 0.5]

assert grid_samples ** 3 < 0.5 * num_testpoints
samples_per_param_set = (0.5 * num_testpoints) // (grid_samples ** 3)
samples_per_param_set = int(samples_per_param_set)

print('generating', (samples_per_param_set * 2 * grid_samples ** 3), 'samples out of the desired', num_testpoints)

# generate ramp data
beta_values = np.linspace(*beta_range, grid_samples)
sigma_values = np.linspace(*log_sigma_range, grid_samples)
sigma_values = np.exp(sigma_values)
x0_values = np.linspace(*x0_range, grid_samples)

# permute
B, s, X0 = np.meshgrid(beta_values, sigma_values , x0_values, indexing='ij')
combinations = np.stack((B, s, X0), axis=-1)
combinations = np.reshape(combinations, (grid_samples ** 3, 3))

# generate step data that well approximates the ramp
ramp_spikes = []
step_spikes = []

for combination in combinations:
    step_params = optimize_step(*combination, T)

    combination_step_spikes, _ = models.StepModel(*step_params).simulate(Ntrials=samples_per_param_set, T=T, get_rate=False)
    combination_ramp_spikes, _ = models.RampModel(*combination).simulate(Ntrials=samples_per_param_set, T=T, get_rate=False)

    ramp_spikes.append(combination_ramp_spikes)
    step_spikes.append(combination_step_spikes)

print(np.array(step_spikes).shape)

'''ramp_spikes = np.concatenate(ramp_spikes)
step_spikes = np.concatenate(step_spikes)

rc, re = PSTH(ramp_spikes, bins=np.arange(T+1))
sc, se = PSTH(step_spikes, bins=np.arange(T+1))

plt.plot(re[:-1], rc, label='ramp')
plt.plot(se[:-1], sc, label='step')
plt.legend()
plt.show()'''



batch_size = 20
num_batches = (samples_per_param_set * grid_samples ** 3) // batch_size

print('dataset (total spikes per each unique param combination) size is ', samples_per_param_set)

wrong = 0

for i_dataset in range(len(step_spikes)):
    # print(step_spikes[i_dataset].shape)
    X_step = compute_summary_statistics(step_spikes[i_dataset], batch_size)
    X_ramp = compute_summary_statistics(ramp_spikes[i_dataset], batch_size)


    X_step = scaler.transform(X_step)
    X_ramp = scaler.transform(X_ramp)


    X_step = torch.FloatTensor(X_step)
    X_ramp = torch.FloatTensor(X_ramp)

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():

        test_outputs = model(X_step).numpy()

        epsilon = 1e-3  # Small constant (e.g., machine epsilon)
        test_outputs = np.clip(test_outputs, epsilon, 1 - epsilon)

        LLR_step = np.sum(np.log(test_outputs)) - np.sum(np.log(1 - test_outputs))




        test_outputs = model(X_ramp).numpy()

        epsilon = 1e-3  # Small constant (e.g., machine epsilon)
        test_outputs = np.clip(test_outputs, epsilon, 1 - epsilon)

        LLR_ramp = np.sum(np.log(test_outputs)) - np.sum(np.log(1 - test_outputs))

        print('STEP', LLR_step, 'RAMP', LLR_ramp)

        if LLR_step > 0:
            wrong += 1

print('wrong', wrong)