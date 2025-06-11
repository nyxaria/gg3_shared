import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as OD
import scipy.special
import os
import scipy
from scipy.interpolate import griddata
import w3_2
import w3_utils
from models_hmm import RampModelHMM
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
import multiprocessing
from tqdm import tqdm
import pandas as pd
from ML_models import StepRampClassifier, compute_summary_statistics
from models_hmm import RampModelHMM, StepModelHMM

# Define the files and their corresponding sigma_frac values
files = [
    "C:/Users/henry/Github/gg3_shared/results/0.5GU_D240_T3.csv",
    "C:/Users/henry/Github/gg3_shared/results/0.25GU_D240_T3.csv",
    "C:/Users/henry/Github/gg3_shared/results/0.125GU_D240_T3.csv",
    "C:/Users/henry/Github/gg3_shared/results/UU_D240_T3.csv"
]
sigma_fracs = [0.5, 0.25, 0.125, 0]  # Corresponding sigma_frac values

step_accs = []
ramp_accs = []

# Process each file
for fn in files:
    step_acc, ramp_acc = w3_2.plot_confusion_matrix(fn,
                       f'Confusion matrix, sigma_frac = {sigma_fracs[files.index(fn)]}',
                       save_name=fn.replace('.csv', '.png'),
                       fig_size_factor=0.8)
    
    step_accs.append(step_acc)
    ramp_accs.append(ramp_acc)

# Plot results
plt.figure(figsize=(10 * 0.8, 5 * 0.8))

# Plot ramp (dashed) and step (solid) accuracies
plt.plot(sigma_fracs, ramp_accs, 'k--', label='ramp')  # black dashed
plt.plot(sigma_fracs, step_accs, 'k-', label='step')   # black solid

plt.title('Classification accuracy vs sigma_frac')
plt.xlabel('sigma_frac')
plt.ylabel('Classification accuracy')
plt.legend()

plt.show()