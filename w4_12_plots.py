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
N_TRIALS_LIST = [5, 10, 15, 20, 30, 50]
N_DATASETS=96

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
                           fig_size_factor=0.8)

    fn = "./results/UU_D" + str(N_DATASETS) + "_shape3_T" + str(N_TRIALS)
    s3_step_acc,s3_ramp_acc = w3_2.plot_confusion_matrix(fn + '.csv',
                               r'Confusion matrix, shape = 3, ' + str(N_TRIALS) + ' trials/dataset',
                               save_name=fn + '.png',
                               fig_size_factor=0.8)

    fn = "./results/UU_D" + str(N_DATASETS) + "_shape5_T" + str(N_TRIALS)
    s5_step_acc,s5_ramp_acc = w3_2.plot_confusion_matrix(fn + '.csv',
                               r'Confusion matrix, shape = 5, ' + str(N_TRIALS) + ' trials/dataset',
                               save_name=fn + '.png',
                               fig_size_factor=0.8)

    s1_step_accs.append(s1_step_acc)
    s1_ramp_accs.append(s1_ramp_acc)
    s3_step_accs.append(s3_step_acc)
    s3_ramp_accs.append(s3_ramp_acc)
    s5_step_accs.append(s5_step_acc)
    s5_ramp_accs.append(s5_ramp_acc)

# plot results
plt.figure(figsize=(10 * 0.8, 5 * 0.8))

# Plot s1 (baseline - black)
plt.plot(N_TRIALS_LIST, s1_ramp_accs, 'k--', label='poisson, ramp')  # black dashed
plt.plot(N_TRIALS_LIST, s1_step_accs, 'k-', label='poisson, step')   # black solid

# Plot s3 (lighter blue)
plt.plot(N_TRIALS_LIST, s3_ramp_accs, '--', color='#809bce', label='shape 3, ramp')
plt.plot(N_TRIALS_LIST, s3_step_accs, '-', color='#809bce', label='shape 3, step')

# Plot s5 (darker blue)
plt.plot(N_TRIALS_LIST, s5_ramp_accs, '--', color='#e27396', label='shape 5, ramp')
plt.plot(N_TRIALS_LIST, s5_step_accs, '-', color='#e27396', label='shape 5, step')

plt.title('Classification accuracy with different gamma distributions')
plt.xlabel('Number of trials')
plt.ylabel('Classification accuracy')
plt.legend()

plt.show()