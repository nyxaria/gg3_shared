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

for N_TRIALS in N_TRIALS_LIST:
    fn = "./results/UU_D" + str(N_DATASETS) + "_shape1_T" + str(N_TRIALS) + ".csv"
    w3_2.plot_confusion_matrix(fn + '.csv',
                           r'Confusion matrix, shape = 1, ' + str(N_TRIALS) + ' trials/dataset',
                           save_name=fn + '.png',
                           fig_size_factor=0.8)

    fn = "./results/UU_D" + str(N_DATASETS) + "_shape3_T" + str(N_TRIALS) + ".csv"
    w3_2.plot_confusion_matrix(fn + '.csv',
                               r'Confusion matrix, shape = 3, ' + str(N_TRIALS) + ' trials/dataset',
                               save_name=fn + '.png',
                               fig_size_factor=0.8)

    fn = "./results/UU_D" + str(N_DATASETS) + "_shape5_T" + str(N_TRIALS) + ".csv"
    w3_2.plot_confusion_matrix(fn + '.csv',
                               r'Confusion matrix, shape = 5, ' + str(N_TRIALS) + ' trials/dataset',
                               save_name=fn + '.png',
                               fig_size_factor=0.8)