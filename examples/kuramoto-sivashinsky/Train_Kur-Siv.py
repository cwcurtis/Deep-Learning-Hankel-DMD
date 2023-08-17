"""
    Script to train an HDLDMD model on the Kuramoto-Sivashinsky system

    Author: Jay Lago, NIWC Pacific, 55280
    Created: 13-May-2023
"""
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pickle
import os
import sys

sys.path.insert(0, '../../src')
import Models as models
import ModelUtils as mutils
import DataUtils as dutils
from scipy.io import loadmat

# ===========================================================================80
# Setup
# ===========================================================================80
DEVICE, DTYPE, CDTYPE = mutils.device_setup(0, 'single')

# Hyperparameters
hyp = dict()
hyp['sim_start'] = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
hyp['experiment'] = 'kur_siv'
hyp['model_path'] = f"./checkpoints/{hyp['experiment']}_{hyp['sim_start']}/"
hyp['plot_freq'] = 5
hyp['device'] = DEVICE
hyp['dtype'] = DTYPE
hyp['cdtype'] = CDTYPE
hyp['seed'] = 1997
hyp['num_ic'] = 8000
hyp['num_train'] = 7000
hyp['num_test'] = 500
hyp['num_val'] = 500
hyp['t_final'] = int((11./np.pi)**4)
hyp['delta_t'] = 0.25
hyp['num_steps'] = int(hyp['t_final']/hyp['delta_t']) + 1
hyp['num_pred'] = hyp['num_steps']
hyp['epochs'] = 101
hyp['batch_size'] = 64
hyp['lr'] = .00081

# Hankel-DMD Window Size and Threshhold
hyp['max_win_stp'] = 1
hyp['ysteps'] = 14
hyp['frac'] = .15
hyp['threshhold'] = 10
hyp['tikhonov_param'] = 1e-10
hyp['lag_number'] = 1
hyp['phys_dim'] = 12

# Change number of latent variables and number of observables
hyp['latent_dim'] = 12
hyp['operation_mode'] = 'DLHDMD'
hyp['num_observables'] = 5

# Neural network hyperparameters
hyp['num_neurons'] = 128
hyp['act_fn'] = torch.nn.ReLU()
hyp['enc_layers'] = [
    (hyp['phys_dim'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['latent_dim'])]
hyp['dec_layers'] = [
    (hyp['latent_dim'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['phys_dim'])]

# Loss function parameters (recon, dmd, prediction, L2 on weights)
hyp['a1'] = 1.0
hyp['a2'] = 1.0
hyp['a3'] = 1.0
hyp['a4'] = 4.85e-12

# Set RNG seeds
mutils.seed_model(hyp)

# Create save directories and store the hyperparameters
mutils.make_save_dirs(hyp)

# Generate / load data
data_fname = f"time_coeffs_small.mat"
if os.path.exists(data_fname):
    time_coeffs_dict = loadmat('time_coeffs_small.mat')
    modes_dict = loadmat('modes_small.mat')
    time_coeffs = time_coeffs_dict['time_coeffs']
    print(np.shape(time_coeffs))
    modes = modes_dict['modes']

    # Make train/test/val data sets
    train_dataloader, test_dataloader, val_dataloader = mutils.make_train_test(np.transpose(time_coeffs,(0, 2, 1)), hyp)

    # Initialize the model for training
    model = models.Hankel_DLDMD(hyp)

    # Move the model to the compute device
    model.to(torch.device(hyp['device']))

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr'])

    # Set the loss function
    loss_fn = models.Hankel_DLDMD_Loss(hyp)

    # ===========================================================================80
    # Train
    # ===========================================================================80
    history = mutils.train_model(model, hyp, loss_fn, optimizer,
                                 train_dataloader, val_dataloader)

    print('done')
else:
    print("[ERROR] No Kuramoto-Sivashinsky data set found")
