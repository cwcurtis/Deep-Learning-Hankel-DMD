"""
    Script to train an HDLDMD model on the Lorenz-63 system

    Author: Jay Lago, NIWC Pacific, 55280
    Created: 11-May-2023
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

# ===========================================================================80
# Setup
# ===========================================================================80
DEVICE, DTYPE, CDTYPE = mutils.device_setup(0, 'single')

# Hyperparameters
hyp = dict()
hyp['sim_start'] = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
hyp['experiment'] = 'Rossler_tune'
hyp['model_path'] = f"./checkpoints/{hyp['experiment']}_{hyp['sim_start']}/"
hyp['plot_freq'] = 5
hyp['device'] = DEVICE
hyp['dtype'] = DTYPE
hyp['cdtype'] = CDTYPE
hyp['seed'] = 1997
hyp['num_ic'] = 10000
hyp['num_train'] = 7000
hyp['num_test'] = 2000
hyp['num_val'] = 1000
hyp['t_final'] = 20
hyp['delta_t'] = 0.05
hyp['num_steps'] = int(hyp['t_final']/hyp['delta_t']) + 1
hyp['num_pred'] = hyp['num_steps']
hyp['epochs'] = 201
hyp['batch_size'] = 256
hyp['lr'] = 0.0001

# Hankel-DMD Window Size and Threshhold
hyp['max_win_stp'] = 1
hyp['ysteps'] = 20
hyp['frac'] = .25
hyp['threshhold'] = 16
hyp['tikhonov_param'] = 0.0
hyp['phys_dim'] = 3

# Change number of latent variables and number of observables
hyp['operation_mode'] = 'DLHDMD'
hyp['latent_dim'] = 3
hyp['num_observables'] = 10

# Neural network hyperparameters
hyp['num_neurons'] = 128
hyp['act_fn'] = torch.nn.ReLU()
hyp['enc_layers'] = [
    (hyp['phys_dim'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['latent_dim'])]
hyp['dec_layers'] = [
    (hyp['latent_dim'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['num_neurons']),
    (hyp['num_neurons'], hyp['phys_dim'])]

# Loss function parameters (recon, dmd, prediction, L2 on weights)
hyp['a1'] = 1.0
hyp['a2'] = 1.0
hyp['a3'] = 1.0
hyp['a4'] = 2.52271e-12

# Set RNG seeds
mutils.seed_model(hyp)

# Create save directories and store the hyperparameters
mutils.make_save_dirs(hyp)

# Generate / load data
data_fname = f"data_{hyp['experiment']}.pkl"
if os.path.exists(data_fname):
    # Load data from file
    data = pickle.load(open(data_fname, 'rb'))
else:
    # Create
    data = dutils.generate_rossler(
        x1min=-15, x1max=15, x2min=-20,
        x2max=20, x3min=0.0, x3max=40.0,
        num_ic=hyp['num_ic'],
        dt=hyp['delta_t'],
        tf=hyp['t_final'],
        seed=hyp['seed'],
    )
    pickle.dump(data, open(data_fname, 'wb'))

# Make train/test/val data sets
train_dataloader, test_dataloader, val_dataloader = mutils.make_train_test(data, hyp)

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
val_history = mutils.train_model(model, hyp, loss_fn, optimizer,
                             train_dataloader, val_dataloader)

print('done')
