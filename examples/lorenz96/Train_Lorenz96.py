"""
    Script to train an HDLDMD model on the Lorenz-96 system

    Author: Jay Lago, NIWC Pacific, 55280
    Created: 12-May-2023
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
hyp['experiment'] = 'lorenz96'
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
hyp['t_final'] = 20.0
hyp['delta_t'] = 0.05
hyp['num_steps'] = int(hyp['t_final']/hyp['delta_t']) + 1
hyp['num_pred'] = hyp['num_steps']
hyp['epochs'] = 20
hyp['batch_size'] = 256
hyp['lr'] = 1e-4

# Hankel-DMD Window Size and Threshhold
hyp['num_observables'] = 5
hyp['max_win_stp'] = 1
hyp['ysteps'] = 20
hyp['frac'] = .01
hyp['threshhold'] = 16
hyp['tikhonov_param'] = 1e-10

# Neural network hyperparameters
hyp['phys_dim'] = 20
hyp['latent_dim'] = 20
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
hyp['a4'] = 1e-11

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
    winval = 10.0
    xbounds = np.zeros((hyp['phys_dim'], 2), dtype=np.float64)
    for ll in range(hyp['phys_dim']):
        xbounds[ll, 0] = -winval
        xbounds[ll, 1] = winval
    data = dutils.generate_lorenz96(
        xbounds=xbounds,
        num_ic=hyp['num_ic'],
        dim=hyp['phys_dim'],
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
history = mutils.train_model(model, hyp, loss_fn, optimizer,
                             train_dataloader, val_dataloader)

print('done')
