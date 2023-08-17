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
# ===========================================================================80
# Setup
# ===========================================================================80
DEVICE, DTYPE, CDTYPE = mutils.device_setup(0, 'single')

# Hyperparameters
hyp = dict()
#hyp['sim_start'] = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
#hyp['experiment'] = 'kur_siv'
#hyp['model_path'] = f"./checkpoints/{hyp['experiment']}_{hyp['sim_start']}/"
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

# Neural network hyperparameters
hyp['phys_dim'] = 12
hyp['latent_dim'] = 12
hyp['operation_mode'] = 'DLHDMD'
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

chkpt_folder = 'kur_siv_2023-07-09-0026'
hyp['model_path'] = './checkpoints/' + chkpt_folder + '/'
perm_fname = ''.join(['./checkpoints/', chkpt_folder, '/permvec.pkl'])
train_history_fname = ''.join(['./checkpoints/', chkpt_folder, '/train_history.pkl'])
val_history_fname = ''.join(['./checkpoints/', chkpt_folder, '/val_history.pkl'])

perm = pickle.load(open(perm_fname , 'rb'))
train_history = pickle.load(open(train_history_fname , 'rb'))
val_history = pickle.load(open(val_history_fname , 'rb'))

time_coeffs_dict = loadmat('time_coeffs_small.mat')
data = np.transpose(time_coeffs_dict['time_coeffs'], (0, 2, 1))
shuffled_data = data[perm, ...]

ii_train = hyp['num_train']
ii_test = ii_train + hyp['num_test']
ii_val = ii_test + hyp['num_val']

train_dataset = torch.tensor(shuffled_data[:ii_train, ...], dtype=hyp['dtype'],
                          device=torch.device(hyp['device']))

test_dataset = torch.tensor(shuffled_data[ii_train:ii_test, ...], dtype=hyp['dtype'],
                          device=torch.device(hyp['device']))

val_dataset = torch.tensor(shuffled_data[ii_test:, ...], dtype=hyp['dtype'],
                          device=torch.device(hyp['device']))

train_dataloader = DataLoader(train_dataset, batch_size=hyp['batch_size'],
                              shuffle=True, drop_last=True) 
                              
test_dataloader = DataLoader(test_dataset, batch_size=hyp['batch_size'],
                              shuffle=True, drop_last=True) 
                              
val_dataloader = DataLoader(val_dataset, batch_size=hyp['batch_size'],
                              shuffle=True, drop_last=True) 

# Load the trained model
model_fname = ''.join(['./checkpoints/', chkpt_folder, '/trained_model'])
hyp['num_observables'] = 12 # Don't love this, fix later
model = models.Hankel_DLDMD(hyp)
model.load_state_dict(torch.load(model_fname, map_location=torch.device(hyp['device'])))
model.eval()
model.to(torch.device(hyp['device']))

optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr'])
loss_fn = models.Hankel_DLDMD_Loss(hyp)

#nzero_inds = np.abs(val_history[:,0]) > 0.
#hyp['current_epoch'] = np.sum(nzero_inds) # set remaining number of epochs 
hyp['current_epoch'] = 90 # set remaining number of epochs 

print(f"Current epoch is: {hyp['current_epoch']}")

history = mutils.restart_model(model, hyp, loss_fn, optimizer, train_dataloader, 
                                            val_dataloader, train_history, val_history)

print('done')
