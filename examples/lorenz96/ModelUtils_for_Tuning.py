"""
    Utilities to train DLDMD-based models in PyTorch

    Author: Jay Lago, NIWC Pacific, 55280
    Created: 20-Sept-2022
"""
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import datetime as dt
from tqdm import tqdm
from ray import tune
import pickle
import sys
import os


def testing_run(dataloader, model, loss_fn, window_cap):
    total_losses = np.array([])
    recon_losses = np.array([])
    pred_losses = np.array([])
    dmd_losses = np.array([])
    with tqdm(dataloader, unit='batch', desc='Testing/Validation', disable=True) as epoch:
        with torch.no_grad():
            for (cnt, batch) in enumerate(epoch):
                if cnt < window_cap:
                    results = model(batch)
                    losses = loss_fn(batch, results, model.parameters())
                    total_losses = np.append(total_losses, losses[0].item())
                    recon_losses = np.append(recon_losses, losses[1].item())
                    dmd_losses = np.append(dmd_losses, losses[2].item())
                    pred_losses = np.append(pred_losses, losses[3].item())
                else:
                    break
    avg_total_loss = np.mean(total_losses)
    avg_recon_loss = np.mean(recon_losses)
    avg_dmd_loss = np.mean(dmd_losses)
    avg_pred_loss = np.mean(pred_losses)
    return np.array([avg_total_loss, avg_recon_loss, avg_dmd_loss, avg_pred_loss])


def window_test(dataloader, model, loss_fn, window_cap, frac):
    init_nm_ob = model.num_observables
    num_obsvs_opt = model.num_observables
    all_losses = testing_run(dataloader, model, loss_fn, window_cap)
    min_loss = all_losses[0]

    for num_obsvs in range(init_nm_ob - model.max_win_stp, init_nm_ob + model.max_win_stp + 1):
        if num_obsvs >= 1 and num_obsvs != init_nm_ob:
            model.num_observables = num_obsvs
            model.window = model.n_ts - (num_obsvs - 1)
            loss_fn.num_observables = num_obsvs
            loss_fn.window = model.n_ts - (num_obsvs - 1)
            all_losses = testing_run(dataloader, model, loss_fn, window_cap)
            cur_tot_loss = all_losses[0]
            rel_change = np.abs(1.0 - cur_tot_loss/min_loss)
            if cur_tot_loss < min_loss and rel_change >= frac:
                num_obsvs_opt = num_obsvs
                min_loss = cur_tot_loss
    # Update window
    model.num_observables = num_obsvs_opt
    model.window = model.n_ts - (num_obsvs_opt - 1)
    loss_fn.num_observables = num_obsvs_opt
    loss_fn.window = model.n_ts - (num_obsvs_opt - 1)
    return


# ===========================================================================80
# Function Implementations
# ===========================================================================80
def train_model(model=None, hyp=None, loss_fn=None, optimizer=None,
                train_dataloader=None, val_dataloader=None):
    """Trains a model, saving model parameters to a checkpoints folder and
    returns the training loss history.

    Args:
        model (torch.nn.Module): Model to be trained.
        hyp (dict): Dictionary of model hyperparameters.
        loss_fn (torch.nn.Module): Loss function used during training.
        optimizer (torch.optim): Optimizer.
        train_dataloader (torch DataLoader): Data loader used to produce train batches per epoch.
        val_dataloader (torch DataLoader): Data loader used to produce val batches per epoch.

    Returns:
        history (dict): Dictionary containing history of each loss component.
    """
    # Set up training timer and objects to store training history
    t0 = dt.datetime.now()
    num_losses = loss_fn.num_loss_comps
    train_history = np.zeros((hyp['epochs'], num_losses+1))
    window_cap = 16

    # Begin training over epochs
    for epoch in range(hyp['epochs']):
        with tqdm(train_dataloader, unit='batch', desc='Train', disable=True) as train_epoch:
            for batch in train_epoch:
                # Clear out gradient
                optimizer.zero_grad()

                # Run model
                results = model(batch)

                # Compute losses
                losses = loss_fn(batch, results, model.parameters())
                loss = losses[0]

                # Compute gradient and update
                loss.backward()
                optimizer.step()

            # Storing the losses in a list for plotting
            train_history[epoch, :4] = [torch.log10(losses[ll]).item() for ll in range(num_losses)]
            train_history[epoch, 4] = model.num_observables
            # Save training history and model

        # Window size tuning
        #if epoch>0 and epoch % hyp['window_update_rate']==0:
        window_test(train_dataloader, model, loss_fn, window_cap, hyp['frac'])

    tune.report(loss = losses[0].item())
    return train_history


def device_setup(device_id=0, precision='single'):
    """Sets up a machine to train a deep learning model on a specified compute device.

    Args:
        device_id (int): Device identifier (i.e. 0 => cuda:0, 1 => cuda:1, etc.)

    Returns:
        device_avail (string): Type of device that is available (cuda vs cpu).
        this_dtype (string): Data type used for floats.
        this_cdtype (string): Data type used for complex.
    """
    if sys.platform.lower() == 'darwin':
        device_avail = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f'Apple GPU available: {torch.backends.mps.is_available()}')
        print(f'Apple Metal enabled: {torch.backends.mps.is_built()}')
        torch.device(device_avail)
        this_dtype = torch.float
        this_cdtype = torch.complex64
    else:
        device_avail = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Device available: {device_avail}')
        if device_avail == 'cuda':
            torch.cuda.set_device(f'cuda:{device_id}')
            print(f'Using device: {torch.cuda.get_device_name()}')
        else:
            torch.device(device_avail)
            print(f'Using device: {device_avail}')
        if precision == 'double':
            this_dtype = torch.float64
            this_cdtype = torch.complex128
        else:
            this_dtype = torch.float
            this_cdtype = torch.complex64
    print(f'Using precision (real/complex): {this_dtype}/{this_cdtype}')
    torch.set_default_dtype(this_dtype)
    return device_avail, this_dtype, this_cdtype


def seed_model(hyp):
    """Set the RNG see for both PyTorch and numpy functions.

    Args:
        hyp (dict): Model hyperparameters.
    """
    torch.manual_seed(hyp['seed'])
    np.random.seed(hyp['seed'])
    return

def make_data_sets(data, hyp):
    """Creates torch dataloader class for training.

    Args:
        data (numpy ndarray): Data set (initial conditions, time, dims)
        hyp (dict): Model hyperparameters.
    """
    np.random.shuffle(data)
    ii_train = hyp['num_train']
    ii_test = ii_train + hyp['num_test']
    ii_val = ii_test + hyp['num_val']
    # Create training data loader
    train_dataset = torch.tensor(data[:ii_train, ...], dtype=hyp['dtype'],
                              device=torch.device(hyp['device']))
    test_dataset = torch.tensor(data[ii_train:ii_test, ...], dtype=hyp['dtype'],
                             device=torch.device(hyp['device']))
    val_dataset = torch.tensor(data[ii_test:ii_val, ...], dtype=hyp['dtype'],
                               device=torch.device(hyp['device']))
    return train_dataset, test_dataset, val_dataset

def make_train_test(data, hyp):
    """Creates torch dataloader class for training.

    Args:
        data (numpy ndarray): Data set (initial conditions, time, dims)
        hyp (dict): Model hyperparameters.
    """
    train_dataset, test_dataset, val_dataset = make_data_sets(data, hyp)

    train_dataloader = DataLoader(train_dataset, batch_size=hyp['batch_size'],
                                  shuffle=True, drop_last=True)
    # Create testing data loader
    test_dataloader = DataLoader(test_dataset, batch_size=hyp['batch_size'],
                                 shuffle=True, drop_last=True)
    # Create validation data loader
    val_dataloader = DataLoader(val_dataset, batch_size=hyp['batch_size'],
                                shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader, val_dataloader
