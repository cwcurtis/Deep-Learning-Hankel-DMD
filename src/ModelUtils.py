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
import pickle
import DataUtils as dutils
import matplotlib.pyplot as plt
import sys
import os
from torchviz import make_dot


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
    init_loss = all_losses[0]
    min_loss = all_losses[0]
    min_recon_loss = all_losses[1]
    min_dmd_loss = all_losses[2]
    min_pred_loss = all_losses[3]

    #print(f"Incoming referenc loss: {min_loss}")
    for num_obsvs in range(init_nm_ob - model.max_win_stp, init_nm_ob + model.max_win_stp + 1):
        if num_obsvs >= 1 and num_obsvs != init_nm_ob:
            model.num_observables = num_obsvs
            model.window = model.n_ts - (num_obsvs - 1)
            loss_fn.num_observables = num_obsvs
            loss_fn.window = model.n_ts - (num_obsvs - 1)
            all_losses = testing_run(dataloader, model, loss_fn, window_cap)
            cur_tot_loss = all_losses[0]
            rel_change = np.abs(1.0 - cur_tot_loss/init_loss)
            #print(f"Current loss is: {cur_tot_loss}")
            #print(f"The current relative changes is: {rel_change}")
            if cur_tot_loss < min_loss and rel_change >= frac:
                num_obsvs_opt = num_obsvs
                min_loss = cur_tot_loss
                min_recon_loss = all_losses[1]
                min_dmd_loss = all_losses[2]
                min_pred_loss = all_losses[3]
    # Update window
    model.num_observables = num_obsvs_opt
    model.window = model.n_ts - (num_obsvs_opt - 1)
    loss_fn.num_observables = num_obsvs_opt
    loss_fn.window = model.n_ts - (num_obsvs_opt - 1)
    return [min_loss, min_recon_loss, min_dmd_loss, min_pred_loss]


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
    val_history = np.zeros((hyp['epochs'], num_losses+1))
    window_cap = int(np.floor(hyp['num_val']/hyp['batch_size']))
    operation_mode = hyp['operation_mode']
    # Begin training over epochs
    for epoch in range(hyp['epochs']):
        with tqdm(train_dataloader, unit='batch', desc='Train') as train_epoch:
            for batch in train_epoch:
                # Progress bar (set up)
                train_epoch.set_description("Epoch {}/{}:".format(epoch, hyp['epochs']))

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

                # Progress bar (update)
                loss_str = 'loss/recon/dmd/pred: ' + ''.join(
                    [f'{torch.log10(losses[ll]).item():2.2f} ' for ll in range(num_losses)])
                train_epoch.set_postfix_str(s=loss_str, refresh=True)

        # Storing the losses in a list for plotting
        train_history[epoch, :4] = [torch.log10(losses[ll]).item() for ll in range(num_losses)]
        train_history[epoch, 4] = model.num_observables
        # Save training history and model

        pickle.dump(train_history, open(''.join([hyp['model_path'], 'train_history.pkl']), 'wb'))
        torch.save(model.state_dict(), ''.join([hyp['model_path'], 'trained_model']))

        # Window size tuning
        if operation_mode == 'DLHDMD':
            current_total_loss = train_history[epoch, 0]
            updated_losses = window_test(train_dataloader, model, loss_fn, window_cap, hyp['frac'])

        # Do validation testing.
        val_losses = testing_run(val_dataloader, model, loss_fn, 100)
        val_history[epoch, :4] = np.log10(val_losses)
        val_history[epoch, 4] = model.num_observables
        pickle.dump(val_history, open(''.join([hyp['model_path'], 'val_history.pkl']), 'wb'))

        # Diagnostic plot
        if epoch % hyp['plot_freq'] == 0:
            vx = next(iter(val_dataloader))
            val_results = model(vx)
            vx = dutils.host_detach_np([vx])[0]
            _, vx_ae, vx_adv, _ = dutils.host_detach_np(val_results[:4])
            dutils.plot_all_logistics(vx_adv, val_history, train_history, epoch, hyp)

    # Store final number of observables and *then* save hyperparameters
    hyp['num_observables'] = model.num_observables
    pickle.dump(hyp, open(f"{hyp['model_path']}model_hyperparams.pkl", 'wb'))
    print("Training time: {} minutes".format((dt.datetime.now() - t0).total_seconds() / 60))
    return val_history

# ===========================================================================80
# Function Implementations
# ===========================================================================80
def restart_model(model=None, hyp=None, loss_fn=None, optimizer=None,
                train_dataloader=None, val_dataloader=None, train_history=None, val_history=None):
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
    window_cap = int(np.floor(hyp['num_val']/hyp['batch_size']))
    operation_mode = hyp['operation_mode']
    # Begin training over epochs
    for epoch in range(hyp['current_epoch'], hyp['epochs']):
        with tqdm(train_dataloader, unit='batch', desc='Train') as train_epoch:
            for batch in train_epoch:
                # Progress bar (set up)
                train_epoch.set_description("Epoch {}/{}:".format(epoch, hyp['epochs']))

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

                # Progress bar (update)
                loss_str = 'loss/recon/dmd/pred: ' + ''.join(
                    [f'{torch.log10(losses[ll]).item():2.2f} ' for ll in range(num_losses)])
                train_epoch.set_postfix_str(s=loss_str, refresh=True)

        # Storing the losses in a list for plotting
        train_history[epoch, :4] = [torch.log10(losses[ll]).item() for ll in range(num_losses)]
        train_history[epoch, 4] = model.num_observables
        # Save training history and model

        pickle.dump(train_history, open(''.join([hyp['model_path'], 'train_history.pkl']), 'wb'))
        torch.save(model.state_dict(), ''.join([hyp['model_path'], 'trained_model']))

        # Window size tuning
        if operation_mode == 'DLHDMD':
            current_total_loss = train_history[epoch, 0]
            updated_losses = window_test(train_dataloader, model, loss_fn, window_cap, hyp['frac'])

        # Do validation testing.
        val_losses = testing_run(val_dataloader, model, loss_fn, 100)
        val_history[epoch, :4] = np.log10(val_losses)
        val_history[epoch, 4] = model.num_observables
        pickle.dump(val_history, open(''.join([hyp['model_path'], 'val_history.pkl']), 'wb'))

        # Diagnostic plot
        if epoch % hyp['plot_freq'] == 0:
            vx = next(iter(val_dataloader))
            val_results = model(vx)
            vx = dutils.host_detach_np([vx])[0]
            _, vx_ae, vx_adv, _ = dutils.host_detach_np(val_results[:4])
            dutils.plot_all_logistics(vx_adv, val_history, train_history, epoch, hyp)

    # Store final number of observables and *then* save hyperparameters
    hyp['num_observables'] = model.num_observables
    pickle.dump(hyp, open(f"{hyp['model_path']}model_hyperparams.pkl", 'wb'))
    print("Training time: {} minutes".format((dt.datetime.now() - t0).total_seconds() / 60))
    return val_history

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


def make_save_dirs(hyp):
    """Creates/saves the model to file.

    Args:
        hyp (dict): Model hyperparameters.
    """
    if not os.path.exists(hyp['model_path']):
        os.makedirs(hyp['model_path'])
        os.makedirs(hyp['model_path'] + 'figs/')
    return

def make_data_sets(data, hyp):
    """Creates torch dataloader class for training.

    Args:
        data (numpy ndarray): Data set (initial conditions, time, dims)
        hyp (dict): Model hyperparameters.
    """
    #np.random.shuffle(data)
    permvec = np.random.permutation(np.arange(data.shape[0]))
    pickle.dump(permvec, open(f"{hyp['model_path']}permvec.pkl", 'wb'))
    shuffled_data = data[permvec, ...]

    ii_train = hyp['num_train']
    ii_test = ii_train + hyp['num_test']
    ii_val = ii_test + hyp['num_val']
    # Create training data loader
    train_dataset = torch.tensor(shuffled_data[:ii_train, ...], dtype=hyp['dtype'],
                              device=torch.device(hyp['device']))
    test_dataset = torch.tensor(shuffled_data[ii_train:ii_test, ...], dtype=hyp['dtype'],
                             device=torch.device(hyp['device']))
    val_dataset = torch.tensor(shuffled_data[ii_test:ii_val, ...], dtype=hyp['dtype'],
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
