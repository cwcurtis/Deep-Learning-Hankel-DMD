"""
    Script to train an HDLDMD model on the Kuramoto-Sivashinsky system

    Author: Jay Lago, NIWC Pacific, 55280
    Created: 13-May-2023
"""
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pickle
import os
import sys

#sys.path.insert(0, '../../src')
import Models_for_Tuning as models
import ModelUtils_for_Tuning as mutils

import torch.optim as optim
import torchvision

from functools import partial
from ray import tune, init
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from scipy.io import loadmat

def load_data(data_dir="./data"):
    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    #with FileLock(os.path.expanduser("~/.data.lock")):
    time_coeffs_dict = loadmat(data_dir  + '/time_coeffs_small.mat')
    time_coeffs = np.transpose(time_coeffs_dict['time_coeffs'],(0,2,1))

    return time_coeffs

def tune_model(config, data_dir):
    DEVICE, DTYPE, CDTYPE = mutils.device_setup(0, 'single')
    mutils.seed_model(config)
    time_coeffs = load_data(data_dir)
    # Make train/test/val data sets
    train_dl, test_dl, val_dl = mutils.make_train_test(time_coeffs, config)
    model = models.Hankel_DLDMD(config)
    model.to(torch.device(config['device']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = models.Hankel_DLDMD_Loss(config)
    history = mutils.train_model(model, config, loss_fn, optimizer,
                                 train_dl, val_dl)
    print('done')

def main_fit_approach(config, num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    data_dir = os.path.abspath('./time_coeffs_small.mat')

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(tune_model, data_dir=data_dir),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    result = tuner.fit()

    best_trial = result.get_best_result("loss", "min")
    with open("Tuning_Results_"+config['sim_start']+".txt", "w") as f:
        for key in best_trial.config:
            print("Param: {}= {}".format(key, best_trial.config[key]), file=f)
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]), file=f)

def main(config, num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    data_dir = os.getcwd()
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(tune_model, data_dir=data_dir),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    with open("Tuning_Results_"+config['sim_start']+".txt", "w") as f:
        for key in best_trial.config:
            print("Param: {}= {}".format(key, best_trial.config[key]), file=f)
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]), file=f)

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    # ===========================================================================80
    # Setup
    # ===========================================================================80
    runtime_env = {"working_dir": os.getcwd(),
                'excludes': ['/content/drive/MyDrive/Hankel_DMD_Torch/torch/examples/kuramoto-sivashinsky/modes_small.mat',
                '/content/drive/MyDrive/Hankel_DMD_Torch/torch/examples/kuramoto-sivashinsky/time_coeffs_small.mat']
    }
    init(runtime_env=runtime_env)
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
    hyp['epochs'] = 10

    # Hankel-DMD Window Size and Threshhold
    hyp['num_observables'] = 5
    hyp['max_win_stp'] = 1
    hyp['ysteps'] = 14
    hyp['frac'] = .15
    hyp['threshhold'] = 10
    hyp['tikhonov_param'] = 1e-10
    hyp['lag_number'] = 1

    # Neural network hyperparameters
    hyp['phys_dim'] = 12
    hyp['latent_dim'] = 12
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

    hyp['batch_size'] = tune.choice([64, 128])
    hyp['lr'] = tune.loguniform(1e-5,1e-2)
    hyp['a4'] = tune.loguniform(1e-12,1e-8)

    main(hyp, num_samples=5, max_num_epochs=10, gpus_per_trial=1)
