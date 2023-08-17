"""
    Implementation of the Hankel-based Deep Learning Dynamic Mode Decomposition
    in PyTorch

    Author: Jay Lago, NIWC Pacific, 55280
    Created: 11-May-2023
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from scipy.linalg import hankel

# ===========================================================================80
# Model Implementations
# ===========================================================================80
class Hankel_DLDMD(nn.Module):
    def __init__(self, hyp):
        super(Hankel_DLDMD, self).__init__()

        # Hyperparameters
        self.bs = hyp['batch_size']
        self.p_dim = hyp['phys_dim']
        self.l_dim = hyp['latent_dim']
        self.n_ts = hyp['num_pred']
        self.device = hyp['device']
        self.dtype = hyp['dtype']
        self.cdtype = hyp['cdtype']
        self.enc_layers = hyp['enc_layers']
        self.dec_layers = hyp['dec_layers']
        self.act_fn = hyp['act_fn']

        self.ysteps = hyp['ysteps']
        self.tikhonov_param = hyp['tikhonov_param']
        self.num_observables = hyp['num_observables']
        self.window = self.n_ts - (self.num_observables - 1)
        self.max_win_stp = hyp['max_win_stp']

        self.operation_mode = hyp['operation_mode']

        # Neural Networks
        encoder = []
        for ll in range(len(self.enc_layers)):
            if ll < len(self.enc_layers)-1:
                encoder.append(nn.Linear(self.enc_layers[ll][0], self.enc_layers[ll][1]))
                encoder.append(self.act_fn)
            else:
                encoder.append(nn.Linear(self.enc_layers[ll][0], self.enc_layers[ll][1]))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for ll in range(len(self.dec_layers)):
            if ll < len(self.dec_layers)-1:
                decoder.append(nn.Linear(self.dec_layers[ll][0], self.dec_layers[ll][1]))
                decoder.append(self.act_fn)
            else:
                decoder.append(nn.Linear(self.dec_layers[ll][0], self.dec_layers[ll][1]))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        y = self.encoder(x)
        x_ae = self.decoder(y)

        if self.operation_mode == 'DLHDMD' or self.operation_mode == 'Global_DLDMD':
            initconds = torch.transpose(torch.squeeze(y[:, 0, :]), 0, 1)
            curgm, curgp = self.hankel_maker(y)
            kmat, phi_adv = self.hankel_dmd_real(initconds, curgm, curgp)
            y_adv = torch.reshape(torch.transpose(kmat @ phi_adv, 0, 1),
                                [self.bs, self.window-1, self.l_dim])
        elif self.operation_mode == 'DLDMD':
            y_adv, _ = self.dmd_real(y)

        x_adv = self.decoder(y_adv)
        return y, x_ae, x_adv, y_adv

    def dmd_real(self, y):
        """Computes the standard Dynamic Mode Decomposition of a multi-
        dimensional time series.

        Args:
            y (tensor array): Batched time series of data (batch, dims, time)

        Returns:
            y_adv (tensor array): Batched time series of predicted data (batch, dims, time)
            A (tensor array)
        """
        y = torch.transpose(y, 1, 2)
        y_m = y[:, :, :-1]
        y_p = y[:, :, 1:]

        U, S, Vh = torch.linalg.svd(y_m, full_matrices=False)
        Si = torch.diag_embed(1.0 / S)
        V = torch.adjoint(Vh)
        Uh = torch.adjoint(U)
        A = y_p @ V @ Si @ Uh
        y_adv = torch.transpose(self.phi_adv(y_m, A), 1, 2)
        return y_adv, A


    def hankel_maker(self, yin):
        ws = self.window
        nobs = self.num_observables
        gm = torch.zeros([nobs*self.l_dim, self.bs*(ws-1)],
                            dtype=self.dtype, device=self.device)
        gp = torch.zeros([nobs*self.l_dim, self.bs*(ws-1)],
                            dtype=self.dtype, device=self.device)
        for jj in range(self.l_dim):
            Yobserved = torch.squeeze(yin[:, :, jj])
            for ll in range(self.bs):
                tseries = Yobserved[ll, :]
                hankel_idx = np.array([np.arange(ii, ii+nobs) for ii in range(len(tseries) - nobs + 1)])
                hmat = tseries[hankel_idx].T
                gm[jj*nobs:(jj+1)*nobs, ll * (ws - 1):(ll + 1) * (ws - 1)] = hmat[:, :-1]
                gp[jj*nobs:(jj+1)*nobs, ll * (ws - 1):(ll + 1) * (ws - 1)] = hmat[:, 1:]
        return gm, gp

    def hankel_dmd_real(self, initconds, curgm, curgp):
        U, sig, Vh = torch.linalg.svd(curgm, full_matrices=False)
        sigr_inv = torch.diag_embed(1.0 / sig)
        V = torch.adjoint(Vh)
        Uh = torch.adjoint(U)
        A = curgp @ V @ sigr_inv @ Uh
        phi_adv = self.phi_adv(curgm, A)

        # Build reconstruction
        gm0 = curgm[:, ::(self.window-1)]
        Up, sigp, Vph = torch.linalg.svd(gm0, full_matrices=False)
        Vp = torch.adjoint(Vph)
        sigpr = sigp/(sigp**2.+self.tikhonov_param)
        pinv = Vp @ torch.diag_embed(sigpr) @ torch.adjoint(Up)
        kmat = initconds @ pinv

        return kmat, phi_adv

    def phi_adv(self, curgm, A):
        phi_adv = A @ curgm
        for jj in range(1, self.ysteps):
            phi_adv = A @ phi_adv
        return phi_adv


# ===========================================================================80
# Loss Function Implementations
# ===========================================================================80
class Hankel_DLDMD_Loss(nn.Module):
    def __init__(self, hyp):
        super(Hankel_DLDMD_Loss, self).__init__()

        self.device = hyp['device']
        self.cdtype = hyp['cdtype']
        self.a1 = hyp['a1']
        self.a2 = hyp['a2']
        self.a3 = hyp['a3']
        self.a4 = hyp['a4']
        self.num_observables = hyp['num_observables']
        self.n_ts = hyp['num_pred']
        self.window = self.n_ts - (self.num_observables - 1)
        self.ysteps = hyp['ysteps']
        self.num_loss_comps = 4

    def forward(self, x, model_results, model_params):
        y, x_ae, x_adv, y_adv = model_results
        loss_recon = self.a1 * torch.mean(torch.square(x - x_ae))
        loss_dmd = self.a2 * torch.mean(
            torch.square(y[:, self.ysteps:(self.window-1), :] -
                         y_adv[:, :-self.ysteps, :]))
        loss_pred = self.a3 * torch.mean(
            torch.square(x[:, self.ysteps:(self.window-1), :] -
                         x_adv[:, :-self.ysteps, :]))
        loss_reg = self.a4 * sum([torch.norm(torch.flatten(w)) for w in model_params])
        total_loss = loss_recon + loss_dmd + loss_pred +  loss_reg
        losses = [total_loss, loss_recon, loss_dmd, loss_pred]
        return losses

    def dmd_loss(self, y):
        """Computes the loss for the DMD component of the DLDMD loss function.

        Args:
            y (tensor array): Batched time series of latent data (batch, dims, time)

        Returns:
            loss (tensor array): DMD loss term (batch,)
        """
        y_m = torch.transpose(y, 2, 1)[:, :, :-1]
        y_p = torch.transpose(y, 2, 1)[:, :, 1:]
        _, _, Vh = torch.linalg.svd(y_m, full_matrices=False)
        VVh = torch.adjoint(Vh) @ Vh
        eye_mat = torch.eye(VVh.shape[-1], device=self.device,
                            dtype=self.cdtype, requires_grad=True)
        tmp = y_p.type(dtype=self.cdtype) @ (eye_mat - VVh)
        return torch.mean(torch.linalg.norm(tmp, ord='fro', dim=(-2, -1)))
