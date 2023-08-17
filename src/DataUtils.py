"""
    Various functions and utilities for creating and plotting data sets.

    Created by:
        Opal Issan

    Modified:
        17 Nov 2020 - Jay Lago
        NOTE: Caveat emptor! These functions are not efficient nor consistent
"""
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

plt.rcParams.update({ "text.usetex": True, "font.family": "serif" })
font = {'size': 64}
matplotlib.rc('font', **font)
fmt = 'pdf'
# ===========================================================================80
# Helper functions
# ===========================================================================80
### For Hankel stuff
def plot_all_logistics(x_adv, val_history, train_history, epoch, hyp):
    fig = plt.figure(figsize=(50, 30))
    skip = 16
    if hyp['operation_mode'] == 'DLHDMD':
        plot_3D(fig, 1, x_adv, skip, r"$\mbox{Enc.-Adv.-Dec.}$", "y")
        plot_2D(fig, 2, val_history[0:epoch+1, -1], val_history[0:epoch+1, -1], r"$\mbox{Number of Observables}$", r"$\bar{N}_{ob}$", 'num obs')
        plot_2D(fig, 3, val_history[0:epoch+1, 0], train_history[0:epoch+1, 0], r"$\mbox{Total Loss}$", r"$log_{10}(\mathcal{L}_{tot})$", 'tot')
        fig.tight_layout(pad=1.0)
    elif hyp['operation_mode'] == 'DLDMD' or hyp['operation_mode'] == 'Global_DLDMD':
        plot_3D(fig, 1, x_adv, skip, r"$\mbox{Enc.-Adv.-Dec.}$", "y")
        plot_2D(fig, 2, val_history[0:epoch+1, 0], train_history[0:epoch+1, 0], r"$\mbox{Total Loss}$", r"$log_{10}(\mathcal{L}_{tot})$", 'tot')
        fig.tight_layout(pad=3.0)

    plt.savefig(f"{hyp['model_path']}figs/total_diagnostic_{str(epoch)}."+fmt,bbox_inches='tight',format=fmt)
    plt.close()

    fig = plt.figure(figsize=(50, 30))
    skip = 16
    plot_2D(fig, 1, val_history[0:epoch+1, 1], train_history[0:epoch+1, 1], r"$\mbox{Recon. Loss}$", r"$log_{10}(\mathcal{L}_{recon})$", 'recon')
    plot_2D(fig, 2, val_history[0:epoch+1, 3], train_history[0:epoch+1, 3], r"$\mbox{Pred. Loss}$", r"$log_{10}(\mathcal{L}_{pred})$", 'pred')
    plot_2D(fig, 3, val_history[0:epoch+1, 2], train_history[0:epoch+1, 2], r"$\mbox{DMD Loss}$", r"$log_{10}(\mathcal{L}_{dmd})$", 'dmd')
    fig.tight_layout(pad=1.0)
    plt.savefig(f"{hyp['model_path']}figs/components_diagnostic_{str(epoch)}."+fmt,bbox_inches='tight',format=fmt)
    plt.close()

def plot_3D(fig, index, data, skip, title, ax_var):
    ax = fig.add_subplot(2, 3, index, projection='3d')
    # Validation batch
    for ii in np.arange(0, data.shape[0], skip):
        ax.plot3D(data[ii, :, 0], data[ii, :, 1], data[ii, :, 2], '-')
    ax.scatter(data[::skip, 0, 0], data[::skip, 0, 1], data[::skip, 0, 2])
    ax.grid()
    ax.set_xlabel(r"$" + ax_var + "_{1}$", labelpad=50)
    ax.set_ylabel(r"$" + ax_var + "_{2}$", labelpad=50)
    ax.set_zlabel(r"$" + ax_var + "_{3}$", labelpad=50)
    ax.set_title(title)

def plot_2D(fig, index, val_data, train_data, title, ylabel, pltlabel):
    lw = 3
    ax = fig.add_subplot(2, 3, index)
    #ax.plot(data, color='k', linewidth=lw, label=pltlabel)
    ax.plot(val_data, color='k', linewidth=lw, linestyle='solid')
    ax.plot(train_data, color='b', linewidth=lw, linestyle='dashed')
    ax.set_xlabel(r"$\mbox{Epoch}$")
    ax.set_title(title)
    if pltlabel == 'tot':
        ax.set_yticks([2, 1, 0, -1, -2])
    ax.grid()
    #ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    #ax.legend(loc="upper right")

### Other stuff
def quickplot_3d(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for ii in range(data.shape[0]):
        ax.scatter3D(data[ii, :, 0], data[ii, :, 1], data[ii, :, 2],
                     c=data[ii, :, 2], cmap='jet')
    plt.show()

# Diagnostic plots
def plot_diagnostic(x, x_ae, x_adv, history, hyp, epoch, dim3=False):
    nloss = history.shape[-1]
    nplots = 3 + nloss
    nrow, ncol = 3, int(np.ceil(nplots/3))
    fig = plt.figure(figsize=(20, 10), dpi=150)
    if dim3:
        ax1 = plt.subplot(nrow, ncol, 1, projection='3d')   # data
        ax2 = plt.subplot(nrow, ncol, 2, projection='3d')   # recon
        ax3 = plt.subplot(nrow, ncol, 3, projection='3d')   # prediction
        for ii in range(x.shape[0]):
            ax1.plot(x[ii, :, 0], x[ii, :, 1], x[ii, :, 2], 'k-', lw=0.25)
            ax2.plot(x_ae[ii, :, 0], x_ae[ii, :, 1], x_ae[ii, :, 2], 'r-', lw=0.25)
            ax3.plot(x_adv[ii, :, 0], x_adv[ii, :, 1], x_adv[ii, :, 2], 'b-', lw=0.25)
    else:
        ax1 = plt.subplot(nrow, ncol, 1)   # data
        ax2 = plt.subplot(nrow, ncol, 2)   # recon
        ax3 = plt.subplot(nrow, ncol, 3)   # prediction
        for ii in range(x.shape[0]):
            ax1.plot(x[ii, :, 0], x[ii, :, 1], 'k-', lw=0.25)
            ax2.plot(x_ae[ii, :, 0], x_ae[ii, :, 1], 'r-', lw=0.25)
            ax3.plot(x_adv[ii, :, 0], x_adv[ii, :, 1], 'b-', lw=0.25)
    ax1.set_title('test data')
    ax2.set_title('x_ae')
    ax3.set_title('x_adv')
    titles = ['total', 'recon', 'x_fwd', 'x_bwd', 'inv', 'reg']
    # titles = ['total', 'recon', 'x_pred', 'reg']
    for ii in range(nloss):
        ax = plt.subplot(nrow, ncol, ii+4)
        ax.plot(history[:, ii], '-', lw=3)
        ax.set_title(titles[ii])

    plt.savefig(f"{hyp['model_path']}figs/diagnostic_{str(epoch)}.png")
    plt.close()

def host_detach_np(tarrays):
    return [tmp.to(torch.device('cpu')).detach().numpy() for tmp in tarrays]

# ===========================================================================80
# Data functions
# ===========================================================================80
def data_scaler(data):
    return (data - np.mean(data)) / (np.max(data) - np.min(data))

def rk4(lhs, dt, function):
    k1 = dt * function(lhs)
    k2 = dt * function(lhs + k1 / 2.0)
    k3 = dt * function(lhs + k2 / 2.0)
    k4 = dt * function(lhs + k3)
    rhs = lhs + 1.0 / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    return rhs

def trajectory(func, ic, start, stop, dt):
    num_dims = np.size(ic)
    num_steps = int((stop - start)/dt)
    traj = np.zeros((num_steps, num_dims))
    traj[0, :] = ic
    for ii in range(1, num_steps):
        traj[ii, :] = rk4(traj[ii-1, :], dt, func)
    return traj

def pendulum(lhs):
    """ pendulum example:
    ODE =>
    dx1/dt = x2
    dx2/dt = -sin(x1)
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = -np.sin(lhs[0])
    return rhs

def duffing(lhs):
    """ Duffing oscillator:
    dx/dt = y
    dy/dt = x - x^3
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = lhs[0] - lhs[0]**3
    return rhs

def vanderpol(lhs, mu=1.5):
    """ Van der Pol example:
    ODE =>
    dx1/dt = x2
    dx2/dt = mu*(1-x1^2)*x2-x1
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = mu*(1 - lhs[0]**2) * lhs[1] - lhs[0]
    return rhs

def lorenz63(lhs, rho=28.0, sigma=10.0, beta=8./3.):
    """ Lorenz63 example:
    ODE =>
    dx1/dt = sigma*(x2 - x1)
    dx2/dt = x1*(rho - x3) - x2
    dx3/dt = x1*x2 - beta*x3
    """
    rhs = np.zeros(3)
    rhs[0] = sigma*(lhs[1] - lhs[0])
    rhs[1] = lhs[0]*(rho - lhs[2]) - lhs[1]
    rhs[2] = lhs[0]*lhs[1] - beta*lhs[2]
    return rhs

def lorenz96(lhs):
    F = 8.0
    rhs = -lhs + F + ( np.roll(lhs,-1) - np.roll(lhs,2) ) * np.roll(lhs,1)
    return rhs

def rossler(lhs, alpha=0.1, beta=0.1, gamma=14):
    """ Rossler system:
    ODE =>
    dx1/dt = -x2 - x3
    dx2/dt = x1 + alpha*x2
    dx3/dt = beta + x3*(x1 - gamma)
    """
    rhs = np.zeros(3)
    rhs[0] = -lhs[1] - lhs[2]
    rhs[1] = lhs[0] + alpha*lhs[1]
    rhs[2] = beta + lhs[2] * (lhs[0] - gamma)
    return rhs

def generate_rossler(x1min, x1max, x2min, x2max, x3min, x3max, num_ic=10000, dt=0.02, tf=100.0, seed=None):
    np.random.seed(seed=seed)
    num_dim = 3
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    ic_list = np.zeros((num_ic, num_dim))
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    icond3 = np.random.uniform(x3min, x3max, num_ic)
    data_mat = np.zeros((num_ic, 3, num_steps+1), dtype=np.float64)
    for ii in tqdm(range(num_ic),
                   desc='Generating Rossler system data...', ncols=100):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii], icond3[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, rossler)
    return np.transpose(data_mat, [0, 2, 1])

def generate_pendulum(x1min, x1max, x2min, x2max, num_ic=10000, dt=0.02, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    rand_x1 = np.random.uniform(x1min, x1max, 100 * num_ic)
    rand_x2 = np.random.uniform(x2min, x2max, 100 * num_ic)
    max_potential = 0.99
    potential = lambda x, y: (1 / 2) * y ** 2 - np.cos(x)
    iconds = np.asarray([[x, y] for x, y in zip(rand_x1, rand_x2)
                         if potential(x, y) <= max_potential])[:num_ic, :]
    data_mat = np.zeros((num_ic, 2, num_steps), dtype=np.float64)
    for ii in tqdm(range(num_ic),
                   desc='Generating pendulum system data...', ncols=100):
        data_mat[ii, :, 0] = np.array([iconds[ii, 0], iconds[ii, 1]], dtype=np.float64)
        for jj in range(num_steps-1):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, pendulum)

    return np.transpose(data_mat, [0, 2, 1])

def generate_duffing(x1min, x1max, x2min, x2max, num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    data_mat = np.zeros((num_ic, 2, num_steps), dtype=np.float64)
    for ii in tqdm(range(num_ic),
                   desc='Generating Duffing system data...', ncols=100):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
        for jj in range(num_steps-1):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, duffing)
    return np.transpose(data_mat, [0, 2, 1])

def generate_vanderpol(x1min, x1max, x2min, x2max, num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    data_mat = np.zeros((num_ic, 2, num_steps), dtype=np.float64)
    for ii in tqdm(range(num_ic),
                   desc='Generating Van der Pol system data...', ncols=100):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
        for jj in range(num_steps-1):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, vanderpol)
    return np.transpose(data_mat, [0, 2, 1])

def generate_lorenz63(x1min, x1max, x2min, x2max,
                      x3min, x3max, num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    icond3 = np.random.uniform(x3min, x3max, num_ic)
    data_mat = np.zeros((num_ic, 3, num_steps+1), dtype=np.float64)
    for ii in tqdm(range(num_ic),
                   desc='Generating Lorenz63 system data...', ncols=100):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii], icond3[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, lorenz63)
    return np.transpose(data_mat, [0, 2, 1])

def generate_lorenz96(xbounds, num_ic=15000, dim=8, dt=0.05, tf=20.0, seed=None):
    np.random.seed(seed=seed)
    nsteps = int(tf / dt)
    num_ic = int(num_ic)
    iconds = np.zeros((num_ic, dim), dtype=np.float64)
    for ll in range(dim):
        iconds[:, ll] = np.linspace(xbounds[ll, 0], xbounds[ll, 1], num_ic) + .05*2.*(np.random.rand(1)-.5)
    data_mat = np.zeros((num_ic, dim, nsteps + 1), dtype=np.float64)
    for ii in tqdm(range(num_ic), desc='Generating Lorenz96 system data...', ncols=100):
        data_mat[ii, :, 0] = iconds[ii, :]
        for jj in range(nsteps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, lorenz96)
    return np.transpose(data_mat, [0, 2, 1])


#===== Unused systems =====
def spider(lhs, mu=-0.05, lam=-1):
    """ example 1:
    ODE =>
    dx1/dt = mu*x1
    dx2/dt = lam*(x2-x1^2)

    By default: mu =-0.05, and lambda = -1.
    """
    rhs = np.zeros(2)
    rhs[0] = mu * lhs[0]
    rhs[1] = lam * (lhs[1] - (lhs[0]) ** 2.)
    return rhs

def fluid_3d(lhs, mu=0.1, omega=1, A=-0.1, lam=10):
    """fluid flow example:
    ODE =>
    dx1/dt = mu*x1 - omega*x2 + A*x1*x3
    dx2/dt = omega*x1 + mu*x2 + A*x2*x3
    dx3/dt = -lam(x3 - x1^2 - x2^2)
    """
    rhs = np.zeros(3)
    rhs[0] = mu * lhs[0] - omega * lhs[1] + A * lhs[0] * lhs[2]
    rhs[1] = omega * lhs[0] + mu * lhs[1] + A * lhs[1] * lhs[2]
    rhs[2] = -lam * (lhs[2] - lhs[0] ** 2 - lhs[1] ** 2)
    return rhs

def kdv(lhs, a1=0, c=3):
    """ planar kdv:
    dx1/dt = x2
    dx2/dt = a1 + c*x1 - 3*x2^2
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = a1 + c*lhs[0] - 3*lhs[0]**2
    return rhs

def duffing_driven(lhs, alpha=0.1, gamma=0.05, omega=1.1):
    """ Duffing oscillator:
    dx/dt = y
    dy/dt = x - x^3 - gamma*y + alpha*cos(omega*t)
    """
    rhs = np.zeros(3)
    rhs[0] = lhs[1]
    rhs[1] = lhs[0] - lhs[0]**3 - gamma*lhs[1] + alpha*np.cos(omega*lhs[2])
    rhs[2] = lhs[2]
    return rhs

def duffing_bollt(lhs, alpha=1.0, beta=-1.0, delta=0.5):
    """ Duffing oscillator:
    dx/dt = y
    dy/dt = -delta*y - x*(beta + alpha*x^2)
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = -delta*lhs[1] - lhs[0]*(beta + alpha*lhs[0]**2)
    return rhs

def generate_spider(x1min, x1max, x2min, x2max,
                    num_ic=1e4, dt=0.02, tf=1.0, seed=None, testing=False):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    if testing:
        icond1 = np.linspace(x1min, x1max, 10)
        icond2 = np.linspace(x2min, x2max, 2)
        xx, yy = np.meshgrid(icond1, icond2)
        data_mat = np.zeros((num_ic, 2, num_steps + 1), dtype=np.float64)
        ic = 0
        for x1 in range(2):
            for x2 in range(10):
                data_mat[ic, :, 0] = np.array([xx[x1, x2], yy[x1, x2]], dtype=np.float64)
                for jj in range(num_steps):
                    data_mat[ic, :, jj + 1] = rk4(data_mat[ic, :, jj], dt, spider)
                ic += 1
    else:
        icond1 = np.random.uniform(x1min, x1max, num_ic)
        icond2 = np.random.uniform(x2min, x2max, num_ic)
        data_mat = np.zeros((num_ic, 2, num_steps + 1), dtype=np.float64)
        for ii in range(num_ic):
            data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
            for jj in range(num_steps):
                data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, spider)
    return np.transpose(data_mat, [0, 2, 1])

def generate_pendulum_uniform(num_ic=10000, dt=0.02, tf=3.0, seed=None):
    num_steps = np.int(tf / dt)
    num_ic = np.int(num_ic)
    rand_x1 = np.random.uniform(-3.1, 0, 100*num_ic)
    rand_x2 = np.zeros((100*num_ic))
    max_potential = 0.99
    potential = lambda x, y: (1 / 2) * y ** 2 - np.cos(x)
    iconds = np.asarray([[x, y] for x, y in zip(rand_x1, rand_x2)
                         if potential(x, y) <= max_potential])[:num_ic, :]
    data_mat = np.zeros((num_ic, 2, num_steps + 1), dtype=np.float64)
    for ii in range(num_ic):
        data_mat[ii, :, 0] = np.array([iconds[ii, 0], iconds[ii, 1]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, pendulum)

    return np.transpose(data_mat, [0, 2, 1])

def generate_fluid_flow_slow(r_lower=0, r_upper=1.1, t_lower=0,
                             t_upper=2*np.pi, num_ic=1e4, dt=0.05, tf=6, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    n_ic_slow = int(num_ic)
    r = np.random.uniform(r_lower, r_upper, n_ic_slow)
    theta = np.random.uniform(t_lower, t_upper, n_ic_slow)
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    x3 = np.power(x1, 2) + np.power(x2, 2)
    iconds = np.zeros((n_ic_slow, 3))
    iconds[:n_ic_slow] = np.asarray([[x, y, z] for x, y, z in zip(x1, x2, x3)])
    data_mat = np.zeros((n_ic_slow, 3, num_steps + 1), dtype=np.float64)
    for ii in range(n_ic_slow):
        data_mat[ii, :, 0] = np.array([iconds[ii, 0], iconds[ii, 1], iconds[ii, 2]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, fluid_3d)
    return np.transpose(data_mat, [0, 2, 1])

def generate_fluid_flow_full(x1_lower=-1.1, x1_upper=1.1, x2_lower=-1.1,
                             x2_upper=1.1, x3_lower=0.0, x3_upper=2.43,
                               num_ic=1e4, dt=0.05, tf=6, seed=None):
    np.random.seed(seed=seed)
    num_steps = np.int(tf / dt)
    num_ic = np.int(num_ic)
    x1 = np.random.uniform(x1_lower, x1_upper, num_ic)
    x2 = np.random.uniform(x2_lower, x2_upper, num_ic)
    x3 = np.random.uniform(x3_lower, x3_upper, num_ic)
    iconds = np.zeros((num_ic, 3))
    iconds[:num_ic] = np.asarray([[x, y, z] for x, y, z in zip(x1, x2, x3)])
    data_mat = np.zeros((num_ic, 3, num_steps+1), dtype=np.float64)
    for ii in range(num_ic):
        data_mat[ii, :, 0] = np.array([iconds[ii, 0], iconds[ii, 1], iconds[ii, 2]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, fluid_3d)
    return np.transpose(data_mat, [0, 2, 1])

def generate_kdv(x1min, x1max, x2min, x2max, num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, 10*num_ic)
    icond2 = np.random.uniform(x2min, x2max, 10*num_ic)
    n_try = 10*num_ic
    data_mat = np.zeros((n_try, 2, num_steps+1), dtype=np.float64)
    for ii in range(n_try):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, kdv)
            # if (data_mat[ii, 0, jj+1] < x1min or data_mat[ii, 1, jj+1] > x1max
            #         or data_mat[ii, 1, jj+1] < x2min or data_mat[ii, 1, jj+1] > x2max):
            #     break
    accept = np.abs(data_mat[:, 0, -1]) < 3
    data_mat = data_mat[accept, :, :]
    accept = np.abs(data_mat[:, 1, -1]) < 3
    data_mat = data_mat[accept, :, :]
    data_mat = data_mat[:num_ic, :, :]
    return np.transpose(data_mat, [0, 2, 1])

def generate_duffing_driven(x1min, x1max, x2min, x2max, num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    data_mat = np.zeros((num_ic, 3, num_steps+1), dtype=np.float64)
    for ii in range(num_ic):
        data_mat[ii, :2, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, duffing_driven)
            data_mat[ii, 2, jj+1] = data_mat[ii, 2, jj] + dt
    return np.transpose(data_mat, [0, 2, 1])

def generate_duffing_bollt(x1min, x1max, x2min, x2max, num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    data_mat = np.zeros((num_ic, 2, num_steps+1), dtype=np.float64)
    for ii in range(num_ic):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, duffing_bollt)
    return np.transpose(data_mat, [0, 2, 1])

# ===========================================================================80
# Test program
# ===========================================================================80
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_spider = False
    test_pendulum = False
    test_fluid_flow_slow = False
    test_fluid_flow_full = False
    test_kdv = False
    test_duffing = False
    test_scales = False
    test_rossler = False

    if test_spider:
        data = generate_spider(x1min=-0.5, x1max=0.5, x2min=-0.5,
                                x2max=0.5, num_ic=20, dt=0.02, tf=10)
        plt.figure(1, figsize=(8, 8))
        for ii in range(data.shape[0]):
            plt.plot(data[ii, :, 0], data[ii, :, 1], '-')
        plt.xlabel("x1", fontsize=18)
        plt.ylabel("X2", fontsize=18)
        plt.title("Discrete dataset", fontsize=18)

    if test_pendulum:
        data = generate_pendulum(x1min=-3.1, x1max=3.1,
                                 x2min=-2, x2max=2, num_ic=20, dt=0.02, tf=20)
        plt.figure(2, figsize=(8, 8))
        for ii in range(data.shape[0]):
            plt.plot(data[ii, :, 0], data[ii, :, 1], '-')
        plt.xlabel("x1", fontsize=18)
        plt.ylabel("X2", fontsize=18)
        plt.title("Pendulum dataset", fontsize=18)

    if test_fluid_flow_slow:
        data = generate_fluid_flow_slow(r_lower=0, r_upper=1.1, t_lower=0,
                                        t_upper=2*np.pi, num_ic=20, dt=0.05, tf=10)
        fig = plt.figure(3, figsize=(8, 8))
        ax = plt.axes(projection='3d')
        for ii in range(data.shape[0]):
            ax.plot3D(data[ii, :, 0], data[ii, :, 1], data[ii, :, 2])
        ax.set_xlabel("$x_{1}$", fontsize=18)
        ax.set_ylabel("$x_{2}$", fontsize=18)
        ax.set_zlabel("$x_{3}$", fontsize=18)
        plt.title("Fluid Flow dataset", fontsize=20)

    if test_fluid_flow_full:
        data = generate_fluid_flow_full(x1_lower=-1.1, x1_upper=1.1, x2_lower=-1.1, x2_upper=1.1,
                                          x3_lower=0.0, x3_upper=2.43, num_ic=20, dt=0.05, tf=6)
        fig = plt.figure(4, figsize=(8, 8))
        ax = plt.axes(projection='3d')
        for ii in range(data.shape[0]):
            ax.plot3D(data[ii, :, 0], data[ii, :, 1], data[ii, :, 2])
        ax.set_xlabel("$x_{1}$", fontsize=18)
        ax.set_ylabel("$x_{2}$", fontsize=18)
        ax.set_zlabel("$x_{3}$", fontsize=18)
        plt.title("Fluid Flow dataset", fontsize=20)

    if test_kdv:
        data = generate_kdv(x1min=-2, x1max=2, x2min=-2, x2max=2, num_ic=1000, dt=0.01, tf=20)
        plt.figure(2, figsize=(8, 8))
        for ii in range(data.shape[0]):
            npts = np.sum(np.abs(data[ii, :, 0]) > 0)
            plt.plot(data[ii, :npts, 0], data[ii, :npts, 1], 'r-', lw=0.25)
        plt.xlabel("x1", fontsize=18)
        plt.ylabel("X2", fontsize=18)
        plt.title("KdV dataset", fontsize=18)

    if test_duffing:
        data = generate_duffing(x1min=-1, x1max=1, x2min=-1, x2max=1, num_ic=2, dt=0.05, tf=200)
        plt.figure(2, figsize=(8, 8))
        plt.plot(data[0, :, 0], data[0, :, 1], 'r-', lw=0.5)
        plt.plot(data[1, :, 0], data[1, :, 1], 'b-', lw=0.5)
        plt.xlabel("x1", fontsize=18)
        plt.ylabel("X2", fontsize=18)
        plt.title("Duffing oscillator", fontsize=18)

    if test_scales:
        t0, tf, dt = 0, 50, 0.01
        eps = 0.5
        num_steps = int(20/dt)
        t = np.linspace(0, tf, num_steps)
        num_ic = 100
        data = np.zeros((num_steps, 1))
        a = np.sin(t)
        x = np.cos(t) + np.cos(t/eps) + np.cos(t/eps**2) + np.cos(t/eps**3) + np.cos(t/eps**4)
        plt.plot(t, x, 'b-', lw=0.5)

    if test_rossler:
        data = generate_rossler(
            x1min=-5.0, x1max=5.0, x2min=-5.0, x2max=5.0,
            num_ic=100, dt=0.02, tf=100.0
        )
        fig = plt.figure(4, figsize=(8, 8))
        ax = plt.axes(projection='3d')
        for ii in range(data.shape[0]):
            ax.plot3D(data[ii, :, 0], data[ii, :, 1], data[ii, :, 2], lw=0.25)
        ax.set_xlabel("$x_{1}$", fontsize=18)
        ax.set_ylabel("$x_{2}$", fontsize=18)
        ax.set_zlabel("$x_{3}$", fontsize=18)
        plt.title("rossler", fontsize=20)

    plt.show()
    print("done")
