import numpy as np
import os
import sys
import datetime
import pickle
import math

from contextlib import contextmanager
from scipy.linalg import eigvals

from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW, SGD
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from dapper.mods.KS import Model as DapperKS

@contextmanager
def redirect_output(save_output=True, save_folder=".", filename="output.txt"):
    """
    Context manager to optionally redirect all stdout and stderr output to a file.
    
    Args:
        save_output (bool): If True (default), redirect output to the specified file.
                            If False, output will be printed to the console as normal.
        save_folder (str): The folder where the output file should be saved.
        filename (str): The name of the output file (default: "output.txt").
    """
    if save_output:
        # This block contains the original logic to save output to a file.
        os.makedirs(save_folder, exist_ok=True)  # Ensure the directory exists
        output_file = os.path.join(save_folder, filename)
        
        # Backup original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        file_exists = os.path.exists(output_file)
        
        with open(output_file, "a") as f:  # Open in append mode
            if file_exists:
                # Add a timestamp for new output sessions
                f.write(f"\n--- [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
            
            sys.stdout = f
            sys.stderr = f
            try:
                yield  # Execute the code block within the 'with' statement
            finally:
                # Restore original stdout and stderr after execution
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                print(f"Output appended to {output_file}") # Notify in the console
    else:
        # If save_output is False, just execute the code block without redirection.
        try:
            yield
        finally:
            # No redirection occurred, so no cleanup is needed.
            pass

def check_nan_in_model(model):
    for param in model.parameters():
        if torch.isnan(param).any():
            # print("NaN detected in model parameters")
            return True
    # print("No NaN in model parameters")
    return False

def get_mean_std(data_tensor):
    return torch.mean(data_tensor).item(), torch.std(data_tensor).item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

def rk4(func, u, t, dt):
    # Compute intermediary values for k
    k1 = func(dt, u)
    k2 = func(t + dt/2, u + dt/2*k1)
    k3 = func(t + dt/2, u + dt/2*k2)
    k4 = func(t + dt, u + dt*k3)
    # Compute updated values for u and t
    u_n = u + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return u_n

class VL20(nn.Module):
    """Modeled after dapper implementation"""

    def __init__(self, nX=36, F=10, G=10, alpha=1, gamma=1):
        super(VL20, self).__init__()
        self.fe = 0
        self.nX = nX
        self.F = F
        self.G = G
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, t, x):
        self.fe += 1
        out = torch.zeros_like(x)
        # print( torch.split(x, self.nX, -1))
        X, theta = x[:, 0, :], x[:, 1, :]

        # Velocities
        out[:, 0, :] = (torch.roll(X, 1, -1) - torch.roll(X, -2, -1)) * torch.roll(X, -1, -1)
        out[:, 0, :] -= self.gamma * X
        out[:, 0, :] += self.F - self.alpha * theta
        # Temperatures
        out[:, 1, :] = torch.roll(X, 1, -1) * torch.roll(theta, 2, -1) - \
                       torch.roll(X, -1, -1) * torch.roll(theta, -2, -1)
        out[:, 1, :] -= self.gamma * theta
        out[:, 1, :] += self.alpha * X + self.G
        return out


class L63(nn.Module):
    def __init__(self):
        super(L63, self).__init__()
        self.fe = 0

    @staticmethod
    def forward(t, x, sig=10.0, rho=28.0, beta=8.0 / 3):
        rvals = torch.zeros_like(x)
        rvals[:, 0] = sig * (x[:, 1] - x[:, 0])
        rvals[:, 1] = x[:, 0] * (rho - x[:, 2]) - x[:, 1]
        rvals[:, 2] = x[:, 0] * x[:, 1] - beta * x[:, 2]
        return rvals
    
class L96(nn.Module):
    def __init__(self):
        super(L96, self).__init__()

    def forward(t, x, F=8.):
        x_m2 = torch.roll(x, -2, -1)
        x_m1 = torch.roll(x, -1, -1)
        x_p1 = torch.roll(x, 1, -1)
        return (x_p1 - x_m2) * x_m1 - x + F * torch.ones_like(x)
    
class CircleODE(nn.Module):
    """
    Defines the ODE: dx/dt = omega*y, dy/dt = -omega*x.
    """
    def __init__(self):
        super(CircleODE, self).__init__()

    @staticmethod
    def forward(t, x, omega=1.0):
        # Input: t, x(batch,2), omega
        # Output: dx/dt(batch,2)
        rvals = torch.zeros_like(x)
        rvals[:, 0] = omega * x[:, 1]
        rvals[:, 1] = -omega * x[:, 0]
        return rvals

class DoubleWellODE(nn.Module):
    """
    Defines the ODE for a double-well potential: dx/dt = y, dy/dt = b*x - a*x^3.
    """
    def __init__(self):
        super(DoubleWellODE, self).__init__()

    @staticmethod
    def forward(t, x, a=1.0, b=1.0):
        # Input: t, x(batch,2), a, b
        # Output: dx/dt(batch,2)
        rvals = torch.zeros_like(x)
        rvals[:, 0] = x[:, 1]
        rvals[:, 1] = b * x[:, 0] - a * (x[:, 0]**3)
        return rvals

def gen_data(dataset, t, steps_test, steps_valid, args, v0=None, sigma_v=0,
             check_disk=True, steps_burn=1000, dt_iter=2, prefix="", test_only=False):
    """
    Generate training, validation, and test data for a given model.

    Parameters
    ----------
    dataset : str
        The name of the dataset, determines which model to use.
    t : array-like
        Time steps for integration.
    steps_test : int
        Number of steps in the test dataset.
    steps_valid : int
        Number of steps in the validation dataset.
    v0 : torch.Tensor, optional
        Initial state vector.
    sigma_v : float, default=0
        Standard deviation of Gaussian noise to add at each step.
    check_disk : bool, default=True
        If True, check if saved data exists on disk before generating.
    steps_burn : int, default=1000
        Number of initial steps to discard (burn-in).
    dt_iter : int, default=2
        Number of sub-iterations within each time step.
    prefix : str, default=""
        Prefix for saved file names.
    test_only : bool, default=False
        If True, generate only test data.

    Returns
    -------
    torch.Tensor
        Training sequence.
    torch.Tensor
        Validation sequence.
    torch.Tensor
        Test sequence.
    """

    dt = t[1] - t[0]  # Time step size for integration
    directory = f'data/{dataset}/'
    os.makedirs(directory, exist_ok=True)

    # Internal function for saving or loading data
    def __save_or_load_data(file_path, data=None):
        """Save data if provided, otherwise load data from the file path."""
        if data is not None:
            np.save(file_path, data.astype(np.float32))  # Save as float32
        else:
            return torch.tensor(np.load(file_path), dtype=torch.float32)  # Load as float32

    # Internal function to generate trajectory
    def __generate_trajectory(vf, steps, model, dt, dt_iter, sigma_v, burn_in=0):
        """Generate a trajectory for a given initial state, steps, model, and noise level."""
        trajectory = []
        for step in range(burn_in + steps):
            for _ in range(dt_iter):
                vf = rk4(model.forward, vf, step * dt, dt / dt_iter)
            vf = vf + sigma_v * torch.randn_like(vf, dtype=torch.float32, device=vf.device)  # Ensure float32
            trajectory.append(vf.unsqueeze(0))
        return torch.cat(trajectory).to(dtype=torch.float32)  # Ensure the output is float32

    # Define model and dimensions based on dataset type
    if dataset == "lorenz63":
        model, dim, default_v0 = L63, 3, torch.randn(1, 3, dtype=torch.float32)  # Ensure float32
    elif dataset == "lorenz96":
        model, dim, default_v0 = L96, args.ori_dim, torch.randn(1, args.ori_dim, dtype=torch.float32) + 5  # Ensure float32
    elif dataset == "circle":
        model, dim, default_v0 = CircleODE, 2, torch.tensor([[-1.0, 1.0]], dtype=torch.float32)  # Ensure float32
    elif dataset == "Hdoublewell":
        model, dim, default_v0 = DoubleWellODE, 2, torch.tensor([[-1.0, 1.0]], dtype=torch.float32)  # Ensure float32
    elif dataset == "ks":
        # Kuramoto-Sivashinsky model specifics
        model = etd_rk4_wrapper(device=None, dt=dt / dt_iter)
        # grid = 32 * np.pi * torch.linspace(0, 1, 128 + 1, dtype=torch.float32)[1:]  # Ensure float32
        # x0_Kassam = torch.cos(grid / 16) * (1 + torch.sin(grid / 16))
        # x0 = x0_Kassam.clone().unsqueeze(0)
        # # Single 150-step integration to stabilize x0
        # x0 = custom_int(x0, model, 150, dt, dt_iter)[-1:].to(dtype=torch.float32)  # Ensure float32
        # default_v0 = custom_int(x0, model, 10 ** 3, dt, dt_iter)[-1:].to(dtype=torch.float32)  # Ensure float32
        dapper_x0 = DapperKS(dt=dt.numpy() / dt_iter).x0
        default_v0 = torch.randn(1, 128, dtype=torch.float32) + dapper_x0
    else:
        raise ValueError('Dataset not implemented')

    v0 = v0 if v0 is not None else default_v0
    prefix_path = f'{directory}/{prefix}true_v_withnoise_{dt:.3f}step.npy'
    test_file_path = f'{directory}/{prefix}test_{steps_valid}_{steps_test}_v_{dt:.3f}step.npy'
    
    # Training data generation
    with torch.no_grad():
        if not test_only:
            if check_disk and os.path.exists(prefix_path):
                v_traj = __save_or_load_data(prefix_path)
            else:
                if dataset == "ks":
                    # Custom integration for KS model with modified dt_iter
                    v_traj = custom_int(v0, model, len(t), dt, dt_iter).unsqueeze(1).to(dtype=torch.float32)  # Ensure float32
                else:
                    # General case for Lorenz models
                    v_traj = __generate_trajectory(v0, len(t), model, dt, dt_iter, sigma_v)
                if check_disk:
                    __save_or_load_data(prefix_path, v_traj.numpy())

        # Determine final state for testing
        vf = v_traj[-1].view(1, -1) if not test_only else v0

        # Testing data generation
        if check_disk and os.path.exists(test_file_path):
            v_traj_test = __save_or_load_data(test_file_path)
        else:
            if dataset == "ks":
                # Custom integration for KS model's testing data with modified dt_iter
                v_traj_test = custom_int(vf, model, steps_burn + steps_test + steps_valid, dt, dt_iter).unsqueeze(1).to(dtype=torch.float32)  # Ensure float32
            else:
                v_traj_test = __generate_trajectory(vf, steps_burn + steps_test + steps_valid, model, dt, dt_iter, sigma_v)
            if check_disk:
                __save_or_load_data(test_file_path, v_traj_test.numpy())

    if test_only:
        return v_traj_test[steps_burn:steps_valid + steps_burn], v_traj_test[steps_burn + steps_valid:]
    
    return v_traj, v_traj_test[steps_burn:steps_valid + steps_burn], v_traj_test[steps_burn + steps_valid:]





def etd_rk4_wrapper(device=None, dt=0.5, DL=32, Nx=128):
    """ Returns an ETD-RK4 integrator for the KS equation. Currently very specific, need
    to adjust this to fit into the same framework as the ODE integrators

    Directly ported from https://github.com/nansencenter/DAPPER/blob/master/dapper/mods/KS/core.py
    which is adapted from kursiv.m of Kassam and Trefethen, 2002, doi.org/10.1137/S1064827502410633.
    """
    if device is None:
        device = torch.device('cpu')
    kk = np.append(np.arange(0, Nx / 2), 0) * 2 / DL  # wave nums for rfft
    h = dt

    # Operators
    L = kk ** 2 - kk ** 4  # Linear operator for K-S eqn: F[ - u_xx - u_xxxx]

    # Precompute ETDRK4 scalar quantities
    E = torch.Tensor(np.exp(h * L)).unsqueeze(0).to(device)  # Integrating factor, eval at dt
    E2 = torch.Tensor(np.exp(h * L / 2)).unsqueeze(0).to(device)  # Integrating factor, eval at dt/2

    # Roots of unity are used to discretize a circular countour...
    nRoots = 16
    roots = np.exp(1j * np.pi * (0.5 + np.arange(nRoots)) / nRoots)
    # ... the associated integral then reduces to the mean,
    # g(CL).mean(axis=-1) ~= g(L), whose computation is more stable.
    CL = h * L[:, None] + roots  # Contour for (each element of) L
    # E * exact_integral of integrating factor:
    Q = torch.Tensor(h * ((np.exp(CL / 2) - 1) / CL).mean(axis=-1).real).unsqueeze(0).to(device)
    # RK4 coefficients (modified by Cox-Matthews):
    f1 = torch.Tensor(h * ((-4 - CL + np.exp(CL) * (4 - 3 * CL + CL ** 2)) / CL ** 3).mean(axis=-1).real).unsqueeze(
        0).to(device)
    f2 = torch.Tensor(h * ((2 + CL + np.exp(CL) * (-2 + CL)) / CL ** 3).mean(axis=-1).real).unsqueeze(0).to(device)
    f3 = torch.Tensor(h * ((-4 - 3 * CL - CL ** 2 + np.exp(CL) * (4 - CL)) / CL ** 3).mean(axis=-1).real).unsqueeze(
        0).to(device)

    D = 1j * torch.Tensor(kk).to(device)  # Differentiation to compute:  F[ u_x ]

    def NL(v, verb=False):
        return -.5 * D * torch.fft.rfft(torch.fft.irfft(v, dim=-1) ** 2, dim=-1)

    def inner(v, t, dt, verb=False):
        v = torch.fft.rfft(v, dim=-1)
        N1 = NL(v, verb)
        v1 = E2 * v + Q * N1

        N2a = NL(v1)
        v2a = E2 * v + Q * N2a

        N2b = NL(v2a)
        v2b = E2 * v1 + Q * (2 * N2b - N1)

        N3 = NL(v2b)
        v = E * v + N1 * f1 + 2 * (N2a + N2b) * f2 + N3 * f3
        return torch.fft.irfft(v, dim=-1)

    return inner


# This basically is just a hack for KS training
def custom_int(x0, int_function, steps, dt=0.25, dt_iter=1):
    out = [x0]
    x = x0
    for _ in range(steps):
        for _ in range(dt_iter):  # Execute the integration function `dt_iter` times per step
            x = int_function(x, None, dt / dt_iter)
        out.append(x)
    return torch.cat(out, 0)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class TimeStack:
    def __call__(self, batch):
        return torch.cat(batch, dim=1)


class ChunkedTimeseries(Dataset):
    """Chunked timeseries dataset."""

    def __init__(self, seq, chunk_size=40, overlap=.25, transform=None):
        """
        Args:
            seq (torch.Tensor): Tensor containing time series
            chunk_size (int): size of chunks to produce
            overlap (float):
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.seq = seq
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.n = seq.shape[0]
        self.starts = np.array([i * chunk_size for i in range(self.n // chunk_size)])
        self.transform = transform

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.overlap <= 0:
            add_on = 0
        else:
            add_on = np.random.randint(int(self.overlap * self.chunk_size))
        start = min(self.starts[idx] + add_on,
                    self.n - self.chunk_size)
        sample = self.seq[start:start + self.chunk_size]
        if self.transform:
            sample = self.transform(sample)
        return sample


def mystery_operator(H_size, device, seed=None):
    """ Creates a random projection matrix for
    random lossy feature generation. """
    if seed is not None:
        torch.manual_seed(seed)
    proj = torch.randn(*H_size).to(device)
    def inner(x):
        return x @ proj
    return inner, proj

def partial_obs_operator(ori_dim, obs_inds, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    proj = torch.zeros(ori_dim, len(obs_inds), device=device)
    for i, obs_ind in enumerate(obs_inds):
        proj[obs_ind, i] = 1
    def inner(x):
        return x @ proj
    return inner, proj

def get_dataloader(args, x0=None, test_only=False):
    if args.dataset.startswith('linear'):
        # A = torch.randn(args.ori_dim, args.ori_dim, dtype=torch.float32)
        # eigs = torch.linalg.eigvals(A)
        # max_abs_eig = torch.max(torch.abs(eigs))
        # scaling_factor = 1.1 / max_abs_eig
        # A = A * scaling_factor
        
        if args.ori_dim == 2:
            dt_A = torch.tensor(0.5)
            A = torch.tensor([[torch.cos(dt_A), torch.sin(dt_A)], [-torch.sin(dt_A), torch.cos(dt_A)]])
        else:
            # A = torch.zeros(args.ori_dim, args.ori_dim)
            # for i in range(0, args.ori_dim, 2):
            #     # Generate a random rotation angle, ensuring it's not a multiple of pi.
            #     # theta (float): The rotation angle.
            #     theta = torch.tensor(0.2 + 0.8 * (args.ori_dim - i) / args.ori_dim)

            #     # Create a 2x2 rotation matrix (which is orthogonal)
            #     # R (torch.Tensor): A 2x2 orthogonal rotation matrix.
            #     c, s = torch.cos(theta), torch.sin(theta)
            #     R = torch.tensor([[c, -s],
            #                     [s,  c]])

            #     # Place the 2x2 block on the diagonal of A
            #     A[i:i+2, i:i+2] = R
            
            J = torch.randn(args.ori_dim, args.ori_dim)
            A, _ = torch.linalg.qr(J)
            
        eigenvalues = torch.linalg.eigvals(A)
        print("\nEigenvalues of A:")
        print(eigenvalues)
        
        
        H = torch.zeros(args.obs_dim, args.ori_dim)
        for i in range(args.obs_dim):
            H[i, 2*i] = 1
            
        train_data = LinearSystemDataset(
            d=args.ori_dim, 
            d_obs=args.obs_dim, 
            num_samples=args.train_traj_num, 
            m=None, C=None, A=A, H=H,
            sigma_v=args.sigma_v, 
            data_name=f"training_{args.seed}",
            sigma_y=args.sigma_y,
            load_existing=False,
            seed=None
        )
        test_data = LinearSystemDataset(
            d=args.ori_dim, 
            d_obs=args.obs_dim, 
            num_samples=args.test_traj_num, 
            m=None, C=None, A=A, H=H,
            sigma_v=args.sigma_v, 
            sigma_y=args.sigma_y,
            data_name=f"test_{args.seed}",
            load_existing=False,
            seed=None
        )
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_loader_workers)
        print("Train loader length:", len(train_loader))
        # test data
        test_loader = DataLoader(test_data, batch_size=args.test_batch_size,
                                    shuffle=False, num_workers=args.num_loader_workers)
        print("Dataloader generated.")

        if test_only:
            return test_loader

        return train_loader, test_loader
    else:
        t = torch.arange(0, args.train_steps * args.train_traj_num * args.dt, args.dt)
        gen_trajs = gen_data(args.dataset, t, v0=x0, sigma_v=args.sigma_v,
                                                        steps_test=args.test_steps * args.test_traj_num,
                                                        steps_valid=args.valid_steps,
                                                        args=args,
                                                        check_disk=args.new_data,
                                                        steps_burn=args.burn_steps,
                                                        dt_iter=args.dt_iter,
                                                        prefix=f"{args.sigma_v}_{args.train_steps}_{args.train_traj_num}_{args.trail}",
                                                        test_only=test_only
                                                        )
        if test_only:
            _, true_v_test = gen_trajs
        else:
            true_v, _, true_v_test = gen_trajs

        # training data
        if not test_only:
            train_data = ChunkedTimeseries(true_v, args.train_steps, args.overlap_rate)
            train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_loader_workers, collate_fn=TimeStack())
            print("Train loader length:", len(train_loader))


        # test data
        test_data = ChunkedTimeseries(true_v_test, args.test_steps, 0)
        test_loader = DataLoader(test_data, batch_size=args.test_batch_size,
                                    shuffle=False, num_workers=args.num_loader_workers, collate_fn=TimeStack())

        print("Dataloader generated.")

        if test_only:
            return test_loader

        return train_loader, test_loader

def create_optimizer(model, args, apply_multiplier=False):
    # Check if model is a list
    if isinstance(model, list):
        # Aggregate all parameters from the list of models
        parameters = []
        for m in model:
            parameters += list(m.parameters())
    else:
        # If model is not a list, use its parameters directly
        parameters = model.parameters()

    # Create the optimizer based on whether it's SGD or AdamW
    if args.SGD:
        return SGD(parameters, lr=args.learning_rate,
                momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        if apply_multiplier:
            return AdamW(parameters, lr=args.learning_rate * 4,
                        weight_decay=args.weight_decay)
        return AdamW(parameters, lr=args.learning_rate,
                    weight_decay=args.weight_decay)


# def combined_lr_scheduler(args):
#     def lr_lambda(epoch):
#         if epoch < args.warm_up_epochs:
#             # Warm-up phase
#             return args.warm_up_rate ** epoch
#         else:
#             # After warm-up, apply step decay
#             decay_epochs = [int(e) for e in args.lr_decay_epochs.split(',')]
#             decay_factor = sum([epoch >= e for e in decay_epochs])
#             return args.warm_up_rate ** args.warm_up_epochs * (args.lr_decay_rate ** decay_factor)

#     return lr_lambda

def combined_lr_scheduler(args):
    def lr_lambda(epoch):
        if epoch < args.warm_up_epochs:
            # Exponential warm-up normalized to reach 1.0 at the end
            base = args.warm_up_rate
            normalized = base ** (epoch + 1) / base ** args.warm_up_epochs
            return normalized
        else:
            # After warm-up: step decay
            decay_epochs = [int(e) for e in args.lr_decay_epochs.split(',')]
            decay_factor = sum([epoch >= e for e in decay_epochs])
            return args.lr_decay_rate ** decay_factor
    return lr_lambda


def setup_optimizer_and_scheduler(model, args, apply_multiplier=False):
    optimizer = create_optimizer(model, args, apply_multiplier=apply_multiplier)
    lr_lambda = combined_lr_scheduler(args)
    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def projection_fun(V, H):
    if V.dim() == 2:
        return V @ H.T
    elif V.dim() == 3:
        B = V.shape[0]
        H = H.unsqueeze(0).expand(B, -1, -1)
        return torch.bmm(V, H.transpose(1, 2))
    else:
        raise ValueError("Input tensor dimension should be 2 or 3")


def save_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth"):
    """
    Save the model, optimizer, and scheduler state dictionaries to a checkpoint file.
    Handles single models or lists of models, with support for DataParallel.
    """
    # Check if model is a list
    if isinstance(model, list):
        # Save state dicts for each model in the list
        model_state_dicts = [m.module.state_dict() if isinstance(m, torch.nn.DataParallel) else m.state_dict() for m in model]
    else:
        # Save state dict for a single model, handle DataParallel
        model_state_dicts = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

    # Create the checkpoint dictionary
    checkpoint = {
        "model_state_dict": model_state_dicts,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None
    }

    # Save the checkpoint
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer=None, scheduler=None, filename="checkpoint.pth", use_data_parallel=False):
    """
    Load the model, optimizer, and scheduler state dictionaries from a checkpoint file.
    Handles single models or lists of models, with support for DataParallel.
    """
    if not os.path.exists(filename):
        print(f"Checkpoint file {filename} does not exist.")
        return model, optimizer, scheduler

    # Load the checkpoint
    checkpoint = torch.load(filename, weights_only=True)

    # Check if model is a list and load the state dicts accordingly
    if isinstance(model, list):
        # Ensure the model list length matches the state dicts list length
        if len(model) != len(checkpoint["model_state_dict"]):
            raise ValueError("The number of models in the checkpoint does not match the number of provided models.")

        for i, m in enumerate(model):
            state_dict = checkpoint["model_state_dict"][i]
            # Handle DataParallel
            if use_data_parallel:
                state_dict = {"module." + k if not k.startswith("module.") else k: v for k, v in state_dict.items()}
            else:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            m.load_state_dict(state_dict)
    else:
        # Handle single model
        state_dict = checkpoint["model_state_dict"]
        if use_data_parallel:
            state_dict = {"module." + k if not k.startswith("module.") else k: v for k, v in state_dict.items()}
        else:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # Load optimizer state dict if provided
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state dict if provided
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Checkpoint loaded from {filename}")

    return model, optimizer, scheduler

def batch_covariance(x, y=None):
    B, N, D = x.shape
    mean_x = x.mean(dim=1, keepdim=True)
    x_centered = x - mean_x

    if y is None:
        cov_matrix = torch.bmm(x_centered.transpose(1, 2), x_centered) / N
    else:
        B_y, N_y, d = y.shape
        assert B == B_y and N == N_y
        mean_y = y.mean(dim=1, keepdim=True)
        y_centered = y - mean_y
        cov_matrix = torch.bmm(x_centered.transpose(1, 2), y_centered) / N
    
    return cov_matrix

def center(E, axis=1, rescale=False):
    x = torch.mean(E, dim=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= torch.sqrt(torch.tensor(N / (N - 1))).to(E.device)
    
    # x = x.squeeze(axis=axis)

    return X, x

def mean0(E, axis=1, rescale=True):
    return center(E, axis=axis, rescale=rescale)[0]

def mrdiv(b, A):
    """b/A."""
    if A.dim() == 3:
        return torch.linalg.solve(A.transpose(1,2), b.transpose(1,2)).transpose(1,2)
    
    return torch.linalg.solve(A.T, b.T).T

def svd0(A):
    U, S, V = torch.svd(A, some=False)
    return U, S, V

def pad0(x, N):
    """Pad `x` with zeros so that `x.shape[1]==N`."""
    out = torch.zeros(x.shape[0], N, device=x.device)
    out[:, :x.shape[1]] = x
    return out

def check_infl(infl):
    do_infl = False  

    if isinstance(infl, torch.Tensor):
        if infl.dim() == 3:
            do_infl = True
        elif infl.dim() == 1:
            if infl.item() != 1.0 and infl.item() != "-N":
                do_infl = True
            else:
                do_infl = False
        else:
            raise ValueError("Tensor dim is not supported, should be either 1 or 3.")
    
    elif isinstance(infl, (int, float)):
        if infl != 1.0 and infl != "-N":
            do_infl = True
        else:
            do_infl = False
    
    else:
        raise TypeError("infl must be a Tensor, int, or float.")

    return do_infl

def post_process(E, infl):
    """Inflate, Rotate.

    To avoid recomputing/recombining anomalies,
    this should have been inside `EnKF_analysis`

    But it is kept as a separate function

    - for readability;
    - to avoid inflating/rotationg smoothed states (for the `EnKS`).
    """

    if check_infl(infl):
        A, mu = center(E)
        # B, N, _ = E.shape
        # T = torch.eye(N).unsqueeze(0).expand(B, -1, -1).to(E.device)
        # T = infl * T
        # E = mu + torch.bmm(T, A)
        E = mu + infl * A
    return E

class LinearSystemDataset(Dataset):
    """
    PyTorch Dataset for generating random or fixed linear system parameters.
    
    Each sample contains:
    - m: vector of length d
    - C: d x d symmetric matrix (Cholesky factor)
    - A: d x d matrix with eigenvalues having absolute values < 1.1
    - H: d_obs x d matrix
    - sigma_v: positive scalar (observation noise)
    - sigma_y: positive scalar (process noise)
    """
    
    def __init__(self, d, d_obs, num_samples, 
                 m=None, C=None, A=None, H=None,
                 sigma_v=None, sigma_y=None, 
                 data_name="default", load_existing=True, seed=None):
        """
        Initialize the dataset.
        
        Args:
            d (int): Dimension of the state vector.
            d_obs (int): Dimension of the observation vector.
            num_samples (int): Number of samples to generate.
            m (torch.Tensor or None): Fixed value for m, or None for random generation.
            C (torch.Tensor or None): Fixed value for C, or None for random generation.
            A (torch.Tensor or None): Fixed value for A, or None for random generation.
            H (torch.Tensor or None): Fixed value for H, or None for random generation.
            sigma_v (float or None): Fixed value for sigma_v, or None for random generation.
            sigma_y (float or None): Fixed value for sigma_y, or None for random generation.
            data_name (str): Name for saving/loading data files.
            load_existing (bool): Whether to load existing data if available.
            seed (int or None): Random seed for reproducibility.
        """
        self.d = d
        self.d_obs = d_obs
        self.num_samples = num_samples
        
        # Store fixed parameters if provided
        self.m_fixed = m
        self.C_fixed = C
        self.A_fixed = A
        self.H_fixed = H
        self.sigma_v_fixed = sigma_v
        self.sigma_y_fixed = sigma_y
        
        self.data_name = data_name
        self.seed = seed
        
        # Create data directory if it doesn't exist
        os.makedirs("data/linear", exist_ok=True)
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # File path for saving/loading data
        self.data_file = f"data/linear/{data_name}_d{d}_dobs{d_obs}_n{num_samples}.pkl"
        
        # Try to load existing data if requested
        if load_existing and os.path.exists(self.data_file):
            print(f"Loading existing data from {self.data_file}")
            self._load_data()
        else:
            print(f"Generating new data and saving to {self.data_file}")
            self._generate_data()
            self._save_data()
            
    def _generate_stable_matrix_A(self):
        """Generates a random matrix A with eigenvalues |eig| < 1."""
        A = torch.randn(self.d, self.d, dtype=torch.float32)
        eigs = torch.linalg.eigvals(A)
        max_abs_eig = torch.max(torch.abs(eigs))
        
        # Scale matrix if the largest eigenvalue is too large
        if max_abs_eig >= 1.0:
            # Scale to be slightly less than 1.0 for stability
            scaling_factor = torch.rand(1, dtype=torch.float32) * 0.98 / max_abs_eig
            A = A * scaling_factor
        
        return A

    def _generate_symmetric_matrix_C(self):
        """Generates a random symmetric positive definite matrix C (as Cholesky factor)."""
        temp = torch.randn(self.d, self.d, dtype=torch.float32)
        # Create a symmetric positive semi-definite matrix
        C_full = temp @ temp.T
        # Add a small identity matrix to ensure it is positive definite
        C_full += 1 * torch.eye(self.d, dtype=torch.float32)
        # Return the Cholesky decomposition
        return torch.linalg.cholesky(C_full)
    
    def _generate_data(self):
        """Generate all parameters for the dataset, using fixed values if provided."""
        self.data = []
        
        for i in range(self.num_samples):
            # Use fixed tensors if provided, otherwise generate randomly
            m = self.m_fixed if self.m_fixed is not None else torch.randn(self.d, dtype=torch.float32)
            C = self.C_fixed if self.C_fixed is not None else self._generate_symmetric_matrix_C()
            A = self.A_fixed if self.A_fixed is not None else self._generate_stable_matrix_A()
            H = self.H_fixed if self.H_fixed is not None else torch.randn(self.d_obs, self.d, dtype=torch.float32)
            
            if self.sigma_v_fixed is not None:
                sigma_v = torch.tensor([self.sigma_v_fixed], dtype=torch.float32)
            else:
                sigma_v = torch.rand(1, dtype=torch.float32) * 0.3 # Uniformly sample from [0, 0.3)
                
            if self.sigma_y_fixed is not None:
                sigma_y = torch.tensor([self.sigma_y_fixed], dtype=torch.float32)
            else:
                sigma_y = 0.1 + torch.rand(1, dtype=torch.float32) * 0.9 # Uniformly sample from [0.1, 1.0)

            sample = {
                'm': m, 'C': C, 'A': A, 'H': H,
                'sigma_v': sigma_v, 'sigma_y': sigma_y
            }
            self.data.append(sample)

    def _save_data(self):
        """Save generated data and metadata to a file."""
        save_dict = {
            'data': self.data,
            'metadata': {
                'd': self.d, 'd_obs': self.d_obs, 'num_samples': self.num_samples,
                'm_fixed': self.m_fixed, 'C_fixed': self.C_fixed,
                'A_fixed': self.A_fixed, 'H_fixed': self.H_fixed,
                'sigma_v_fixed': self.sigma_v_fixed, 'sigma_y_fixed': self.sigma_y_fixed,
                'seed': self.seed
            }
        }
        with open(self.data_file, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Data saved to {self.data_file}")
    
    def _load_data(self):
        """Load data from a file and verify metadata."""
        try:
            with open(self.data_file, 'rb') as f:
                loaded_dict = pickle.load(f)
            
            self.data = loaded_dict['data']
            metadata = loaded_dict['metadata']
            
            # Verify that loaded data dimensions match current parameters
            if (metadata['d'] != self.d or 
                metadata['d_obs'] != self.d_obs or 
                metadata['num_samples'] != self.num_samples):
                print("Warning: Loaded data dimensions do not match current parameters.")
                print(f"  Loaded:  d={metadata['d']}, d_obs={metadata['d_obs']}, n={metadata['num_samples']}")
                print(f"  Current: d={self.d}, d_obs={self.d_obs}, n={self.num_samples}")
            
            print(f"Successfully loaded {len(self.data)} samples.")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Generating new data instead...")
            self._generate_data()
            self._save_data()
            
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset by index."""
        if not 0 <= idx < len(self.data):
            raise IndexError(f"Index {idx} is out of range for a dataset of size {len(self.data)}")
        return self.data[idx]

    def get_sample_info(self, idx=0):
        """Print detailed information about a specific sample for debugging."""
        if not 0 <= idx < len(self.data):
            print(f"Index {idx} out of range.")
            return
            
        sample = self.data[idx]
        A = sample['A']
        C = sample['C']
        C_full = C @ C.T # Reconstruct full covariance matrix from Cholesky factor
        eigs_A = torch.linalg.eigvals(A)
        max_abs_eig = torch.max(torch.abs(eigs_A))
        
        print(f"--- Sample {idx} Information ---")
        print(f"  m shape: {sample['m'].shape}")
        print(f"  C (Cholesky) shape: {C.shape}")
        print(f"  Is C_full symmetric? {torch.allclose(C_full, C_full.T)}")
        print(f"  A shape: {A.shape}")
        print(f"  Max |eigenvalue of A|: {max_abs_eig.item():.4f}")
        print(f"  H shape: {sample['H'].shape}")
        print(f"  sigma_v: {sample['sigma_v'].item():.4f}")
        print(f"  sigma_y: {sample['sigma_y'].item():.4f}")
        print("--------------------------")