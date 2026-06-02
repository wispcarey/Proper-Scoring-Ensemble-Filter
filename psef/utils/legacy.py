import numpy as np
import os
import sys
import datetime
import pickle
import math
import importlib

from contextlib import contextmanager
from scipy.linalg import eigvals

from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW, SGD
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

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
            # No redirection occurred.
            pass


def should_redirect_output(args):
    """
    Return whether output should be redirected to file.

    If args.normal_output exists and is True, print directly to console.
    """
    return not bool(getattr(args, "normal_output", False))

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
    
class DoublingMap1D(nn.Module):
    """Discrete map: x_{k+1} = (2 * x_k) mod 1."""

    def __init__(self):
        super(DoublingMap1D, self).__init__()

    @staticmethod
    def forward(t, x=None):
        # Support both forward(x) and forward(t, x) call styles.
        if x is None:
            x = t
        return torch.remainder(2.0 * x, 1.0)

def gen_data(dataset, t, steps_test, steps_valid, args, v0=None, sigma_v=0,
             check_disk=True, steps_burn=1000, dt_iter=2, prefix="", test_only=False):
    
    dt = t[1] - t[0]
    directory = f'data/{dataset}/'
    os.makedirs(directory, exist_ok=True)

    def __save_or_load_data(file_path, data=None):
        if data is not None:
            np.save(file_path, data.astype(np.float32))
        else:
            return torch.tensor(np.load(file_path), dtype=torch.float32)

    def __load_valid_cached_data(file_path, tag):
        if not (check_disk and os.path.exists(file_path)):
            return None
        cached = __save_or_load_data(file_path)
        if torch.isfinite(cached).all():
            return cached
        print(f"[WARN] Found non-finite values in cached {tag}: {file_path}. Regenerating this file.")
        return None

    def __generate_trajectory(vf, steps, model, dt, dt_iter, sigma_v, burn_in=0):
        trajectory = []
        sigma_v_eff = 0.0 if sigma_v is None else sigma_v
        for step in range(burn_in + steps):
            if dataset == "doubling1d":
                vf = model.forward(vf)
                vf = torch.remainder(
                    vf + sigma_v_eff * torch.randn_like(vf, dtype=torch.float32, device=vf.device),
                    1.0,
                )
            else:
                for _ in range(dt_iter):
                    vf = rk4(model.forward, vf, step * dt, dt / dt_iter)
                vf = vf + sigma_v_eff * torch.randn_like(vf, dtype=torch.float32, device=vf.device)
            trajectory.append(vf.unsqueeze(0))
        return torch.cat(trajectory).to(dtype=torch.float32)

    # --- Model Setup ---
    if dataset == "lorenz63":
        model, dim, default_v0 = L63, 3, torch.randn(1, 3)
    elif dataset == "lorenz96":
        model, dim, default_v0 = L96, args.ori_dim, torch.randn(1, args.ori_dim) + 5
    elif dataset == "doubling1d":
        model, dim, default_v0 = DoublingMap1D, 1, torch.rand(1, 1)
    else:
        raise ValueError('Dataset not implemented')

    v0 = v0 if v0 is not None else default_v0
    v0 = v0.to(dtype=torch.float32)

    prefix_path = f'{directory}/{prefix}true_v_withnoise_{dt:.3f}step.npy'
    test_file_path = f'{directory}/{prefix}test_{steps_valid}_{steps_test}_v_{dt:.3f}step.npy'
    
    # --- Training Data Generation ---
    with torch.no_grad():
        if not test_only:
            v_traj = __load_valid_cached_data(prefix_path, "training trajectory")
            if v_traj is None:
                v_traj = __generate_trajectory(v0, len(t), model, dt, dt_iter, sigma_v)
                v_traj = v_traj.to(dtype=torch.float32)
                if check_disk:
                    __save_or_load_data(prefix_path, v_traj.numpy())

            vf = v_traj[-1] # (Batch, Dim)
            vf = vf.view(1, -1)
        else:
            vf = v0

        # --- Test Data Generation ---
        v_traj_test = __load_valid_cached_data(test_file_path, "test trajectory")
        if v_traj_test is None:
            total_steps = steps_burn + steps_test + steps_valid
            v_traj_test = __generate_trajectory(vf, total_steps, model, dt, dt_iter, sigma_v)
            v_traj_test = v_traj_test.to(dtype=torch.float32)
            if check_disk:
                __save_or_load_data(test_file_path, v_traj_test.numpy())

    if test_only:
        return v_traj_test[steps_burn:steps_valid + steps_burn], v_traj_test[steps_burn + steps_valid:]
    
    return v_traj, v_traj_test[steps_burn:steps_valid + steps_burn], v_traj_test[steps_burn + steps_valid:]









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

def _load_custom_obs_fn(custom_fn_path):
    """Load a user-defined observation post-function from 'module.submodule:function'."""
    if custom_fn_path is None or ':' not in custom_fn_path:
        raise ValueError("custom observation function must be 'module.submodule:function_name'.")
    module_name, fn_name = custom_fn_path.split(':', 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Cannot load callable '{fn_name}' from module '{module_name}'.")
    return fn


def _build_obs_post_fn(
    obs_fn,
    base_obs_dim,
    device,
    obs_fn_out_dim=None,
    obs_fn_seed=None,
    obs_custom_fn_path=None,
):
    """Builds the post-processing function g(.) for y = g(Px)."""
    if obs_fn_seed is not None:
        torch.manual_seed(obs_fn_seed)

    obs_fn = (obs_fn or 'identity').lower()
    post_matrix = None

    if obs_fn in ['identity', 'none']:
        if obs_fn_out_dim is not None and int(obs_fn_out_dim) != int(base_obs_dim):
            raise ValueError("obs_fn_out_dim must equal len(obs_inds) for identity observation function.")
        out_dim = int(base_obs_dim)
        post_fn = lambda x: x
    elif obs_fn == 'cos2pi':
        if obs_fn_out_dim is not None and int(obs_fn_out_dim) != int(base_obs_dim):
            raise ValueError("obs_fn_out_dim must equal len(obs_inds) for cos2pi observation function.")
        out_dim = int(base_obs_dim)
        post_fn = lambda x: torch.cos(2.0 * math.pi * x)
    elif obs_fn == 'square':
        if obs_fn_out_dim is not None and int(obs_fn_out_dim) != int(base_obs_dim):
            raise ValueError("obs_fn_out_dim must equal len(obs_inds) for square observation function.")
        out_dim = int(base_obs_dim)
        post_fn = lambda x: x ** 2
    elif obs_fn == 'square_root':
        if obs_fn_out_dim is not None and int(obs_fn_out_dim) != int(base_obs_dim):
            raise ValueError("obs_fn_out_dim must equal len(obs_inds) for square_root observation function.")
        out_dim = int(base_obs_dim)
        post_fn = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
    elif obs_fn == 'cube':
        if obs_fn_out_dim is not None and int(obs_fn_out_dim) != int(base_obs_dim):
            raise ValueError("obs_fn_out_dim must equal len(obs_inds) for cube observation function.")
        out_dim = int(base_obs_dim)
        post_fn = lambda x: x ** 3
    elif obs_fn == 'sin':
        if obs_fn_out_dim is not None and int(obs_fn_out_dim) != int(base_obs_dim):
            raise ValueError("obs_fn_out_dim must equal len(obs_inds) for sin observation function.")
        out_dim = int(base_obs_dim)
        post_fn = torch.sin
    elif obs_fn == 'tanh':
        if obs_fn_out_dim is not None and int(obs_fn_out_dim) != int(base_obs_dim):
            raise ValueError("obs_fn_out_dim must equal len(obs_inds) for tanh observation function.")
        out_dim = int(base_obs_dim)
        post_fn = torch.tanh
    elif obs_fn == 'arctan':
        if obs_fn_out_dim is not None and int(obs_fn_out_dim) != int(base_obs_dim):
            raise ValueError("obs_fn_out_dim must equal len(obs_inds) for arctan observation function.")
        out_dim = int(base_obs_dim)
        post_fn = torch.atan
    elif obs_fn == 'linear':
        if obs_fn_out_dim is None:
            raise ValueError("obs_fn_out_dim must be provided for obs_fn='linear'.")
        out_dim = int(obs_fn_out_dim)
        post_matrix = torch.randn(base_obs_dim, out_dim, device=device)
        post_fn = lambda x: x @ post_matrix
    elif obs_fn == 'custom':
        if obs_fn_out_dim is None:
            raise ValueError("obs_fn_out_dim must be provided for obs_fn='custom'.")
        out_dim = int(obs_fn_out_dim)
        custom_fn = _load_custom_obs_fn(obs_custom_fn_path)
        post_fn = custom_fn
    else:
        raise ValueError(
            f"Unsupported obs_fn='{obs_fn}'. Use one of "
            "['identity','cos2pi','square','square_root','cube','sin','tanh','arctan','linear','custom']."
        )

    return post_fn, post_matrix, out_dim


def partial_obs_operator(
    ori_dim,
    obs_inds,
    device,
    seed=None,
    obs_fn='identity',
    obs_fn_out_dim=None,
    obs_fn_seed=None,
    obs_custom_fn_path=None,
):
    if obs_inds is None:
        raise ValueError("obs_inds cannot be None for partial_obs_operator.")
    if seed is not None:
        torch.manual_seed(seed)
    obs_inds = torch.as_tensor(obs_inds, dtype=torch.long, device=device)
    proj = torch.zeros(ori_dim, len(obs_inds), device=device)
    for i, obs_ind in enumerate(obs_inds):
        proj[obs_ind, i] = 1
    post_fn, post_matrix, _ = _build_obs_post_fn(
        obs_fn=obs_fn,
        base_obs_dim=len(obs_inds),
        device=device,
        obs_fn_out_dim=obs_fn_out_dim,
        obs_fn_seed=obs_fn_seed,
        obs_custom_fn_path=obs_custom_fn_path,
    )

    def inner(x):
        return post_fn(x @ proj)

    op_info = {
        'proj': proj,
        'post_matrix': post_matrix,
        'obs_inds': obs_inds,
        'obs_fn': obs_fn,
    }
    # Backward-compatible: for pure linear partial observation, keep returning matrix.
    if obs_fn in ['identity', 'none'] and post_matrix is None:
        return inner, proj
    return inner, op_info


def build_observation_operator(args):
    """Construct observation operator from CLI args."""
    seed_obs = getattr(args, 'seed_obs', None)
    if seed_obs is None:
        seed_raw = getattr(args, 'seed', None)
        if seed_raw is not None and str(seed_raw).lower() != "none":
            seed_obs = int(seed_raw)
    return partial_obs_operator(
        ori_dim=args.ori_dim,
        obs_inds=args.obs_inds,
        device=args.device,
        seed=seed_obs,
        obs_fn=getattr(args, 'obs_fn', 'identity'),
        obs_fn_out_dim=getattr(args, 'obs_fn_out_dim', None),
        obs_fn_seed=getattr(args, 'obs_fn_seed', None),
        obs_custom_fn_path=getattr(args, 'obs_custom_fn_path', None),
    )


def get_test_noise_generator(args):
    """
    Create a dedicated RNG stream for test observation noise.

    This generator is only meant for noise used in:
      true state -> noisy observation (y = H(x_true) + noise).
    """
    seed_val = getattr(args, 'test_random_seed', None)
    if seed_val is None:
        seed_raw = getattr(args, 'seed', None)
        if seed_raw is not None and str(seed_raw).lower() != "none":
            seed_val = int(seed_raw)
        else:
            seed_val = 123
    else:
        seed_val = int(seed_val)

    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed_val)
    return gen

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
        prefix = f"{args.sigma_v}_{args.train_steps}_{args.train_traj_num}_{args.trail}"
        if args.dataset == "lorenz96":
            prefix = f"{args.ori_dim}_" + prefix
        gen_trajs = gen_data(args.dataset, t, v0=x0, sigma_v=args.sigma_v,
                                                        steps_test=args.test_steps * args.test_traj_num,
                                                        steps_valid=args.valid_steps,
                                                        args=args,
                                                        check_disk=args.new_data,
                                                        steps_burn=args.burn_steps,
                                                        dt_iter=args.dt_iter,
                                                        prefix=prefix,
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

import ot
def wasserstein_distance_pt(x, y, n_projections=100):
    """
    Calculates the W-2 distance between two point clouds using PyTorch.
    - If d=1, it uses the efficient sorting-based method.
      Cost is approx. O(M*log(M) + N*log(N)) for each item in the batch.
    - If d>1, it uses the Sliced-Wasserstein distance.
      Cost is approx. O(L * K*log(K)) for each item, where L is the number of
      projections and K is the total number of points (M+N).

    INPUT:
    - x: A torch.Tensor of shape (M, d) or (B, M, d).
    - y: A torch.Tensor of shape (N, d) or (B, N, d).
    - n_projections: The number of random projections for the Sliced method.

    OUTPUT:
    - dist: A tensor containing the W-2 distance(s).
            Shape is (B,) for batched input, or a scalar for unbatched input.
    """
    # --- 1. Input Validation and Preparation ---
    if x.device != y.device:
        raise ValueError("Input tensors must be on the same device")
    
    original_device = x.device
    is_unbatched = x.dim() == 2
    if is_unbatched:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    B, M, d = x.shape
    _B, N, _d = y.shape

    if B != _B or d != _d:
      raise ValueError(f"Shape mismatch: x is {x.shape}, y is {y.shape}")

    # --- 2. Dimension-specific routing ---
    if d == 1:
        x_1d = x.squeeze(-1)
        y_1d = y.squeeze(-1)
        
        # Move to CPU for ot.wasserstein_1d which may not support CUDA
        dist_list_1d = [ot.wasserstein_1d(x_b.to("cpu"), y_b.to("cpu"), p=2) for x_b, y_b in zip(x_1d, y_1d)]
        dist = torch.stack(dist_list_1d).to(original_device)

    else: # d > 1, use Sliced-Wasserstein
        # FIX: Loop through the batch, as ot.sliced_wasserstein_distance
        # does not support batching in the (B, N, d) format.
        dist_list_sliced = [
            ot.sliced_wasserstein_distance(x_b, y_b, n_projections=n_projections, p=2)
            for x_b, y_b in zip(x, y)
        ]
        dist = torch.stack(dist_list_sliced)

    # --- 3. Final Output Formatting ---
    if is_unbatched:
        return dist.squeeze(0)
    return dist
