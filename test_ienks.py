import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from types import SimpleNamespace
import math

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)

class Lorenz63:
    """
    Function:
        Implements the Lorenz 63 model with an RK4 stepper.
    """
    def __init__(self, sigma=10.0, rho=28.0, beta=8./3.):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _rhs(self, x):
        """
        Function:
            Computes the right-hand side of the Lorenz 63 equations.
        Input:
            x (torch.Tensor): State vector(s) of shape [N, 3] or [3]. It can handle batches, e.g., [B*N, 3].
        Output:
            torch.Tensor: The derivative dx/dt for each state vector.
        """
        is_1d = x.ndim == 1
        if is_1d:
            x = x.unsqueeze(0)
        
        dxdt = torch.zeros_like(x)
        dxdt[:, 0] = self.sigma * (x[:, 1] - x[:, 0])
        dxdt[:, 1] = x[:, 0] * (self.rho - x[:, 2]) - x[:, 1]
        dxdt[:, 2] = x[:, 0] * x[:, 1] - self.beta * x[:, 2]
        
        return dxdt.squeeze(0) if is_1d else dxdt

    def step(self, rhs_func, x, dt):
        """
        Function:
            Advances the model state by one time step using RK4.
        Input:
            rhs_func (callable): The right-hand side function.
            x (torch.Tensor): Current state vector(s).
            dt (float): Time step size.
        Output:
            torch.Tensor: State vector(s) at the next time step.
        """
        k1 = rhs_func(x)
        k2 = rhs_func(x + dt * k1 / 2)
        k3 = rhs_func(x + dt * k2 / 2)
        k4 = rhs_func(x + dt * k3)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def center_ensemble(E):
    """
    Function:
        Centers a batched ensemble by subtracting its mean.
    Input:
        E (torch.Tensor): Ensemble of shape [B, N, D].
    Output:
        tuple[torch.Tensor, torch.Tensor]: Anomalies (X) and mean (x).
    """
    x = E.mean(dim=1, keepdim=True)
    X = E - x
    return X, x

def _ienks_analysis(
    ensemble_f,
    observation_y,
    observation_operator_ens,
    sigma_y,
    # --- Model specific args ---
    model_propagator,
    model_rhs,
    model_dt,
    # --- iEnKS hyperparameters ---
    Lag=1,
    nIter=10,
    wtol=1e-5,
    steps_between_analyses=5,
    inflation_factor=1.0
):
    """
    Function:
        Implements a batched Iterative Ensemble Kalman Smoother (iEnKS) analysis step.
    """
    if inflation_factor != 1.0:
        X_f, x_f = center_ensemble(ensemble_f)
        ensemble_f = x_f + X_f * inflation_factor

    B, N, D_state = ensemble_f.shape
    device = ensemble_f.device
    dtype = ensemble_f.dtype

    if observation_y.ndim == 1:
        y = observation_y.unsqueeze(0)
    else:
        y = observation_y
    
    N1 = N - 1
    X0, x0 = center_ensemble(ensemble_f)
    
    w = torch.zeros(B, N, 1, device=device, dtype=dtype)
    T = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    Tinv = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    
    if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim > 0:
        R_inv_sqrt = (1.0 / sigma_y).view(-1, 1, 1)
    else:
        R_inv_sqrt = 1.0 / sigma_y

    def propagate_ensemble_in_window(ens_in):
        ens_flat = ens_in.view(B * N, D_state)
        num_model_steps = Lag * steps_between_analyses
        propagated_ens = ens_flat
        for _ in range(num_model_steps):
            propagated_ens = model_propagator(lambda x: model_rhs(x), propagated_ens, model_dt)
        return propagated_ens.view(B, N, D_state)

    for iteration in range(nIter):
        E_iter = x0 + T @ X0 + (X0.transpose(-1, -2) @ w).transpose(-1, -2)
        E_fwd = propagate_ensemble_in_window(E_iter)
        Eo = observation_operator_ens(E_fwd)
        
        Y, xo_obs = center_ensemble(Eo)
        dy_eff = (y.unsqueeze(1) - xo_obs) * R_inv_sqrt
        Y_eff = Y * R_inv_sqrt
        za = float(N1)
        
        Y_iter = Tinv @ Y_eff
        C_tilde = (Y_iter @ Y_iter.transpose(-2, -1)) + za * torch.eye(N, device=device, dtype=dtype)
        eig_vals, U = torch.linalg.eigh(C_tilde)
        eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)
        
        Cow1 = U @ torch.diag_embed(1.0 / eig_vals_clamped) @ U.transpose(-2, -1)
        
        # CORRECTED: Reverted to the original, correct logic for the gradient term.
        grad_term = Y_iter @ dy_eff.transpose(-2, -1)
        grad = grad_term - za * w
        
        dw = Cow1 @ grad
        w_new = w + dw
        
        if ((w_new - w).norm(p=2, dim=1)**2 / N).mean() < wtol:
            w = w_new
            break
        w = w_new
            
        eig_vals_sqrt = torch.sqrt(eig_vals_clamped)
        T = U @ torch.diag_embed(1.0 / eig_vals_sqrt) @ U.transpose(-2,-1) * math.sqrt(N1)
        Tinv = U @ torch.diag_embed(eig_vals_sqrt) @ U.transpose(-2,-1) / math.sqrt(N1)

    final_delta_mean = (X0.transpose(-2, -1) @ w).transpose(-2, -1)
    final_X_smoothed = T @ X0
    E_smoothed_at_start = x0 + final_delta_mean + final_X_smoothed
    
    return E_smoothed_at_start


if __name__ == '__main__':
    # -- 1. Experiment Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"Using device: {device}")
    
    batch_size = 64

    lorenz_model = Lorenz63()
    dt = 0.03
    obs_every = 5
    total_obs = 400
    n_steps = total_obs * obs_every

    obs_inds = [0] 
    params = {
        'N': 20,
        'Lag': 1,
        'nIter': 10,
        'wtol': 1e-5,
        'infl': 1.0, 
    }

    obs_operator = lambda x: x[..., obs_inds]
    l63_rhs_func = lambda x: lorenz_model._rhs(x)
    rk4_stepper = lambda rhs, x, dt: lorenz_model.step(rhs, x, dt)

    # -- 2. Generate True Run and Observations for a Batch
    print("Generating true state with model spin-up...")
    x_spinup = torch.randn(batch_size, 3, device=device, dtype=torch.float32)
    num_spinup_steps = 2000
    for _ in range(num_spinup_steps):
        x_spinup = rk4_stepper(l63_rhs_func, x_spinup, dt)
    print("Spin-up complete.")

    print("Generating full true trajectory batch...")
    x_true = torch.zeros((batch_size, n_steps + 1, 3), device=device, dtype=torch.float32)
    x_true[:, 0] = x_spinup
    for i in range(n_steps):
        x_true[:, i+1] = rk4_stepper(l63_rhs_func, x_true[:, i], dt)

    obs_idx = np.arange(0, n_steps + 1, obs_every)
    xx_true_obs = x_true[:, obs_idx]
    obs_noise_std = 1.0
    
    true_obs_vals = xx_true_obs[..., obs_inds]
    yy = true_obs_vals + torch.randn_like(true_obs_vals) * obs_noise_std
    
    # -- 3. Run Assimilation on the Batch
    print("Running assimilation...")
    
    initial_true_state_b = xx_true_obs[:, 0, :].unsqueeze(1)
    noise = torch.randn(batch_size, params['N'], 3, device=device) * 1.0
    E = initial_true_state_b + noise

    analysis_means = []
    rmses = []

    for ko in tqdm(range(total_obs + 1), desc="iEnKS Assimilation"):
        k_start = max(0, ko - params['Lag'])
        y = yy[:, ko, :]
        
        E_smoothed_at_start = _ienks_analysis(
            ensemble_f=E,
            observation_y=y,
            observation_operator_ens=obs_operator,
            sigma_y=obs_noise_std,
            model_propagator=lorenz_model.step,
            model_rhs=lorenz_model._rhs,
            model_dt=dt,
            Lag=ko - k_start,
            nIter=params['nIter'],
            wtol=params['wtol'],
            steps_between_analyses=obs_every,
            inflation_factor=params['infl']
        )

        E_analysis_at_ko = E_smoothed_at_start.clone()
        num_steps_to_propagate = (ko - k_start) * obs_every
        if num_steps_to_propagate > 0:
            B_cur, N_cur, D_cur = E_analysis_at_ko.shape
            E_flat = E_analysis_at_ko.view(-1, 3)
            for _ in range(num_steps_to_propagate):
                E_flat = lorenz_model.step(lorenz_model._rhs, E_flat, dt)
            E_analysis_at_ko = E_flat.view(B_cur, N_cur, D_cur)
        
        analysis_mean = E_analysis_at_ko.mean(dim=1)
        rmse = torch.sqrt(torch.mean((analysis_mean - xx_true_obs[:, ko])**2, dim=-1))
        
        analysis_means.append(analysis_mean.cpu().numpy())
        rmses.append(rmse.cpu().numpy())

        if ko >= params['Lag']:
            B_cur, N_cur, D_cur = E_smoothed_at_start.shape
            E_flat = E_smoothed_at_start.view(-1, 3)
            for _ in range(obs_every):
                E_flat = lorenz_model.step(lorenz_model._rhs, E_flat, dt)
            E = E_flat.view(B_cur, N_cur, D_cur)
        else:
            E = E_smoothed_at_start

    # -- 4. Plot Results
    print("Plotting results...")
    analysis_means = np.array(analysis_means)
    rmses = np.array(rmses)
    time_axis = np.arange(len(yy[0])) * dt * obs_every
    
    batch_idx_to_plot = 0
    xx_true_obs_plot = xx_true_obs[batch_idx_to_plot].cpu().numpy()
    analysis_means_plot = analysis_means[:, batch_idx_to_plot, :]
    
    yy_plot = np.full((len(yy[0]), 3), np.nan)
    yy_plot[:, obs_inds] = yy[batch_idx_to_plot].cpu().numpy()
    
    rmses_plot = rmses.mean(axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    title = (f'iEnKS (Batch Size={batch_size}): N={params["N"]}, Lag={params["Lag"]}, nIter={params["nIter"]}, '
             f'obs_inds={obs_inds}, infl={params["infl"]}')
    fig.suptitle(title, fontsize=16)

    axes[0].plot(time_axis, xx_true_obs_plot[:, 0], 'k-', label='True')
    axes[0].plot(time_axis, yy_plot[:, 0], 'rx', markersize=4, label='Obs')
    axes[0].plot(time_axis, analysis_means_plot[:, 0], 'b-', label='Analysis')
    axes[0].set_ylabel('X'); axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title(f"Displaying results for Batch Element {batch_idx_to_plot}")

    axes[1].plot(time_axis, xx_true_obs_plot[:, 1], 'k-')
    axes[1].plot(time_axis, yy_plot[:, 1], 'rx', markersize=4)
    axes[1].plot(time_axis, analysis_means_plot[:, 1], 'b-')
    axes[1].set_ylabel('Y'); axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(time_axis, xx_true_obs_plot[:, 2], 'k-')
    axes[2].plot(time_axis, yy_plot[:, 2], 'rx', markersize=4)
    axes[2].plot(time_axis, analysis_means_plot[:, 2], 'b-')
    axes[2].set_ylabel('Z'); axes[2].grid(True, linestyle='--', alpha=0.6)

    axes[3].plot(time_axis, rmses_plot, 'g-', label='RMSE (Batch Avg)')
    axes[3].set_xlabel('Time'); axes[3].set_ylabel('RMSE')
    axes[3].legend(); axes[3].grid(True, linestyle='--', alpha=0.6)
    axes[3].set_yscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    stable_start_index = 10 
    average_rmse = np.mean(rmses[stable_start_index:])
    print(f"\nAverage RMSE (across batch and after stabilization): {average_rmse:.4f}")