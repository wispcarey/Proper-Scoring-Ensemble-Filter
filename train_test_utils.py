import numpy as np
import time
import os

import torch
import torch.nn as nn
import math

from utils import L63, L96, Rossler, rk4, etd_rk4_wrapper, CircleODE, DoubleWellODE
from utils import AverageMeter, mystery_operator, get_mean_std
from utils import post_process, mean0
from visualization import plot_particle_trajectories_with_histograms, plot_particle_trajectories, plot_and_test_point_clouds
from localization import dist2coeff, create_loc_mat
from loss import compute_loss, compute_es, wasserstein2_multivariate_gaussian
from networks import NaiveNetwork, SetTransformer, Simple_MLP, ConditionTransformerNetwork
from benchmark_analysis import ensemble_kalman_filter_analysis, bootstrap_particle_filter_analysis
from typing import Optional, List, Tuple, Dict, Any

from tqdm.auto import tqdm


def set_models(args):
    # set models
    if args.v == 'CorrTerms':
        model = Simple_MLP(d_input=args.input_dim, d_output=args.obs_dim + args.ori_dim, num_hidden_layers=3).to(args.device)
    elif args.v == 'EtE' or args.v == 'EtE-LRes':
        model = Simple_MLP(d_input=args.input_dim, d_output=args.ori_dim, num_hidden_layers=4).to(args.device)
    elif args.v == 'EtE2':
        model = ConditionTransformerNetwork(u_dim=args.ori_dim + args.obs_dim, y_dim=args.obs_dim, output_dim=args.ori_dim, hidden_dim=args.hidden_dim, 
                                            num_blocks=4, num_heads=8, ff_expansion=2).to(args.device)
    else:
        model = NaiveNetwork(1)
    if args.no_localization or args.v.startswith('EtE'):
        local_model = NaiveNetwork(1)
    else:
        local_model = Simple_MLP(d_input=args.local_input_dim, d_output=args.num_dist, num_hidden_layers=2).to(args.device)
    if args.v == 'EtE2':
        st_model1 = NaiveNetwork(1)
        st_model2 = NaiveNetwork(1)
        
        infl_model = NaiveNetwork(1)
    else:
        if args.st_type == 'separate':
            st_model1 = SetTransformer(input_dim=args.ori_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                        hidden_dim=args.hidden_dim, num_layers=2, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            st_model2 = SetTransformer(input_dim=args.obs_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                        hidden_dim=args.hidden_dim, num_layers=2, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            if args.v.startswith('EtE'):
                infl_model = NaiveNetwork(1)
            else:
                infl_model = Simple_MLP(d_input=args.ori_dim + args.st_output_dim, d_output=args.ori_dim, num_hidden_layers=2).to(args.device)
        elif args.st_type == 'state_only':
            st_model1 = SetTransformer(input_dim=args.ori_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                        hidden_dim=args.hidden_dim, num_layers=2, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            st_model2 = NaiveNetwork(1)
            if args.v.startswith('EtE'):
                infl_model = NaiveNetwork(1)
            else:
                infl_model = Simple_MLP(d_input=args.ori_dim + args.st_output_dim, d_output=args.ori_dim, num_hidden_layers=2).to(args.device)
        elif args.st_type == 'joint':
            st_model1 = SetTransformer(input_dim=args.ori_dim + args.obs_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim * 2, 
                                        hidden_dim=args.hidden_dim, num_layers=3, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            st_model2 = NaiveNetwork(1)
            if args.v.startswith('EtE'):
                infl_model = NaiveNetwork(1)
            else:
                infl_model = Simple_MLP(d_input=args.ori_dim + 2 * args.st_output_dim, d_output=args.ori_dim, num_hidden_layers=2).to(args.device)
        else:
            st_model1, st_model2, infl_model, local_model = NaiveNetwork(1), NaiveNetwork(1), NaiveNetwork(1), NaiveNetwork(1)
    if args.use_data_parallel:
        model, infl_model, local_model, st_model1, st_model2 = \
            nn.DataParallel(model), nn.DataParallel(infl_model), nn.DataParallel(local_model), nn.DataParallel(st_model1), nn.DataParallel(st_model2)
    model_list = [model, infl_model, local_model, st_model1, st_model2]
    total_params = sum(sum(p.numel() for p in model.parameters()) for model in model_list)
    print(f'Total number of parameters: {total_params}')
    
    return model_list

def _get_model_inputs(args, st_model1, st_model2, ens_v_f, hv, obs_y):
    """Prepare input tensors for the models based on args.st_type."""
    N_ens = ens_v_f.shape[1]
    
    if args.noise_st_input:
        st_input_2 = hv + args.sigma_y * torch.randn_like(obs_y, device=args.device)
    else:
        st_input_2 = hv
    
    if args.mlp_y_type == 'obs':
        mlp_y_input = obs_y.expand(-1, N_ens, -1)
    elif args.mlp_y_type == 'innov':
        mlp_y_input = obs_y.expand(-1, N_ens, -1) - hv
    elif args.mlp_y_type == 'noise_innov':
        mlp_y_input = obs_y.expand(-1, N_ens, -1) - hv - args.sigma_y * torch.randn_like(obs_y, device=args.device)
    else:
        raise ValueError("args.mlp_y_type must be one within 'obs', 'innov', 'noise_innov'.")
    
    
    if args.v == 'EtE2':
        nn_input_cat_list = [ens_v_f, st_input_2]
        nn_input = torch.cat(nn_input_cat_list, dim=-1)
    else:
        # Initialize lists to avoid NameError
        local_nn_input_cat_list = []
        infl_nn_input_cat_list = []

        if args.st_type == 'state_only':
            ens_nn_output = st_model1(ens_v_f)
            nn_input_cat_list = [ens_v_f, hv, mlp_y_input, ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            if args.v == 'CorrTerms':
                local_nn_input_cat_list = [obs_y.squeeze(1), ens_nn_output] if args.obs_in_loc else [ens_nn_output]
                infl_nn_input_cat_list = [ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            st_output_dim_actual = args.st_output_dim
            
        elif args.st_type == 'separate':
            ens_nn_output = st_model1(ens_v_f)
            ens_o_nn_output = st_model2(st_input_2)
            nn_input_cat_list = [ens_v_f, hv, obs_y.expand(-1, N_ens, -1), ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1), ens_o_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            if args.v == 'CorrTerms':
                local_nn_input_cat_list = [obs_y.squeeze(1), ens_nn_output, ens_o_nn_output] if args.obs_in_loc else [ens_nn_output, ens_o_nn_output]
                infl_nn_input_cat_list = [ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            st_output_dim_actual = args.st_output_dim
            
        elif args.st_type == 'joint':
            ens_nn_output = st_model1(torch.cat([ens_v_f, st_input_2], dim=-1))
            nn_input_cat_list = [ens_v_f, hv, mlp_y_input, ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            if args.v == 'CorrTerms':
                local_nn_input_cat_list = [obs_y.squeeze(1), ens_nn_output] if args.obs_in_loc else [ens_nn_output]
                infl_nn_input_cat_list = [ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            st_output_dim_actual = args.st_output_dim * 2
            
        else:
            raise ValueError(f"Unknown args.st_type: {args.st_type}")
        nn_input = torch.cat(nn_input_cat_list, dim=-1).view(-1, args.input_dim)
    
    if args.v == 'CorrTerms':
        # These are only computed and returned if needed
        local_nn_input = torch.cat(local_nn_input_cat_list, dim=-1)
        infl_nn_input = torch.cat(infl_nn_input_cat_list, dim=-1).view(-1, args.ori_dim + st_output_dim_actual)
        return nn_input, local_nn_input, infl_nn_input
    elif args.v == 'EtE' or args.v == 'EtE-LRes':
        return nn_input, None, None # Return None for unused values
    elif args.v == 'EtE2':
        return nn_input, obs_y.squeeze(1), None
    else:
        raise NotImplementedError(f"args.v = {args.v} is not implemented.")

def _process_analysis_step(args, model_list, ens_v_f, hv, obs_y, ens_i_innov, mean_ens_v_f, mean_hv, sigma_y=None, infl=1.0, loc_radius=None):
    """
    Process the analysis ensemble step.
    
    Inputs:
        args: Argument parser with config (must contain Lvy/Lyy for EnKF/LETKF loc).
        model_list: List of neural networks.
        ens_v_f: Forecast ensemble (B, N_ens, D_state).
        hv: Observation of forecast ensemble (B, N_ens, d_obs).
        obs_y: Observations (B, d_obs).
        ens_i_innov: Innovation (obs_y - hv) + noise (B, N_ens, d_obs). [Used for EnKF]
        mean_ens_v_f: Mean of forecast ensemble (B, 1, D_state).
        mean_hv: Mean of observation forecast (B, 1, d_obs).
        sigma_y: Observation noise std.
        infl: Multiplicative inflation factor (applied post-analysis).
        loc_radius: Localization radius.

    Outputs:
        ens_v_a: Analyzed ensemble (B, N_ens, D_state).
        loc_nn_output: Localization mask from NN (if applicable, else None).
    """
    if sigma_y is None:
        sigma_y = args.sigma_y
        
    if isinstance(sigma_y, torch.Tensor):
        sigma_y = sigma_y.detach().to(args.device)
    else:
        sigma_y = torch.tensor(sigma_y, dtype=torch.float32).to(args.device)
    
    model, infl_model, local_model, st_model1, st_model2 = model_list
    B, N_ens, D_state = ens_v_f.shape
    d_obs = hv.shape[2]
    
    # Handle sigma_y shape
    if sigma_y.ndim == 0:
        sigma_y = sigma_y.expand(B).view(B, 1, 1)
    else:
        sigma_y = sigma_y.view(B, 1, 1)
    
    loc_nn_output = None 

    if args.v == 'CorrTerms':
        nn_input, local_nn_input, infl_nn_input = _get_model_inputs(
            args, st_model1, st_model2, ens_v_f, hv, obs_y
        )
        
        nn_output = model(nn_input).view(B, N_ens, -1)
        infl_output = infl_model(infl_nn_input).view(B, N_ens, -1)
        
        if hasattr(args, 'zero_infl') and args.zero_infl:
            Vnn1 = ens_v_f
        else:
            Vnn1 = ens_v_f + infl_output
        Vnn2 = ens_v_f - mean_ens_v_f + nn_output[:, :, :D_state]
        Ynn = hv - mean_hv + nn_output[:, :, D_state:]

        R_cov = (sigma_y.view(B, 1, 1) ** 2) * torch.eye(d_obs, device=args.device).unsqueeze(0).repeat(B, 1, 1)
        
        if args.no_localization:
            K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) 
            K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) + R_cov * (N_ens - 1)
        else:
            loc_nn_output = torch.sigmoid(local_model(local_nn_input)) * args.loc_max_val
            loc_mat_vy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lvy)
            loc_mat_yy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lyy)
            
            K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) * loc_mat_vy
            K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) * loc_mat_yy + R_cov * (N_ens - 1)
        
        K = torch.bmm(K1, torch.inverse(K2))
        current_analyzed_ens_v_a = Vnn1 + torch.bmm(ens_i_innov, K.transpose(1, 2))
    
    elif args.v == 'EtE':
        nn_input, _, _ = _get_model_inputs(
            args, st_model1, st_model2, ens_v_f, hv, obs_y
        )
        nn_output = model(nn_input).view(B, N_ens, -1)
        current_analyzed_ens_v_a = nn_output
        
    elif args.v == 'EtE-LRes':
        nn_input, _, _ = _get_model_inputs(
            args, st_model1, st_model2, ens_v_f, hv, obs_y
        )
        nn_output = model(nn_input).view(B, N_ens, -1)
        current_analyzed_ens_v_a = ens_v_f + nn_output
        
    elif args.v == 'EtE2':
        nn_input, y, _ = _get_model_inputs(
            args, st_model1, st_model2, ens_v_f, hv, obs_y
        )
        nn_output = model(nn_input, y).view(B, N_ens, -1)
        current_analyzed_ens_v_a = nn_output
        
    elif args.v == 'EnKF':
        Vnn1 = ens_v_f
        Vnn2 = ens_v_f - mean_ens_v_f
        Ynn = hv - mean_hv

        R_cov = (sigma_y.view(B, 1, 1) ** 2) * torch.eye(d_obs, device=args.device).unsqueeze(0).repeat(B, 1, 1)
        
        K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) 
        K2_ens = torch.bmm(Ynn.transpose(1, 2), Ynn)
        
        # Apply Covariance Localization
        if loc_radius is not None and loc_radius > 0:
            if args.Lvy is None or args.Lyy is None:
                raise ValueError("EnKF localization requires pre-computed args.Lvy and args.Lyy")

            rho_vy = dist2coeff(args.Lvy, loc_radius, tag='GC')
            rho_yy = dist2coeff(args.Lyy, loc_radius, tag='GC')
            
            K1 = K1 * rho_vy.unsqueeze(0)
            K2_ens = K2_ens * rho_yy.unsqueeze(0)

        K2 = K2_ens + R_cov * (N_ens - 1)
        K = torch.bmm(K1, torch.inverse(K2))
        current_analyzed_ens_v_a = Vnn1 + torch.bmm(ens_i_innov, K.transpose(1, 2))
        
        # Apply Post-Analysis Inflation
        if infl != 1.0:
            ens_mean = current_analyzed_ens_v_a.mean(dim=1, keepdim=True)
            current_analyzed_ens_v_a = ens_mean + infl * (current_analyzed_ens_v_a - ens_mean)

    elif args.v == 'LETKF':
        if args.Lvy is None:
            raise ValueError("LETKF requires args.Lvy for localization distances.")

        # Prepare Anomalies (B, N_ens, D)
        X_b = ens_v_f - mean_ens_v_f
        Y_b = hv - mean_hv
        
        # Innovation (Mean only, no perturbations for LETKF usually)
        # d_mean: (B, 1, D_obs)
        if mean_hv.shape[1] != 1:
            mean_hv_for_innov = mean_hv.mean(dim=1, keepdim=True)
        else:
            mean_hv_for_innov = mean_hv
        d_mean = obs_y.view(B, 1, d_obs) - mean_hv_for_innov

        # To store the analyzed state for each grid point
        analyzed_cols = []
        
        # Pre-compute Identity matrix
        eye_N = torch.eye(N_ens, device=args.device).unsqueeze(0) # (1, N, N)

        # Iterate over each state dimension (Grid Point)
        # Using args.ori_dim or D_state (they should be consistent here)
        for k in range(D_state):
            # 1. Identify Local Observations
            # args.Lvy[k] is distance from state k to all obs
            dists = args.Lvy[k] # Shape (d_obs,)
            local_obs_idx = torch.where(dists <= loc_radius)[0]
            
            # If no local observations, analysis = forecast
            if len(local_obs_idx) == 0:
                analyzed_cols.append(ens_v_f[:, :, k:k+1])
                continue
            
            # 2. Extract Local Quantities
            # Y_loc: (B, N_ens, n_loc)
            Y_loc = Y_b[:, :, local_obs_idx]
            
            # d_loc: (B, 1, n_loc)
            d_loc = d_mean[:, :, local_obs_idx]
            
            # Normalize by R (assuming diagonal sigma)
            # sigma_y is (B, 1, 1), broadcast to (B, 1, n_loc)
            sigma_loc = sigma_y.view(B, 1, 1)
            Y_loc_norm = Y_loc / sigma_loc
            d_loc_norm = d_loc / sigma_loc
            
            # 3. Compute Transform Matrix T
            # Hessian H = (N-1)I + Y_loc_norm @ Y_loc_norm.T
            # Shape: (B, N_ens, N_ens)
            YRY = torch.bmm(Y_loc_norm, Y_loc_norm.transpose(1, 2))
            H = (N_ens - 1) * eye_N + YRY
            
            # Pa = H^-1 (Analysis Covariance in Ensemble Space)
            # Use cholesky or standard inverse. Inverse is safer for general batch.
            Pa = torch.inverse(H)
            
            # 4. Compute Weights
            # Mean weights w_bar: (B, N_ens, 1)
            # w_bar = Pa @ Y_loc_norm @ d_loc_norm.T
            w_bar = torch.bmm(torch.bmm(Pa, Y_loc_norm), d_loc_norm.transpose(1, 2))
            
            # Perturbation weights W_a: (B, N_ens, N_ens)
            # W_a = sqrt(N-1) * Pa^(1/2)
            # Using Eigendecomposition for square root: Pa = V L V^T -> Pa^0.5 = V L^0.5 V^T
            L, V = torch.linalg.eigh(Pa)
            L = torch.clamp(L, min=1e-10) # Numerical stability
            Pa_sqrt = torch.bmm(V, torch.diag_embed(torch.sqrt(L)))
            Pa_sqrt = torch.bmm(Pa_sqrt, V.transpose(1, 2))
            
            W_a = math.sqrt(N_ens - 1) * Pa_sqrt
            
            # 5. Update State
            # T = w_bar + W_a (Broadcasting w_bar to all columns)
            # x_a = x_f_mean + X_b @ T
            # Note: X_b is (B, N_ens, D_state). We need local X_b_k: (B, N_ens, 1)
            # In ensemble space logic:
            # X_a_k = X_b_k.T @ (w_bar + W_a) -> Result (B, 1, N_ens) then transpose back?
            # Let's write explicitly:
            #   Analyzed Ensemble = Mean + Perturbation
            #   Ens_A = (mean_f + X_b^T w_bar) + (X_b^T W_a)
            #   X_b_k: (B, N_ens, 1) -> X_b_k^T: (B, 1, N_ens)
            
            X_b_k_T = X_b[:, :, k:k+1].transpose(1, 2) # (B, 1, N_ens)
            
            # Transform matrix T_total: (B, N_ens, N_ens)
            # Adding w_bar to every column of W_a effectively adds the mean update to every member
            T_total = W_a + w_bar 
            
            # Apply transform
            # (B, 1, N_ens) @ (B, N_ens, N_ens) -> (B, 1, N_ens) -> transpose -> (B, N_ens, 1)
            update_k = torch.bmm(X_b_k_T, T_total).transpose(1, 2)
            
            analyzed_cols.append(mean_ens_v_f[:, :, k:k+1] + update_k)
            
        current_analyzed_ens_v_a = torch.cat(analyzed_cols, dim=2)
        
        # Apply Post-Analysis Inflation (Consistent with EnKF branch)
        if infl != 1.0:
            ens_mean = current_analyzed_ens_v_a.mean(dim=1, keepdim=True)
            current_analyzed_ens_v_a = ens_mean + infl * (current_analyzed_ens_v_a - ens_mean)

    else:
        raise NotImplementedError(f"args.v = {args.v} is not implemented.")

    if args.clamp is not None:
        ens_v_a = torch.clamp(current_analyzed_ens_v_a, min=-args.clamp, max=args.clamp)
    else:
        ens_v_a = current_analyzed_ens_v_a
    
    return ens_v_a, loc_nn_output

### compute likelihoods for weight particle filter training
def compute_likelihood_weights(hv, obs_y, sigma_y):
    """
    Computes normalized importance weights w ~ p(y|x) for the forecast ensemble.
    
    Args:
        hv (torch.Tensor): H(x_forecast). Shape: [B, N, D_obs]
        obs_y (torch.Tensor): Observation y. Shape: [B, D_obs]
        sigma_y (float): Observation noise standard deviation.
        
    Returns:
        torch.Tensor: Normalized weights. Shape: [B, N]
    """
    # Ensure obs_y matches particle dimension for broadcasting: [B, 1, D_obs]
    # obs_y_expanded = obs_y.unsqueeze(1)
    
    # Compute squared error summed over observation dimensions: ||y - H(x)||^2
    # Shape: [B, N]
    sq_error = torch.sum((hv - obs_y)**2, dim=2)
    
    # Calculate Log-Weights: -0.5 * error / sigma^2
    # Adding a small epsilon to sigma is good practice but args.sigma_y implies valid std.
    log_w = -0.5 * sq_error / (sigma_y ** 2)
    
    # Stable Softmax over the particle dimension (dim=1) to get normalized weights
    w = torch.softmax(log_w, dim=1)
    
    return w

###############################################################

def train_model(epoch, loader, model_list, optimizer, scheduler, args, H_info=None):
    """
    Function to train the model for one epoch with support for WPF and NLL losses.
    """
    model, infl_model, local_model, st_model1, st_model2 = model_list
    m = args.N
    losses = AverageMeter()
    batch_time = AverageMeter()

    # --- Forward Function Selection ---
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
    elif args.dataset == 'rossler':
        forward_fun = Rossler.forward
    elif args.dataset == "lorenz96":
        forward_fun = L96.forward
    elif args.dataset == "circle":
        forward_fun = CircleODE.forward
    elif args.dataset == "Hdoublewell":
        forward_fun = DoubleWellODE.forward
    elif args.dataset == "ks":
        if args.dt_iter <= 0:
            raise ValueError("args.dt_iter must be positive for KS model.")
        forward_fun = etd_rk4_wrapper(device=args.device, dt=args.dt / args.dt_iter)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")
    
    # --- Observation Operator Setup ---
    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info

    # --- Set Models to Train Mode ---
    model.train()
    if hasattr(infl_model, 'train'): infl_model.train()
    if hasattr(local_model, 'train'): local_model.train()
    if hasattr(st_model1, 'train'): st_model1.train()
    if hasattr(st_model2, 'train'): st_model2.train()

    all_trainable_params = []
    for m_ in [model, infl_model, local_model, st_model1, st_model2]:
        if hasattr(m_, 'parameters') and not isinstance(m_, NaiveNetwork):
            all_trainable_params.extend(list(filter(lambda p: p.requires_grad, m_.parameters())))

    num_batches_all_nan = 0
    
    # --- Loss Mode Detection ---
    wpf_loss_names = ['wpf_ed', 'wpf_fmmd', 'wpf_ammd']
    is_wpf_mode = any(lt in args.loss_type for lt in wpf_loss_names)
    is_nll_mode = 'nll' in args.loss_type

    # --- Batch Loop ---
    for batch_ind, batch_v_trajectory in enumerate(loader):
        t_start = time.time()
        batch_v_trajectory = batch_v_trajectory.to(device=args.device)
        current_actual_batch_size = batch_v_trajectory.shape[1]
        optimizer.zero_grad()
        
        ens_v_a = batch_v_trajectory[0].unsqueeze(1).repeat(1, m, 1)
        ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens
        
        end_ind_t = min(epoch + 1, len(batch_v_trajectory) - 1) if args.loss_warm_up else len(batch_v_trajectory) - 1
        if end_ind_t <= 0:
            if (batch_ind + 1) % args.print_batch == 0:
                print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
                      f'Skipped batch due to end_ind_t <=0')
            batch_time.update(time.time() - t_start)
            continue
        
        accumulated_loss_for_batch_load = 0.0
        num_valid_loss_contributions = 0
        
        # Storage for Trajectory Loss (running_loss = False)
        collected_ens_v_a = [] if not args.running_loss else None
        collected_additional_data = {'target_ens': [], 'target_weights': [], 'true_obs': []} if not args.running_loss else None

        # ======================= [UNIFIED TIME-STEP LOOP] =======================
        for i in range(end_ind_t):
            # --- Forecast Step ---
            obs_y = H_fun(batch_v_trajectory[i + 1].unsqueeze(1))
            obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)
            
            ens_v_a_forecast_input = ens_v_a.reshape(-1, args.ori_dim)
            for j_iter in range(args.dt_iter):
                if args.dataset == 'ks':
                    ens_v_a_forecast_input = forward_fun(ens_v_a_forecast_input, None, args.dt / args.dt_iter)
                else:
                    current_time_for_rk4 = i * args.dt + j_iter * (args.dt / args.dt_iter)
                    ens_v_a_forecast_input = rk4(forward_fun, ens_v_a_forecast_input, current_time_for_rk4, args.dt / args.dt_iter)
            
            ens_v_f = ens_v_a_forecast_input.view(-1, m, args.ori_dim)
            ens_v_f = ens_v_f + torch.randn_like(ens_v_f, device=args.device) * args.sigma_v
            
            # Forecast in Observation Space
            hv = H_fun(ens_v_f)
            
            # --- Compute Likelihood Weights (if needed) ---
            current_lik_weights = None
            if is_wpf_mode:
                current_lik_weights = compute_likelihood_weights(hv, obs_y, args.sigma_y)

            # --- Analysis Step ---
            r_noise = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
            ens_i_innov = obs_y - hv - r_noise
            mean_hv = torch.mean(hv, dim=1, keepdim=True)
            mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True)

            current_analyzed_ens_v_a, _ = _process_analysis_step(
                args, model_list, ens_v_f, hv, obs_y,
                ens_i_innov, mean_ens_v_f, mean_hv,
            )

            # --- Strategy-Dependent Action ---
            if args.running_loss:
                if (i + 1) > args.ignore_first:
                    ens_tensor_step = current_analyzed_ens_v_a.unsqueeze(0)
                    batch_v_step = batch_v_trajectory[i + 1].unsqueeze(0)
                    
                    # Prepare additional inputs for this step
                    additional_inputs_step = {}
                    if is_wpf_mode:
                        additional_inputs_step.update({
                            'target_ens': ens_v_f.unsqueeze(0),
                            'target_weights': current_lik_weights.unsqueeze(0),
                            'ens_weights': None,
                            'sigma': getattr(args, 'kes_sigma', 1.0)
                        })
                    if is_nll_mode:
                        additional_inputs_step.update({
                            'obs_map': H_fun,
                            'sigma_y': args.sigma_y,
                            'true_obs': obs_y.unsqueeze(0), # [1, B, d_obs]
                            'ens_weights': None
                        })

                    nan_mask_this_step = torch.isnan(ens_tensor_step).any(dim=(0, 2, 3)).squeeze(0) 
                    valid_B_mask_this_step = ~nan_mask_this_step

                    if valid_B_mask_this_step.any():
                        step_loss_sum = 0
                        for loss_type_val in args.loss_type:
                            step_loss_sum += compute_loss(
                                ens_tensor=ens_tensor_step, batch_v=batch_v_step,
                                loss_type=loss_type_val, ignore_first=0, end_ind=None,
                                valid_B_mask=valid_B_mask_this_step.unsqueeze(0),
                                norm_p=args.es_p, kes_sigma=args.kes_sigma, return_sum=True,
                                additional_inputs=additional_inputs_step
                            )
                        
                        accumulated_loss_for_batch_load += step_loss_sum
                        num_valid_in_step = torch.sum(valid_B_mask_this_step).item()
                        num_valid_loss_contributions += num_valid_in_step
                        losses.update(step_loss_sum.item() / num_valid_in_step, num_valid_in_step)
            else:
                # Trajectory loss: collect tensors
                collected_ens_v_a.append(current_analyzed_ens_v_a)
                if is_wpf_mode or is_nll_mode:
                    if is_wpf_mode:
                        collected_additional_data['target_ens'].append(ens_v_f)
                        collected_additional_data['target_weights'].append(current_lik_weights)
                    if is_nll_mode:
                        # obs_y is [B, 1, d_obs], squeeze to [B, d_obs]
                        collected_additional_data['true_obs'].append(obs_y.squeeze(1))
            
            # --- State Update and Detach ---
            ens_v_a = current_analyzed_ens_v_a
            if epoch <= args.detach_training_epoch and args.detach_steps > 0 and (i + 1) % args.detach_steps == 0 and (i + 1) < end_ind_t:
                ens_v_a = ens_v_a.detach()
        
        # ======================= [POST-LOOP BACKPROPAGATION] =======================
        if args.running_loss:
            if num_valid_loss_contributions > 0:
                average_loss_for_loaded_batch = accumulated_loss_for_batch_load / num_valid_loss_contributions
                average_loss_for_loaded_batch.backward()
                if all_trainable_params:
                    nn.utils.clip_grad_norm_(all_trainable_params, max_norm=getattr(args, 'grad_clip_norm', 1.0))
                optimizer.step()
            else:
                num_batches_all_nan += 1
        else:
            # Trajectory loss
            if len(collected_ens_v_a) > args.ignore_first:
                ens_tensor = torch.stack(collected_ens_v_a, dim=0)[args.ignore_first:] # [T, B, N, D]
                batch_v = batch_v_trajectory[1:end_ind_t + 1][args.ignore_first:]
                
                additional_inputs_traj = {}
                if is_wpf_mode:
                    additional_inputs_traj.update({
                        'target_ens': torch.stack(collected_additional_data['target_ens'], dim=0)[args.ignore_first:],
                        'target_weights': torch.stack(collected_additional_data['target_weights'], dim=0)[args.ignore_first:],
                        'ens_weights': None,
                        'sigma': getattr(args, 'kes_sigma', 1.0)
                    })
                if is_nll_mode:
                    additional_inputs_traj.update({
                        'obs_map': H_fun,
                        'sigma_y': args.sigma_y,
                        'true_obs': torch.stack(collected_additional_data['true_obs'], dim=0)[args.ignore_first:], # [T, B, d_obs]
                        'ens_weights': None
                    })
                
                if ens_tensor.shape[0] > 0:
                    valid_B_mask = ~torch.isnan(ens_tensor).any(dim=(2, 3))
                    num_valid_loss_contributions = torch.sum(valid_B_mask).item()

                    if num_valid_loss_contributions > 0:
                        total_loss = sum(compute_loss(
                                ens_tensor=ens_tensor, batch_v=batch_v, loss_type=lt,
                                ignore_first=0, end_ind=None, valid_B_mask=valid_B_mask,
                                norm_p=args.es_p, kes_sigma=args.kes_sigma, return_sum=True,
                                additional_inputs=additional_inputs_traj
                            ) for lt in args.loss_type)
                        
                        average_loss = total_loss / num_valid_loss_contributions
                        average_loss.backward()
                        if all_trainable_params:
                            nn.utils.clip_grad_norm_(all_trainable_params, max_norm=getattr(args, 'grad_clip_norm', 1.0))
                        optimizer.step()
                        losses.update(average_loss.item(), num_valid_loss_contributions)
                    else:
                        num_batches_all_nan += 1
                else:
                    num_batches_all_nan += 1
            else:
                num_batches_all_nan += 1

        # --- Logging ---
        batch_time.update(time.time() - t_start)
        total_possible = current_actual_batch_size * max(0, end_ind_t - args.ignore_first)
        no_nan_perc = (num_valid_loss_contributions / total_possible * 100) if total_possible > 0 else 100.0

        if (batch_ind + 1) % args.print_batch == 0:
            print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
                f'Batch time {batch_time.val:.3f} (Avg: {batch_time.avg:.3f})\t'
                f'Loss {losses.val:.3f} (Avg: {losses.avg:.3f})\t'
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}\tNo NAN %: {no_nan_perc:.2f}%')

    if num_batches_all_nan == len(loader) and len(loader) > 0:
        print(f"Warning: All batches in epoch {epoch} resulted in NaN.")
        if losses.count == 0: return float('nan')

    scheduler.step()
    return losses.avg


def generate_and_cache_pf_results(
    loader,
    args,
    H_info,
    check_disk: bool = True,
    calculate_crps: bool = True,
    save_figure: bool = False,
):
    """
    Runs a particle filter, saves results (means, covariances) to a cache file,
    and computes performance metrics (RMSE, optional CRPS).

    NEW IN THIS VERSION:
    - Records BOTH prior (forecast) and posterior (analysis) ensemble statistics in the cache:
        {'prior_means', 'prior_covs', 'post_means', 'post_covs'}
    - Plots TWO figures per selected batch index and time step when `save_figure=True`:
        (1) Prior (blue points) + black history trajectory + orange observation star
        (2) Posterior (red points) + black history trajectory + orange observation star

    If calculate_crps is False, CRPS keys are omitted from the output.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader providing dataset batches (shape: T x B x D).
        args (argparse.Namespace): Script arguments (expects fields used below).
        H_info (tuple): (H_fun, H) observation operator function and matrix; if None, uses mystery_operator.
        check_disk (bool): If True, checks cache file existence and returns NaNs (metrics) if present.
        calculate_crps (bool): If True, calculates CRPS and RCRPS metrics.
        save_figure (bool): If True, saves prior/posterior figures at selected steps.

    Returns:
        dict: A dictionary containing performance metrics. CRPS-related keys exist only if calculate_crps=True.
    """

    # --- Model and Observation Initialization ---
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
    elif args.dataset == 'rossler':
        forward_fun = Rossler.forward
    elif args.dataset == "lorenz96":
        forward_fun = L96.forward
    elif args.dataset == "circle":
        forward_fun = CircleODE.forward
    elif args.dataset == "Hdoublewell":
        forward_fun = DoubleWellODE.forward
    elif args.dataset == "ks":
        if args.dt_iter <= 0:
            raise ValueError("args.dt_iter must be positive for KS model.")
        forward_fun = etd_rk4_wrapper(device=args.device, dt=args.dt / args.dt_iter)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info

    # --- Cache Filepath Generation ---
    first_batch_for_shape = next(iter(loader))
    traj_len = first_batch_for_shape.shape[0]
    batch_size = first_batch_for_shape.shape[1]
    cache_dir = os.path.join('data', args.dataset)
    cache_filename = f"pf_results_sigma_y_{args.sigma_y}_batch_{batch_size}_len_{traj_len}_pfN_{args.pf_N}_{args.seed}.pt"
    cache_filepath = os.path.join(cache_dir, cache_filename)

    # --- Check for Existing Cache ---
    if check_disk and os.path.exists(cache_filepath):
        print(f"Particle filter results already exist at: {cache_filepath}")
        metrics_keys = ['mean_rmse', 'std_rmse', 'mean_rrmse', 'std_rrmse']
        if calculate_crps:
            metrics_keys.extend(['mean_crps', 'std_crps', 'mean_rcrps', 'std_rcrps'])
        return {key: float('nan') for key in metrics_keys}

    print(f"Generating particle filter results and saving to: {cache_filepath}")
    all_pf_results_to_cache = []

    # --- Initialize Metric Dictionaries ---
    all_pf_metrics = {
        'rmse': torch.empty(0, device=args.device),
        'rrmse': torch.empty(0, device=args.device),
    }
    if calculate_crps:
        all_pf_metrics['crps'] = torch.empty(0, device=args.device)
        all_pf_metrics['rcrps'] = torch.empty(0, device=args.device)

    # --- Which batch indices to visualize when saving figures ---
    # (Matches your previous behavior of plotting the first two items if available.)
    vis_indices: List[int] = [i for i in [0, 1] if i < batch_size]

    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)  # shape: (T, B, D)

            # --- Particle Filter Initialization ---
            pf_ens_v_a = batch_v[0].unsqueeze(1).repeat(1, args.pf_N, 1)  # (B, Np, D)
            pf_ens_v_a += torch.randn_like(pf_ens_v_a, device=args.device) * args.sigma_ens

            # These will be stacked and cached per-batch
            batch_prior_means_to_cache, batch_prior_covs_to_cache = [], []
            batch_post_means_to_cache,  batch_post_covs_to_cache  = [], []

            batch_rmse_steps = []
            if calculate_crps:
                batch_crps_steps = []

            # --- Metrics at t=0 (posterior at initialization) ---
            true_state_t0 = batch_v[0]  # (B, D)
            rmse_t0 = torch.sqrt(torch.mean((pf_ens_v_a.mean(dim=1) - true_state_t0) ** 2, dim=1))
            batch_rmse_steps.append(rmse_t0)
            if calculate_crps:
                crps_t0 = compute_es(pf_ens_v_a.unsqueeze(0), true_state_t0.unsqueeze(0), norm_p=1)
                batch_crps_steps.append(crps_t0)

            # --- Generate Observations y_t ---
            obs_y_list = [
                H_fun(batch_v[i].unsqueeze(1)) + args.sigma_y * torch.randn_like(H_fun(batch_v[i].unsqueeze(1)))
                for i in range(len(batch_v))
            ]

            # --- Main PF Assimilation Loop ---
            for i in range(len(batch_v) - 1):
                # -------- Forecast (PRIOR) step --------
                pf_ens_v_a_forecast_input = pf_ens_v_a.view(-1, args.ori_dim)  # (B*Np, D)
                for j_iter in range(args.dt_iter):
                    if args.dataset == 'ks':
                        pf_ens_v_a_forecast_input = forward_fun(
                            pf_ens_v_a_forecast_input, None, args.dt / args.dt_iter
                        )
                    else:
                        current_time_for_rk4 = i * args.dt + j_iter * (args.dt / args.dt_iter)
                        pf_ens_v_a_forecast_input = rk4(
                            forward_fun, pf_ens_v_a_forecast_input, current_time_for_rk4, args.dt / args.dt_iter
                        )

                pf_ens_v_f = pf_ens_v_a_forecast_input.view(-1, args.pf_N, args.ori_dim)  # (B, Np, D)
                pf_ens_v_f += torch.randn_like(pf_ens_v_f) * args.sigma_v

                # Cache PRIOR stats (means/covs)
                prior_mean = torch.mean(pf_ens_v_f, dim=1)                     # (B, D)
                prior_cov  = get_ens_cov(pf_ens_v_f)                           # (B, D, D)
                batch_prior_means_to_cache.append(prior_mean)
                batch_prior_covs_to_cache.append(prior_cov)

                # Prepare state-space "observation" position for plotting (use true state at t=i+1)
                # This gives a 3D anchor to show as an orange star.
                # If your desired "observation" lives in obs-space, adapt this mapping accordingly.
                true_state_ti1 = batch_v[i + 1]  # (B, D)
                obs_for_plot = true_state_ti1[:, :3].detach().cpu()  # (B, 3)

                # -------- Analysis (POSTERIOR) step --------
                pf_ens_v_a = bootstrap_particle_filter_analysis(
                    pf_ens_v_f,
                    obs_y_list[i + 1].squeeze(1),  # (B, obs_dim)
                    H.transpose(1, 0),
                    args.sigma_y,
                    resampling_method="multinomial",
                    sigma_reg=args.sigma_reg,
                    max_chunk_size=1000000,
                    resample_on_cpu=False,
                )
                pf_ens_v_a = torch.clamp(pf_ens_v_a, min=-args.clamp, max=args.clamp)

                # Cache POSTERIOR stats (means/covs)
                post_mean = torch.mean(pf_ens_v_a, dim=1)                      # (B, D)
                post_cov  = get_ens_cov(pf_ens_v_a)                            # (B, D, D)
                batch_post_means_to_cache.append(post_mean)
                batch_post_covs_to_cache.append(post_cov)

                # Metrics at t=i+1 using POSTERIOR mean
                rmse_ti = torch.sqrt(torch.mean((post_mean - true_state_ti1) ** 2, dim=1))
                batch_rmse_steps.append(rmse_ti)
                if calculate_crps:
                    crps_ti = compute_es(pf_ens_v_a.unsqueeze(0), true_state_ti1.unsqueeze(0), norm_p=1)
                    batch_crps_steps.append(crps_ti)

                # -------- Visualization (PRIOR + POSTERIOR), both include observation --------
                if i < 600 and save_figure:
                    save_folder = f'save/{args.dataset}_pf_vis'
                    os.makedirs(save_folder, exist_ok=True)

                    # Use the same time step prefix, but suffix per batch index and type
                    base_prefix = (
                        f'{save_folder}/sigma_y{args.sigma_y}_batch{batch_size}_len{traj_len}_pfN{args.pf_N}'
                        f'_timestep{i+1}_{args.seed}'
                    )

                    # We plot per selected batch index so each figure gets the correct observation vector
                    for bidx in vis_indices:
                        # Prepare per-item tensors for the plotting helper (shape: (1, Np, 3))
                        prior_cloud_for_plot     = pf_ens_v_f[bidx:bidx+1, :, :3].detach().cpu()
                        posterior_cloud_for_plot = pf_ens_v_a[bidx:bidx+1, :, :3].detach().cpu()

                        # History trajectory for this item up to current step (shape: steps x 1 x 3)
                        hist_traj = batch_v[1:i+2, bidx:bidx+1, :3].detach().cpu()

                        # Per-item observation vector (3,)
                        obs_vec = obs_for_plot[bidx]  # (3,)

                        # PRIOR (blue)
                        prefix_prior = f"{base_prefix}_b{bidx}_PRIOR"
                        plot_and_test_point_clouds(
                            args,
                            prior_cloud_for_plot,             # (1, Np, 3)
                            num_samples_plot=100000,
                            num_samples_test=1000,
                            prefix=prefix_prior,
                            point_color="blue",
                            # observation=obs_vec,
                            observation=None,
                            num_repeats=1,
                            plot_indices=[0],
                            history_traj=hist_traj,
                        )

                        # POSTERIOR (red)
                        prefix_post = f"{base_prefix}_b{bidx}_POST"
                        plot_and_test_point_clouds(
                            args,
                            posterior_cloud_for_plot,         # (1, Np, 3)
                            num_samples_plot=100000,
                            num_samples_test=1000,
                            prefix=prefix_post,
                            point_color="red",
                            # observation=obs_vec,
                            observation=None,
                            num_repeats=1,
                            plot_indices=[0],
                            history_traj=hist_traj,
                        )

            # --- Aggregate and Cache Batch Results ---
            all_pf_results_to_cache.append({
                'prior_means': torch.stack(batch_prior_means_to_cache),  # (T-1, B, D)
                'prior_covs':  torch.stack(batch_prior_covs_to_cache),   # (T-1, B, D, D)
                'post_means':  torch.stack(batch_post_means_to_cache),   # (T-1, B, D)
                'post_covs':   torch.stack(batch_post_covs_to_cache),    # (T-1, B, D, D)
            })

            # --- Calculate and Aggregate Average Metrics for the Batch (posterior-based) ---
            rmse_val = torch.mean(torch.stack(batch_rmse_steps), dim=0)                 # (B,)
            rms_val  = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0) # (B,)
            all_pf_metrics['rmse']  = torch.cat((all_pf_metrics['rmse'],  rmse_val))
            all_pf_metrics['rrmse'] = torch.cat((all_pf_metrics['rrmse'], rmse_val / rms_val))
            if calculate_crps:
                crps_val  = torch.mean(torch.stack(batch_crps_steps), dim=0)            # (B,)
                rcrps_val = crps_val / torch.mean(torch.norm(batch_v, p=2, dim=2), dim=0)
                all_pf_metrics['crps']  = torch.cat((all_pf_metrics['crps'],  crps_val))
                all_pf_metrics['rcrps'] = torch.cat((all_pf_metrics['rcrps'], rcrps_val))

            print("update results")

    # --- Save All Results to Cache File ---
    print(f"Saving PF results to: {cache_filepath}")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(all_pf_results_to_cache, cache_filepath)

    # --- Final Metrics Calculation (posterior-based) ---
    if all_pf_metrics['rrmse'].numel() == 0:
        return {}

    valid_B_mask = ~torch.isnan(all_pf_metrics['rrmse'])
    if not valid_B_mask.any():
        return {}

    mean_rrmse, std_rrmse = get_mean_std(all_pf_metrics['rrmse'][valid_B_mask])
    mean_rmse,  std_rmse  = get_mean_std(all_pf_metrics['rmse'][valid_B_mask])

    final_metrics = {
        'mean_rrmse': mean_rrmse,
        'std_rrmse':  std_rrmse,
        'mean_rmse':  mean_rmse,
        'std_rmse':   std_rmse,
    }

    if calculate_crps:
        mean_crps,  std_crps  = get_mean_std(all_pf_metrics['crps'][valid_B_mask])
        mean_rcrps, std_rcrps = get_mean_std(all_pf_metrics['rcrps'][valid_B_mask])
        final_metrics.update({
            'mean_crps':  mean_crps,
            'std_crps':   std_crps,
            'mean_rcrps': mean_rcrps,
            'std_rcrps':  std_rcrps,
        })

    final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_pf_metrics['rrmse'].numel() * 100.0

    return final_metrics


def test_model(loader, model_list, args, infl=1, H_info=None, plot_figures=True, fig_name='example_fig', save_pdf=False):
    """
    Runs the learned model-based assimilation and (optionally) compares with PF.

    NEW (timing): Records wall-clock time ONLY for the analysis step per i
    (on active trajectories), across all batches. Also records a trajectory-
    weighted variant that replicates each step duration by the number of
    active trajectories during that step. Appends mean/std (in seconds) to
    final_metrics with keys:
        - 'assim_step_time_mean', 'assim_step_time_std'
        - 'assim_step_time_mean_weighted', 'assim_step_time_std_weighted'
    """
    import os
    import time  # <-- NEW: timing
    import torch

    model, infl_model, local_model, st_model1, st_model2 = model_list
    m = args.N
    
    # Select forward function
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
    elif args.dataset == 'rossler':
        forward_fun = Rossler.forward
    elif args.dataset == "lorenz96":
        forward_fun = L96.forward
    elif args.dataset == "circle":
        forward_fun = CircleODE.forward
    elif args.dataset == "Hdoublewell":
        forward_fun = DoubleWellODE.forward
    elif args.dataset == "ks":
        if args.dt_iter <= 0:
            raise ValueError("args.dt_iter must be positive for KS model.")
        forward_fun = etd_rk4_wrapper(device=args.device, dt=args.dt / args.dt_iter)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    # Observation operator
    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info
    
    model.eval()
    if hasattr(infl_model, 'eval'): infl_model.eval()
    if hasattr(local_model, 'eval'): local_model.eval()
    if hasattr(st_model1, 'eval'): st_model1.eval()
    if hasattr(st_model2, 'eval'): st_model2.eval()
    
    # Optional PF cache
    cached_pf_data = None
    if args.pf_verification:
        first_batch_for_shape = next(iter(loader))
        traj_len = first_batch_for_shape.shape[0]
        batch_size = first_batch_for_shape.shape[1]
        
        cache_dir = os.path.join('data', args.dataset)
        cache_filename = f"pf_results_sigma_y_{args.sigma_y}_batch_{batch_size}_len_{traj_len}_pfN_{args.pf_N}_avg.pt"
        cache_filepath = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_filepath):
            print(f"Loading cached PF results from: {cache_filepath}")
            cached_pf_data = torch.load(cache_filepath, map_location=args.device, weights_only=True)
        else:
            raise FileNotFoundError(
                f"Required particle filter cache file not found at: {cache_filepath}. "
                f"Please run generate_and_cache_pf_results() first."
            )

    # Aggregated results (RMV removed)
    all_results = {
        'rmse': torch.empty(0, device=args.device),
        'rrmse': torch.empty(0, device=args.device),
        'crps': torch.empty(0, device=args.device),
        'rcrps': torch.empty(0, device=args.device),
        'cov_diff': torch.empty(0, device=args.device),
        'rcov_diff': torch.empty(0, device=args.device),
        'pf_rmse': torch.empty(0, device=args.device),
        'pf_rrmse': torch.empty(0, device=args.device),
    }
    loc_tensor_all_batches = None

    # NEW: analysis-step timing collectors
    assim_step_times = []              # per analysis step across batches
    assim_step_times_weighted = []     # replicated by #active trajectories

    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)  # [T, B, d]
            B = batch_v.shape[1]

            # ---- Active mask per-trajectory (B,) ----
            active_mask = torch.ones(B, dtype=torch.bool, device=args.device)

            # Initialize ensemble analysis at t0: [B, N, d]
            ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a += torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

            # Deactivate any trajectories that already contain NaNs after initialization
            init_nan = torch.isnan(ens_v_a).any(dim=(1, 2))
            if init_nan.any():
                active_mask = active_mask & (~init_nan)
                ens_v_a[~active_mask] = torch.nan  # keep shapes; mark inactive with NaN

            cov_diff_list, rcov_diff_list, pf_rmse_list = [], [], []
            ens_list = [ens_v_a]
            loc_records = []
            
            # Precompute noisy observations for all times (independent of active_mask)
            obs_y_list = []
            for i in range(len(batch_v)):
                obs_y_step = H_fun(batch_v[i].unsqueeze(1))
                obs_y_step += args.sigma_y * torch.randn_like(obs_y_step, device=args.device)
                obs_y_list.append(obs_y_step)

            # Time loop
            for i in range(len(batch_v) - 1):
                # Early bail: if no active trajectories remain, append a NaN step to keep time length and continue
                if not active_mask.any():
                    ens_list.append(torch.full_like(ens_v_a, float('nan')))
                    continue  # keep T length for plotting/metrics

                # -------- Forecast step (not timed here) --------
                ens_v_a_forecast_input = ens_v_a.view(-1, args.ori_dim)  # [B*N, d]
                for j_iter in range(args.dt_iter):
                    if args.dataset == 'ks':
                        ens_v_a_forecast_input = forward_fun(ens_v_a_forecast_input, None, args.dt / args.dt_iter)
                    else:
                        current_time_for_rk4 = i * args.dt + j_iter * (args.dt / args.dt_iter)
                        ens_v_a_forecast_input = rk4(forward_fun, ens_v_a_forecast_input, current_time_for_rk4, args.dt / args.dt_iter)
                ens_v_f = ens_v_a_forecast_input.view(-1, m, args.ori_dim)
                ens_v_f += torch.randn_like(ens_v_f, device=args.device) * args.sigma_v

                # Deactivate offending trajectories that became NaN after forecast
                nan_now = torch.isnan(ens_v_f).any(dim=(1, 2))
                if nan_now.any():
                    active_mask = active_mask & (~nan_now)
                    ens_v_f[~active_mask] = torch.nan

                # Active indices
                idx_active = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)

                # Prepare container for next analysis; default NaNs for inactive
                ens_v_a_next = torch.full_like(ens_v_f, float('nan'))

                if idx_active.numel() > 0:
                    # Slice active trajectories only
                    ens_v_f_active = ens_v_f.index_select(0, idx_active)                 # [B_active, N, d]
                    obs_y = obs_y_list[i + 1]                                            # [T,B,?] -> [B,1,d_obs]
                    obs_y_active = obs_y.index_select(0, idx_active)                     # [B_active, 1, d_obs]

                    # ------ BEGIN TIMED ANALYSIS STEP ------
                    t0 = time.perf_counter()

                    # Compute H(ens) and innovation on active subset
                    hv_active = H_fun(ens_v_f_active)                                    # [B_active, N, d_obs]
                    r_noise_active = mean0(args.sigma_y * torch.randn_like(hv_active, device=args.device))
                    ens_i_innov_active = obs_y_active - hv_active - r_noise_active

                    # Means on active subset
                    mean_hv_active = torch.mean(hv_active, dim=1, keepdim=True)
                    mean_ens_v_f_active = torch.mean(ens_v_f_active, dim=1, keepdim=True)

                    # Analysis step (delegated to learned components)
                    ens_v_a_active, loc_nn_output = _process_analysis_step(
                        args, model_list, ens_v_f_active, hv_active, obs_y_active, ens_i_innov_active,
                        mean_ens_v_f_active, mean_hv_active
                    )

                    # Record localization-related outputs if any
                    if loc_nn_output is not None:
                        loc_records.append(loc_nn_output)

                    # Inflation / post-processing on active subset only
                    ens_v_a_active = post_process(ens_v_a_active, infl=infl)

                    # ------ END TIMED ANALYSIS STEP ------
                    t1 = time.perf_counter()
                    duration = t1 - t0
                    assim_step_times.append(duration)
                    num_active = int(idx_active.numel())
                    if num_active > 0:
                        assim_step_times_weighted.extend([duration] * num_active)

                    # Scatter active results back; inactive remain NaN
                    ens_v_a_next.index_copy_(0, idx_active, ens_v_a_active)

                # Move to next analysis state
                ens_v_a = ens_v_a_next
                ens_list.append(ens_v_a)

                # PF verification: compute only for current active subset, scatter back
                if args.pf_verification and idx_active.numel() > 0:
                    pf_mean_a_full = cached_pf_data[batch_ind]['means'][i]     # [B, d]
                    pf_cov_ens_a_full = cached_pf_data[batch_ind]['covs'][i]   # [B, d, d]

                    pf_mean_a = pf_mean_a_full.index_select(0, idx_active)
                    pf_cov_ens_a = pf_cov_ens_a_full.index_select(0, idx_active)

                    our_method_mean_a = torch.mean(ens_v_a.index_select(0, idx_active), dim=1)  # [B_active, d]
                    pf_rmse_active = torch.sqrt(torch.mean((our_method_mean_a - pf_mean_a)**2, dim=-1))  # [B_active]

                    # Optional dataset-specific masking (rossler)
                    if args.dataset == 'rossler':
                        cond = batch_v[i + 1, :, 2] <= 5
                        cond_active = cond.index_select(0, idx_active)
                        pf_rmse_active = pf_rmse_active.masked_fill(cond_active, float('nan'))

                    # Scatter PF RMSE back to full B vector
                    pf_rmse_full = torch.full((B,), float('nan'), device=args.device)
                    pf_rmse_full.index_copy_(0, idx_active, pf_rmse_active)
                    pf_rmse_list.append(pf_rmse_full)

                    # Covariance diffs on active subset and scatter back
                    cov_ens_a_active = get_ens_cov(ens_v_a.index_select(0, idx_active))     # [B_active, d, d]
                    cov_diff_active = torch.norm(cov_ens_a_active - pf_cov_ens_a, p='fro', dim=(-2, -1))
                    rcov_diff_active = cov_diff_active / torch.norm(pf_cov_ens_a, p='fro', dim=(-2, -1))

                    if args.dataset == 'rossler':
                        cond_active = cond.index_select(0, idx_active)
                        cov_diff_active = cov_diff_active.masked_fill(cond_active, float('nan'))
                        rcov_diff_active = rcov_diff_active.masked_fill(cond_active, float('nan'))

                    cov_diff_full = torch.full((B,), float('nan'), device=args.device)
                    rcov_diff_full = torch.full((B,), float('nan'), device=args.device)
                    cov_diff_full.index_copy_(0, idx_active, cov_diff_active)
                    rcov_diff_full.index_copy_(0, idx_active, rcov_diff_active)
                    cov_diff_list.append(cov_diff_full)
                    rcov_diff_list.append(rcov_diff_full)

            # Stack ensembles over time: [T, B, N, d]
            ens_tensor = torch.stack(ens_list)
            
            # Metrics (RMV removed; NaNs are handled later by masks)
            crps_val = torch.mean(compute_es(ens_states=ens_tensor, true_states=batch_v, norm_p=1), dim=0)
            rcrps_val = crps_val / torch.mean(torch.norm(batch_v, p=2, dim=2), dim=0)
            rmse_val = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)), dim=0)
            rms_val = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0)
            rrmse_val = rmse_val / rms_val
            
            # Aggregate metrics
            all_results['rmse'] = torch.cat((all_results['rmse'], rmse_val))
            all_results['rrmse'] = torch.cat((all_results['rrmse'], rrmse_val))
            all_results['crps'] = torch.cat((all_results['crps'], crps_val))
            all_results['rcrps'] = torch.cat((all_results['rcrps'], rcrps_val))
            if args.pf_verification and len(pf_rmse_list) > 0:
                # Use nanmean over time for PF-based metrics
                all_results['cov_diff'] = torch.cat((all_results['cov_diff'], torch.stack(cov_diff_list).nanmean(0)))
                all_results['rcov_diff'] = torch.cat((all_results['rcov_diff'], torch.stack(rcov_diff_list).nanmean(0)))
                all_results['pf_rmse'] = torch.cat((all_results['pf_rmse'], torch.stack(pf_rmse_list).nanmean(0)))
                all_results['pf_rrmse'] = torch.cat((all_results['pf_rrmse'], torch.stack(pf_rmse_list).nanmean(0) / rms_val))
            
            # Localization tensor collection (unchanged)
            if args.v != "EtE" and not args.no_localization and len(loc_records) > 0:
                current_loc_tensor = torch.stack(loc_records).unsqueeze(0)
                if loc_tensor_all_batches is None:
                    loc_tensor_all_batches = current_loc_tensor
                else:
                    try:
                        loc_tensor_all_batches = torch.cat((loc_tensor_all_batches, current_loc_tensor), dim=0)
                    except Exception as e:
                        print(f"Warning: Could not concatenate loc_tensors due to shape mismatch or other error: {e}")

            # Build observation tensor for plotting (unchanged)
            obs_tensor = torch.stack(obs_y_list).squeeze(2)
            observations = torch.full_like(batch_v, float('nan'), device=args.device)
            if hasattr(args, 'obs_inds') and args.obs_inds is not None:
                observations[:, :, args.obs_inds] = obs_tensor
            else:
                print("Warning: args.obs_inds not defined. Observations tensor might be all NaNs.")
    
    # Plotting (unchanged)
    if plot_figures: 
        time_idx_plot = -2
        num_dims_plot = 4
        dim_indices_plot = list(range(min(args.ori_dim, num_dims_plot)))

        if args.pf_verification:
            batch_v = cached_pf_data[-1]['means']
            ens_tensor = ens_tensor[1:]
            observations = observations[1:]
        plot_particle_trajectories_with_histograms(
            particles=ens_tensor[:, time_idx_plot, :, :], 
            true_traj=batch_v[:, time_idx_plot, :], 
            observation=None, 
            dim_indices=dim_indices_plot,
            start_time=0,
            end_time=ens_tensor.shape[0],
            mode='quantile',
            save_fig=True,
            save_pdf=save_pdf,
            save_name=fig_name + "_hist",
            hist_step=1,
            fontsize=None
        )
        plot_particle_trajectories(
            particles=ens_tensor[:, time_idx_plot, :, :], 
            true_traj=batch_v[:, time_idx_plot, :], 
            observation=observations[:, time_idx_plot, :],
            cmap_name='bwr',
            start_time=0,
            end_time=ens_tensor.shape[0], 
            main_fig_size=(5, 2), 
            save_fig=True,
            save_pdf=save_pdf,
            save_name=fig_name + "_traj",
            colorbar_range=args.colorbar_range if hasattr(args, 'colorbar_range') else None,
            plot_vertical_colorbar=False,
            plot_horizontal_colorbar=True
        )

    # Final metrics (RMV removed)
    final_metrics = {}
    
    if all_results['rrmse'].numel() == 0:
        metrics_keys = ['mean_rmse', 'std_rmse',
                        'mean_rrmse', 'std_rrmse', 'mean_crps', 'std_crps',
                        'mean_rcrps', 'std_rcrps', 'mean_cov_diff', 'std_cov_diff',
                        'mean_rcov_diff', 'std_rcov_diff', 'mean_pf_rmse', 'std_pf_rmse',
                        'mean_pf_rrmse', 'std_pf_rrmse']
        final_metrics = {key: float('nan') for key in metrics_keys}
        final_metrics['no_nan_percent'] = 0.0
    else:
        nan_mask = torch.isnan(all_results['rrmse'])
        valid_B_mask = ~nan_mask
        
        if not valid_B_mask.any():
            metrics_keys = ['mean_rmse', 'std_rmse',
                            'mean_rrmse', 'std_rrmse', 'mean_crps', 'std_crps',
                            'mean_rcrps', 'std_rcrps', 'mean_cov_diff', 'std_cov_diff',
                            'mean_rcov_diff', 'std_rcov_diff', 'mean_pf_rmse', 'std_pf_rmse',
                            'mean_pf_rrmse', 'std_pf_rrmse']
            final_metrics = {key: float('nan') for key in metrics_keys}
            final_metrics['no_nan_percent'] = 0.0
        else:
            final_metrics['mean_rrmse'], final_metrics['std_rrmse'] = get_mean_std(all_results['rrmse'][valid_B_mask])
            final_metrics['mean_rmse'], final_metrics['std_rmse'] = get_mean_std(all_results['rmse'][valid_B_mask])
            final_metrics['mean_crps'], final_metrics['std_crps'] = get_mean_std(all_results['crps'][valid_B_mask])
            final_metrics['mean_rcrps'], final_metrics['std_rcrps'] = get_mean_std(all_results['rcrps'][valid_B_mask])
            final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_results['rrmse'].numel() * 100.0
            if args.pf_verification:
                final_metrics['mean_cov_diff'], final_metrics['std_cov_diff'] = get_mean_std(all_results['cov_diff'][valid_B_mask])
                final_metrics['mean_rcov_diff'], final_metrics['std_rcov_diff'] = get_mean_std(all_results['rcov_diff'][valid_B_mask])
                final_metrics['mean_pf_rmse'], final_metrics['std_pf_rmse'] = get_mean_std(all_results['pf_rmse'][valid_B_mask])
                final_metrics['mean_pf_rrmse'], final_metrics['std_pf_rrmse'] = get_mean_std(all_results['pf_rrmse'][valid_B_mask])

    # ---- NEW: compute analysis-step timing stats and attach to final_metrics ----
    if len(assim_step_times) > 0:
        t_tensor = torch.tensor(assim_step_times, dtype=torch.float64)
        final_metrics['assim_step_time_mean'] = float(t_tensor.mean().item())
        final_metrics['assim_step_time_std']  = float(t_tensor.std(unbiased=False).item())
    else:
        final_metrics['assim_step_time_mean'] = float('nan')
        final_metrics['assim_step_time_std']  = float('nan')

    if len(assim_step_times_weighted) > 0:
        tw_tensor = torch.tensor(assim_step_times_weighted, dtype=torch.float64)
        final_metrics['assim_step_time_mean_weighted'] = float(tw_tensor.mean().item())
        final_metrics['assim_step_time_std_weighted']  = float(tw_tensor.std(unbiased=False).item())
    else:
        final_metrics['assim_step_time_mean_weighted'] = float('nan')
        final_metrics['assim_step_time_std_weighted']  = float('nan')

    # Return one localization tensor (unchanged)
    final_loc_tensor_to_return = loc_tensor_all_batches[0] if loc_tensor_all_batches is not None and loc_tensor_all_batches.shape[0] > 0 else torch.empty(1, device=args.device)
    final_metrics['loc_tensor'] = final_loc_tensor_to_return

    return final_metrics




def print_test_results(results):
    """Pretty-print test metrics including optional timing stats (times in ms)."""
    # Core metrics
    if 'mean_rmse' in results and 'std_rmse' in results:
        print(f"RMSE: {results['mean_rmse']:.3f} ± {results['std_rmse']:.3f}")
    if 'mean_rrmse' in results and 'std_rrmse' in results:
        print(f"RRMSE: {results['mean_rrmse']:.3f} ± {results['std_rrmse']:.3f}")
    if 'mean_crps' in results and 'std_crps' in results:
        print(f"CRPS: {results['mean_crps']:.3f} ± {results['std_crps']:.3f}")
    if 'mean_rcrps' in results and 'std_rcrps' in results:
        print(f"RCRPS: {results['mean_rcrps']:.3f} ± {results['std_rcrps']:.3f}")

    # Optional PF verification metrics
    if 'mean_cov_diff' in results and 'std_cov_diff' in results:
        print(f"Cov-Diff: {results['mean_cov_diff']:.3f} ± {results['std_cov_diff']:.3f}")
    if 'mean_rcov_diff' in results and 'std_rcov_diff' in results:
        print(f"RCov-Diff: {results['mean_rcov_diff']:.3f} ± {results['std_rcov_diff']:.3f}")
    if 'mean_pf_rmse' in results and 'std_pf_rmse' in results:
        print(f"PF-RMSE: {results['mean_pf_rmse']:.3f} ± {results['std_pf_rmse']:.3f}")
    if 'mean_pf_rrmse' in results and 'std_pf_rrmse' in results:
        print(f"PF-RRMSE: {results['mean_pf_rrmse']:.3f} ± {results['std_pf_rrmse']:.3f}")

    # Percentage of non-NaN trajectories
    if 'no_nan_percent' in results:
        print(f"No NAN Percentage: {results['no_nan_percent']:.2f}%")

    # --- Timing outputs in milliseconds (ms) ---

    # Unweighted per-step assimilation time (originally in seconds)
    if 'assim_step_time_mean' in results and 'assim_step_time_std' in results:
        mean_ms = results['assim_step_time_mean'] * 1000.0
        std_ms  = results['assim_step_time_std'] * 1000.0
        print(f"Assim-step time (ms): {mean_ms:.3f} ± {std_ms:.3f}")

    # Trajectory-weighted per-step assimilation time
    if 'assim_step_time_mean_weighted' in results and 'assim_step_time_std_weighted' in results:
        mean_ms_w = results['assim_step_time_mean_weighted'] * 1000.0
        std_ms_w  = results['assim_step_time_std_weighted'] * 1000.0
        print(f"Assim-step time (ms, weighted): {mean_ms_w:.3f} ± {std_ms_w:.3f}")





def test_ClassicFilter(loader, args, infl=1, H_info=None, plot_figures=True, fig_name='example_fig', loc_radius=None, save_pdf=False):
    """
    Tests a classic data assimilation filter (e.g., EnKF, ESRF, LETKF) and
    optionally compares results against a particle filter baseline.

    NOTE (added): If any trajectory (a specific b in the batch dimension) ever
    produces NaNs at any step, that trajectory is marked inactive and skipped
    for all subsequent steps. We keep shape consistency by writing NaNs for its
    outputs so that downstream metrics (which already NaN-mask) ignore it.

    CHANGE: All RMV-related computation and metrics were removed to avoid OOM.

    NEW (timing): Records wall-clock time for each assimilation step (per i),
    aggregates across all batches. Also records a trajectory-weighted variant
    that replicates each step duration by the number of active trajectories at
    that step. Returns mean/std for both.
    """
    import os
    import time  # <-- NEW: timing
    import torch
    from tqdm import tqdm

    m = args.N

    # Select forward function
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
        model_propagator = rk4
    elif args.dataset == "lorenz96":
        forward_fun = L96.forward
        model_propagator = rk4
    elif args.dataset == 'rossler':
        forward_fun = Rossler.forward
        model_propagator = rk4
    elif args.dataset == "circle":
        forward_fun = CircleODE.forward
        model_propagator = rk4
    elif args.dataset == "Hdoublewell":
        forward_fun = DoubleWellODE.forward
        model_propagator = rk4
    elif args.dataset == "ks":
        if args.dt_iter <= 0:
            raise ValueError("args.dt_iter must be positive for KS model.")
        forward_fun = None
        model_propagator = lambda func, u, t, dt: etd_rk4_wrapper(device=args.device, dt=dt)(u, None, dt)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    # Observation operator
    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info

    if args.no_localization:
        print("Do not use localization")
        loc_radius = None

    # Optional PF cache
    cached_pf_data = None
    if args.pf_verification:
        first_batch_for_shape = next(iter(loader))
        traj_len = first_batch_for_shape.shape[0]
        batch_size = first_batch_for_shape.shape[1]

        cache_dir = os.path.join('data', args.dataset)
        cache_filename = f"pf_results_sigma_y_{args.sigma_y}_batch_{batch_size}_len_{traj_len}_pfN_{args.pf_N}_avg.pt"
        cache_filepath = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_filepath):
            print(f"Loading cached PF results from: {cache_filepath}")
            cached_pf_data = torch.load(cache_filepath, map_location=args.device, weights_only=True)
        else:
            raise FileNotFoundError(
                f"Required particle filter cache file not found at: {cache_filepath}. "
                f"Please run generate_and_cache_pf_results() first."
            )

    # Aggregated results (RMV removed)
    all_results = {
        'rmse': torch.empty(0, device=args.device),
        'rrmse': torch.empty(0, device=args.device),
        'crps': torch.empty(0, device=args.device),
        'rcrps': torch.empty(0, device=args.device),
        'cov_diff': torch.empty(0, device=args.device),
        'rcov_diff': torch.empty(0, device=args.device),
        'pf_rmse': torch.empty(0, device=args.device),
        'pf_rrmse': torch.empty(0, device=args.device),
    }

    # NEW: timing collectors
    assim_step_times = []              # per-step durations across all batches
    assim_step_times_weighted = []     # per-trajectory-weighted durations

    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)  # [T, B, d]

            # ---- NEW: active mask per-trajectory (B,) ----
            B = batch_v.shape[1]
            active_mask = torch.ones(B, dtype=torch.bool, device=args.device)

            # Initialize ensemble analysis at t0: [B, N, d]
            ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a += torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

            # Detect NaN at initialization and deactivate those trajectories
            init_nan = torch.isnan(ens_v_a).any(dim=(1, 2))
            if init_nan.any():
                active_mask = active_mask & (~init_nan)
                # Fill NaNs for the deactivated ones to keep shapes consistent
                ens_v_a[~active_mask] = torch.nan

            ens_list = [ens_v_a]
            cov_diff_list, rcov_diff_list, pf_rmse_list = [], [], []

            # Precompute noisy observations y_t for all t (shape aligned to H_fun outputs)
            obs_y_list = []
            for i in range(len(batch_v)):
                obs_y_step = H_fun(batch_v[i].unsqueeze(1))
                obs_y_step += args.sigma_y * torch.randn_like(obs_y_step, device=args.device)
                obs_y_list.append(obs_y_step)

            # Time loop
            for i in tqdm(range(len(batch_v) - 1), desc="Processing", unit="item"):
                # Early exit if no active trajectories remain
                if not active_mask.any():
                    ens_list.append(torch.full_like(ens_v_a, float('nan')))
                    # Record a zero-duration placeholder for clarity? No: skip to avoid bias
                    continue

                # ---- NEW: start timing the entire assimilation step (forecast + analysis + book-keeping) ----
                t0 = time.perf_counter()

                obs_y = obs_y_list[i + 1]  # [B, 1, d_obs]

                # Forecast step
                if args.v.startswith('iEnKS'):
                    ens_v_f = ens_v_a  # smoothing will handle propagation
                else:
                    ens_v_a_forecast_input = ens_v_a.view(-1, args.ori_dim)  # [B*N, d]
                    for j_iter in range(args.dt_iter):
                        current_time_for_rk4 = i * args.dt + j_iter * (args.dt / args.dt_iter)
                        ens_v_a_forecast_input = model_propagator(forward_fun, ens_v_a_forecast_input, current_time_for_rk4, args.dt / args.dt_iter)
                    ens_v_f = ens_v_a_forecast_input.view(-1, m, args.ori_dim)
                    ens_v_f += torch.randn_like(ens_v_f, device=args.device) * args.sigma_v

                # ---- NEW: detect NaNs in forecast and deactivate offending trajectories ----
                nan_now = torch.isnan(ens_v_f).any(dim=(1, 2))
                if nan_now.any():
                    active_mask = active_mask & (~nan_now)
                    ens_v_f[~active_mask] = torch.nan

                # Shapes
                B_shape, N_ens, D_state = ens_v_f.shape
                d_obs_shape = obs_y.shape[2]

                # Common args
                common_enkf_args = {
                    "observation_y": None,  # will be filled with subset
                    "observation_operator_ens": H_fun,
                    "sigma_y": args.sigma_y,
                    "sigma_v": args.sigma_v,
                    "inflation_factor": infl
                }

                # ---- operate only on active indices for the analysis step ----
                idx_active = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
                ens_v_a_next = torch.full_like(ens_v_f, float('nan'))  # default NaN for inactive

                if idx_active.numel() > 0:
                    ens_v_f_active = ens_v_f.index_select(0, idx_active)
                    obs_y_active = obs_y.squeeze(1).index_select(0, idx_active)
                    common_enkf_args["observation_y"] = obs_y_active

                    if args.v == 'EnKF':
                        loc_vy = dist2coeff(args.Lvy, radius=loc_radius).unsqueeze(0) if loc_radius is not None else None
                        loc_yy = dist2coeff(args.Lyy, radius=loc_radius).unsqueeze(0) if loc_radius is not None else None
                        ens_v_a_active, _ = ensemble_kalman_filter_analysis(
                            ens_v_f_active, **common_enkf_args, method='EnKF-PertObs',
                            localization_matrix_Lxy=loc_vy, localization_matrix_Lyy=loc_yy)

                    elif args.v == 'ESRF':
                        loc_vy = dist2coeff(args.Lvy, radius=loc_radius).unsqueeze(0) if loc_radius is not None else None
                        loc_yy = dist2coeff(args.Lyy, radius=loc_radius).unsqueeze(0) if loc_radius is not None else None
                        ens_v_a_active, _ = ensemble_kalman_filter_analysis(
                            ens_v_f_active, **common_enkf_args, method='ESRF',
                            localization_matrix_Lxy=loc_vy, localization_matrix_Lyy=loc_yy)

                    elif args.v == 'LETKF':
                        coords_state = torch.arange(D_state, device=args.device, dtype=batch_v.dtype).unsqueeze(1)
                        if hasattr(args, 'obs_inds') and args.obs_inds is not None:
                            coords_obs = torch.as_tensor(args.obs_inds, device=args.device, dtype=batch_v.dtype).unsqueeze(1)
                        else:
                            coords_obs = torch.linspace(0, D_state-1, steps=d_obs_shape, device=args.device, dtype=batch_v.dtype).long().unsqueeze(1)
                        domain = torch.tensor([D_state], device=args.device, dtype=batch_v.dtype)
                        ens_v_a_active, _ = ensemble_kalman_filter_analysis(
                            ens_v_f_active, **common_enkf_args, method='LETKF',
                            localization_radius=loc_radius, coords_state=coords_state,
                            coords_obs=coords_obs, localization_domain=domain)

                    elif args.v.startswith('iEnKS'):
                        coords_state = torch.arange(D_state, device=args.device, dtype=batch_v.dtype).unsqueeze(1)
                        if hasattr(args, 'obs_inds') and args.obs_inds is not None:
                            coords_obs = torch.as_tensor(args.obs_inds, device=args.device, dtype=batch_v.dtype).unsqueeze(1)
                        else:
                            coords_obs = torch.linspace(0, D_state-1, steps=d_obs_shape, device=args.device, dtype=batch_v.dtype).long().unsqueeze(1)
                        domain = torch.tensor([D_state], device=args.device, dtype=batch_v.dtype)

                        model_args_ienks = {
                            "propagator": model_propagator,
                            "rhs": forward_fun,
                            "dt": args.dt / args.dt_iter,
                            "steps_between_analyses": args.dt_iter,
                        }
                        E_smoothed_at_start, _ = ensemble_kalman_filter_analysis(
                            ens_v_f_active,
                            **common_enkf_args,
                            method=args.v,
                            localization_radius=loc_radius, 
                            coords_state=coords_state,
                            coords_obs=coords_obs, 
                            localization_domain=domain,
                            ienks_lag=1,
                            ienks_niter=10,
                            ienks_wtol=1e-5,
                            model_args=model_args_ienks
                        )

                        # Forecast from smoothed start to analysis time
                        ens_v_a_forecast_input = E_smoothed_at_start.clone().view(-1, args.ori_dim)
                        for j_iter in range(args.dt_iter):
                            current_time_for_rk4 = i * args.dt + j_iter * (args.dt / args.dt_iter)
                            ens_v_a_forecast_input = model_propagator(forward_fun, ens_v_a_forecast_input, current_time_for_rk4, args.dt / args.dt_iter)
                        ens_v_a_active = ens_v_a_forecast_input.view(-1, m, args.ori_dim)
                        ens_v_a_active += torch.randn_like(ens_v_a_active, device=args.device) * args.sigma_v

                    else:
                        raise NotImplementedError(f"The filter {args.v} is not implemented")

                    # Clamp only on active subset
                    ens_v_a_active = torch.clamp(ens_v_a_active, min=-args.clamp, max=args.clamp)

                    # Scatter back active results into full tensor; inactive stay NaN
                    ens_v_a_next.index_copy_(0, idx_active, ens_v_a_active)

                # Move to next analysis state (NaN for inactive, valid values for active)
                ens_v_a = ens_v_a_next

                # Append for metrics (inactive remain NaN)
                ens_list.append(ens_v_a)

                # PF verification (compute only on currently active trajectories)
                if args.pf_verification and idx_active.numel() > 0:
                    pf_mean_a_full = cached_pf_data[batch_ind]['means'][i]         # [B, d]
                    pf_cov_ens_a_full = cached_pf_data[batch_ind]['covs'][i]      # [B, d, d]

                    # Subset PF data to active ones
                    pf_mean_a = pf_mean_a_full.index_select(0, idx_active)
                    pf_cov_ens_a = pf_cov_ens_a_full.index_select(0, idx_active)

                    our_method_mean_a = torch.mean(ens_v_a.index_select(0, idx_active), dim=1)  # [B_active, d]
                    pf_rmse = torch.sqrt(torch.mean((our_method_mean_a - pf_mean_a) ** 2, dim=-1))  # [B_active]
                    pf_rmse_full = torch.full((B,), float('nan'), device=args.device)
                    pf_rmse_full.index_copy_(0, idx_active, pf_rmse)
                    pf_rmse_list.append(pf_rmse_full)

                    cov_ens_a = get_ens_cov(ens_v_a.index_select(0, idx_active))  # [B_active, d, d]
                    cov_diff_active = torch.norm(cov_ens_a - pf_cov_ens_a, p='fro', dim=(-2, -1))
                    rcov_diff_active = cov_diff_active / torch.norm(pf_cov_ens_a, p='fro', dim=(-2, -1))
                    cov_diff_full = torch.full((B,), float('nan'), device=args.device)
                    rcov_diff_full = torch.full((B,), float('nan'), device=args.device)
                    cov_diff_full.index_copy_(0, idx_active, cov_diff_active)
                    rcov_diff_full.index_copy_(0, idx_active, rcov_diff_active)
                    cov_diff_list.append(cov_diff_full)
                    rcov_diff_list.append(rcov_diff_full)

                # ---- NEW: stop timing and record ----
                t1 = time.perf_counter()
                duration = t1 - t0  # seconds
                assim_step_times.append(duration)
                # Weight by #active trajectories at this step
                num_active = int(idx_active.numel()) if 'idx_active' in locals() else 0
                if num_active > 0:
                    assim_step_times_weighted.extend([duration] * num_active)
                # If no active, we skip weighting to avoid bias

            # Stack ensembles over time: [T, B, N, d]
            ens_tensor = torch.stack(ens_list)

            crps_val = torch.mean(compute_es(ens_states=ens_tensor, true_states=batch_v, norm_p=1), dim=0)
            rcrps_val = crps_val / torch.mean(torch.norm(batch_v, p=2, dim=2), dim=0)
            rmse_val = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)), dim=0)
            rms_val = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0)
            rrmse_val = rmse_val / rms_val

            all_results['rmse'] = torch.cat((all_results['rmse'], rmse_val))
            all_results['rrmse'] = torch.cat((all_results['rrmse'], rrmse_val))
            all_results['crps'] = torch.cat((all_results['crps'], crps_val))
            all_results['rcrps'] = torch.cat((all_results['rcrps'], rcrps_val))
            if args.pf_verification and len(pf_rmse_list) > 0:
                all_results['cov_diff'] = torch.cat((all_results['cov_diff'], torch.stack(cov_diff_list).mean(0)))
                all_results['rcov_diff'] = torch.cat((all_results['rcov_diff'], torch.stack(rcov_diff_list).mean(0)))
                all_results['pf_rmse'] = torch.cat((all_results['pf_rmse'], torch.stack(pf_rmse_list).mean(0)))
                all_results['pf_rrmse'] = torch.cat((all_results['pf_rrmse'], torch.stack(pf_rmse_list).nanmean(0) / rms_val))
            
            # Build observation tensor for plotting (unchanged)
            obs_tensor = torch.stack(obs_y_list).squeeze(2)
            observations = torch.full_like(batch_v, float('nan'), device=args.device)
            if hasattr(args, 'obs_inds') and args.obs_inds is not None:
                observations[:, :, args.obs_inds] = obs_tensor
            else:
                print("Warning: args.obs_inds not defined. Observations tensor might be all NaNs.")

    # Plotting (unchanged)
    if plot_figures: 
        time_idx_plot = -2
        num_dims_plot = 4
        dim_indices_plot = list(range(min(args.ori_dim, num_dims_plot)))

        if args.pf_verification:
            batch_v = cached_pf_data[-1]['means']
            ens_tensor = ens_tensor[1:]
            observations = observations[1:]

        plot_particle_trajectories_with_histograms(
            particles=ens_tensor[:, time_idx_plot, :, :],
            true_traj=batch_v[:, time_idx_plot, :],
            observation=None,
            dim_indices=dim_indices_plot,
            start_time=0, end_time=ens_tensor.shape[0], mode='quantile',
            save_fig=True, save_pdf=save_pdf, save_name=fig_name + "_hist_classic",
            hist_step=1, fontsize=None)

        plot_particle_trajectories(
            particles=ens_tensor[:, time_idx_plot, :, :],
            true_traj=batch_v[:, time_idx_plot, :],
            observation=observations[:, time_idx_plot, :],
            cmap_name='bwr', start_time=0, end_time=ens_tensor.shape[0],
            main_fig_size=(5, 2), save_fig=True, save_pdf=save_pdf,
            save_name=fig_name + "_traj_classic",
            colorbar_range=args.colorbar_range if hasattr(args, 'colorbar_range') else None,
            plot_vertical_colorbar=False, plot_horizontal_colorbar=True)

    final_metrics = {}
    if all_results['rrmse'].numel() == 0:
        metrics_keys = ['mean_rmse', 'std_rmse', 'mean_rrmse', 'std_rrmse',
                        'mean_crps', 'std_crps', 'mean_rcrps', 'std_rcrps',
                        'mean_cov_diff', 'std_cov_diff', 'mean_rcov_diff', 'std_rcov_diff',
                        'mean_pf_rmse', 'std_pf_rmse', 'mean_pf_rrmse', 'std_pf_rrmse']
        final_metrics = {key: float('nan') for key in metrics_keys}
        final_metrics['no_nan_percent'] = 0.0
    else:
        nan_mask = torch.isnan(all_results['rrmse'])
        valid_B_mask = ~nan_mask

        if not valid_B_mask.any():
            metrics_keys = ['mean_rmse', 'std_rmse', 'mean_rrmse', 'std_rrmse',
                            'mean_crps', 'std_crps', 'mean_rcrps', 'std_rcrps',
                            'mean_cov_diff', 'std_cov_diff', 'mean_rcov_diff', 'std_rcov_diff',
                            'mean_pf_rmse', 'std_pf_rmse', 'mean_pf_rrmse', 'std_pf_rrmse']
            final_metrics = {key: float('nan') for key in metrics_keys}
            final_metrics['no_nan_percent'] = 0.0
        else:
            final_metrics['mean_rrmse'], final_metrics['std_rrmse'] = get_mean_std(all_results['rrmse'][valid_B_mask])
            final_metrics['mean_rmse'], final_metrics['std_rmse'] = get_mean_std(all_results['rmse'][valid_B_mask])
            final_metrics['mean_crps'], final_metrics['std_crps'] = get_mean_std(all_results['crps'][valid_B_mask])
            final_metrics['mean_rcrps'], final_metrics['std_rcrps'] = get_mean_std(all_results['rcrps'][valid_B_mask])
            final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_results['rrmse'].numel() * 100.0
            if args.pf_verification:
                final_metrics['mean_cov_diff'], final_metrics['std_cov_diff'] = get_mean_std(all_results['cov_diff'][valid_B_mask])
                final_metrics['mean_rcov_diff'], final_metrics['std_rcov_diff'] = get_mean_std(all_results['rcov_diff'][valid_B_mask])
                final_metrics['mean_pf_rmse'], final_metrics['std_pf_rmse'] = get_mean_std(all_results['pf_rmse'][valid_B_mask])
                final_metrics['mean_pf_rrmse'], final_metrics['std_pf_rrmse'] = get_mean_std(all_results['pf_rrmse'][valid_B_mask])

    # ---- NEW: compute timing stats and attach to final_metrics ----
    # We compute (1) per-step mean/std and (2) trajectory-weighted mean/std.
    if len(assim_step_times) > 0:
        t_tensor = torch.tensor(assim_step_times, dtype=torch.float64)
        final_metrics['assim_step_time_mean'] = float(t_tensor.mean().item())               # seconds
        final_metrics['assim_step_time_std']  = float(t_tensor.std(unbiased=False).item())  # population std
    else:
        final_metrics['assim_step_time_mean'] = float('nan')
        final_metrics['assim_step_time_std']  = float('nan')

    if len(assim_step_times_weighted) > 0:
        tw_tensor = torch.tensor(assim_step_times_weighted, dtype=torch.float64)
        final_metrics['assim_step_time_mean_weighted'] = float(tw_tensor.mean().item())
        final_metrics['assim_step_time_std_weighted']  = float(tw_tensor.std(unbiased=False).item())
    else:
        final_metrics['assim_step_time_mean_weighted'] = float('nan')
        final_metrics['assim_step_time_std_weighted']  = float('nan')

    return final_metrics





def train_model_v2(epoch, loader, model_list, optimizer, scheduler, args):
    """
    Function to train the model for one epoch (V2 with linear dynamics).

    Args:
        epoch (int): The current epoch number.
        loader (DataLoader): The data loader for training data.
        model_list (list): A list of models to be trained.
        optimizer (Optimizer): The optimizer for updating model weights.
        scheduler (Scheduler): The learning rate scheduler.
        args (Namespace): A namespace containing all arguments/hyperparameters.

    Returns:
        float: The average loss for the epoch.
    """
    model, infl_model, local_model, st_model1, st_model2 = model_list
    N = args.N
    losses = AverageMeter()
    batch_time = AverageMeter()
    
    # --- Model and Observation Operator ---
    if args.dataset == "linear":
        forward_fun = lambda x, A: torch.bmm(x, A.transpose(-1,-2))
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")
    H_fun = lambda x, H: torch.bmm(x, H.transpose(-1,-2))
    
    # --- Set Models to Train Mode ---
    model.train()
    if hasattr(infl_model, 'train'): infl_model.train()
    if hasattr(local_model, 'train'): local_model.train()
    if hasattr(st_model1, 'train'): st_model1.train()
    if hasattr(st_model2, 'train'): st_model2.train()

    all_trainable_params = []
    for m_ in [model, infl_model, local_model, st_model1, st_model2]:
        if hasattr(m_, 'parameters') and not isinstance(m_, NaiveNetwork):
            all_trainable_params.extend(list(filter(lambda p: p.requires_grad, m_.parameters())))

    num_batches_all_nan = 0

    # --- Batch Loop ---
    for batch_ind, batch_info in enumerate(loader):
        t_start = time.time()
        
        m, A, C, H, sigma_v, sigma_y = (batch_info['m'], batch_info['A'], batch_info['C'], 
                                        batch_info['H'], batch_info['sigma_v'].squeeze(), 
                                        batch_info['sigma_y'].squeeze())
        m, A, C, H, sigma_v, sigma_y = (m.to(args.device), A.to(args.device), C.to(args.device), 
                                        H.to(args.device), sigma_v.to(args.device), sigma_y.to(args.device))
        
        current_actual_batch_size = m.shape[0]
        D, D_obs = args.ori_dim, args.obs_dim
        optimizer.zero_grad()

        # --- Initialize States and Noise Covariances ---
        ens_v_a = m.unsqueeze(1).repeat(1, N, 1) + torch.bmm(torch.randn_like(m.unsqueeze(1).repeat(1, N, 1)), C.transpose(-1,-2))
        gt_v_a = m.unsqueeze(1)
        Q = (sigma_v.view(current_actual_batch_size, 1, 1) ** 2) * torch.eye(D, device=args.device).unsqueeze(0)
        R = (sigma_y.view(current_actual_batch_size, 1, 1) ** 2) * torch.eye(D_obs, device=args.device).unsqueeze(0)

        # --- Initialize variables for loss calculation ---
        accumulated_loss_for_batch_load = 0.0
        num_valid_loss_contributions = 0
        collected_ens_v_a = [] if not args.running_loss else None
        collected_gt_v_a = [] if not args.running_loss else None

        end_ind_t = min(epoch + 1, args.train_steps - 1) if args.loss_warm_up else args.train_steps - 1
        
        if end_ind_t <= 0:
            if (batch_ind + 1) % args.print_batch == 0:
                print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
                      f'Skipped batch due to end_ind_t <=0 (Warm-up)')
            batch_time.update(time.time() - t_start)
            continue

        # ======================= [REFACTORED UNIFIED TIME-STEP LOOP] =======================
        for i in range(end_ind_t):
            # --- Common Forecast and Analysis Step ---
            gt_v_a = forward_fun(gt_v_a, A) + torch.bmm(torch.randn_like(gt_v_a), Q.sqrt())
            obs_y = H_fun(gt_v_a, H) + torch.bmm(torch.randn_like(H_fun(gt_v_a, H)), R.sqrt())
            ens_v_f = forward_fun(ens_v_a, A) + torch.bmm(torch.randn_like(ens_v_a), Q.sqrt())
            hv = H_fun(ens_v_f, H)
            r_noise = mean0(torch.bmm(torch.randn_like(hv), R.sqrt()))
            ens_i_innov = obs_y - hv - r_noise
            mean_hv = torch.mean(hv, dim=1, keepdim=True)
            mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True)

            current_analyzed_ens_v_a, _ = _process_analysis_step(
                args, model_list, ens_v_f, hv, obs_y,
                ens_i_innov, mean_ens_v_f, mean_hv, sigma_y=sigma_y,
            )
            
            # --- Strategy-Dependent Action inside the loop ---
            if args.running_loss:
                if (i + 1) > args.ignore_first:
                    ens_tensor_step = current_analyzed_ens_v_a.unsqueeze(0)
                    batch_v_step = gt_v_a.squeeze(1).unsqueeze(0)
                    valid_B_mask_this_step = ~torch.isnan(ens_tensor_step).any(dim=(0, 2, 3)).squeeze(0)

                    if valid_B_mask_this_step.any():
                        step_loss = sum(compute_loss(
                            ens_tensor=ens_tensor_step, batch_v=batch_v_step, loss_type=lt, ignore_first=0, end_ind=None,
                            valid_B_mask=valid_B_mask_this_step.unsqueeze(0), norm_p=args.es_p,
                            kes_sigma=args.kes_sigma, return_sum=True) for lt in args.loss_type)
                        
                        accumulated_loss_for_batch_load += step_loss
                        num_valid_in_step = torch.sum(valid_B_mask_this_step)
                        num_valid_loss_contributions += num_valid_in_step.item()
                        if num_valid_in_step > 0:
                            losses.update(step_loss.item() / num_valid_in_step.item(), num_valid_in_step.item())
            else: # Trajectory loss: collect tensors
                collected_ens_v_a.append(current_analyzed_ens_v_a)
                collected_gt_v_a.append(gt_v_a)
            
            # --- Common State Update and Detach Logic ---
            ens_v_a = current_analyzed_ens_v_a
            if epoch <= args.detach_training_epoch and args.detach_steps > 0 and (i + 1) % args.detach_steps == 0 and (i + 1) < end_ind_t:
                ens_v_a = ens_v_a.detach()
        
        # ======================= [POST-LOOP BACKPROPAGATION] =======================
        if args.running_loss:
            if num_valid_loss_contributions > 0:
                average_loss = accumulated_loss_for_batch_load / num_valid_loss_contributions
                average_loss.backward()
                if all_trainable_params:
                    nn.utils.clip_grad_norm_(all_trainable_params, max_norm=getattr(args, 'grad_clip_norm', 1.0))
                optimizer.step()
            else:
                num_batches_all_nan += 1
        else: # Trajectory loss
            if len(collected_ens_v_a) > args.ignore_first:
                ens_tensor = torch.stack(collected_ens_v_a, dim=0)[args.ignore_first:]
                batch_v = torch.stack(collected_gt_v_a, dim=0).squeeze(2)[args.ignore_first:]
                
                if ens_tensor.shape[0] > 0:
                    valid_B_mask = ~torch.isnan(ens_tensor).any(dim=(2, 3))
                    num_valid_loss_contributions = torch.sum(valid_B_mask).item()

                    if num_valid_loss_contributions > 0:
                        # normalize_val = torch.std(batch_v, dim=0)
                        normalize_val = None
                        total_loss = sum(compute_loss(
                            ens_tensor=ens_tensor, batch_v=batch_v, loss_type=lt, ignore_first=0, end_ind=None,
                            valid_B_mask=valid_B_mask, norm_p=args.es_p, kes_sigma=args.kes_sigma, return_sum=True,
                            normalize_val=normalize_val,
                        ) for lt in args.loss_type)
                        
                        average_loss = total_loss / num_valid_loss_contributions
                        average_loss.backward()
                        if all_trainable_params:
                            nn.utils.clip_grad_norm_(all_trainable_params, max_norm=getattr(args, 'grad_clip_norm', 1.0))
                        optimizer.step()
                        losses.update(average_loss.item(), num_valid_loss_contributions)
                    else:
                        num_batches_all_nan += 1
                else:
                    num_batches_all_nan += 1
            else:
                num_batches_all_nan += 1

        # --- Common Logging and Timing ---
        batch_time.update(time.time() - t_start)
        total_possible_contributions = current_actual_batch_size * max(0, end_ind_t - args.ignore_first)
        current_no_nan_percentage = (num_valid_loss_contributions / total_possible_contributions * 100) if total_possible_contributions > 0 else 100.0

        if (batch_ind + 1) % args.print_batch == 0:
            print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
                  f'Batch time {batch_time.val:.3f} (Avg: {batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} (Avg: {losses.avg:.3f})\t'
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}\t'
                  f'No NAN % (batch): {current_no_nan_percentage:.2f}%')

    if num_batches_all_nan == len(loader) and len(loader) > 0:
        print(f"Warning: All {len(loader)} batches in epoch {epoch} resulted in NaN or no valid loss.")
        if losses.count == 0:
            return float('nan')

    scheduler.step()
    return losses.avg

# Helper function to compute batch-wise covariance
def get_ens_cov(ens_v):
    """
    Computes the sample covariance matrix for an ensemble of states.
    Input:
        ens_v: Ensemble of states with shape (B, N, D)
               B: batch size, N: ensemble members, D: state dimension
    Output:
        Covariance matrix with shape (B, D, D)
    """
    mean = torch.mean(ens_v, dim=1, keepdim=True)
    ens_v_minus_mean = ens_v - mean
    cov = torch.bmm(ens_v_minus_mean.transpose(1, 2), ens_v_minus_mean) / (ens_v.shape[1] - 1)
    return cov

def test_model_v2(loader, model_list, args, plot_figures=True, fig_name='example_fig_v2', save_pdf=False, infl=1, loc_radius=None):
    """
    Tests the data assimilation models on a linear dataset.
    This version computes and compares against the ground truth error covariance
    updated via Kalman Filter equations.
    """
    model, infl_model, local_model, st_model1, st_model2 = model_list
    N = args.N

    # --- Forward and Observation Function Initialization for Linear Model ---
    forward_fun = lambda x, A: torch.bmm(x, A.transpose(-1, -2))
    H_fun = lambda x, H: torch.bmm(x, H.transpose(-1, -2))

    # --- Set Models to Evaluation Mode ---
    model.eval()
    if hasattr(infl_model, 'eval'): infl_model.eval()
    if hasattr(local_model, 'eval'): local_model.eval()
    if hasattr(st_model1, 'eval'): st_model1.eval()
    if hasattr(st_model2, 'eval'): st_model2.eval()

    # --- Initialize Result Tensors for All Batches---
    all_results = {
        'rmse': torch.empty(0, device=args.device),
        'rmv': torch.empty(0, device=args.device), # Using RMV for spread
        'rrmse': torch.empty(0, device=args.device),
        'crps': torch.empty(0, device=args.device),
        'rcrps': torch.empty(0, device=args.device),
        'cov_diff': torch.empty(0, device=args.device),
        'rcov_diff': torch.empty(0, device=args.device),
        'w2_diff': torch.empty(0, device=args.device),
    }
    loc_tensor_all_batches = None

    with torch.no_grad():
        min_m_norm, max_m_norm = float('inf'), float('-inf')
        for batch_ind, batch_info in enumerate(loader):
            # --- Unpack Batch Data ---
            m, A, C, H, sigma_v, sigma_y = (batch_info['m'], batch_info['A'], batch_info['C'], 
                                            batch_info['H'], batch_info['sigma_v'].squeeze(), 
                                            batch_info['sigma_y'].squeeze())
            m, A, C, H, sigma_v, sigma_y = (m.to(args.device), A.to(args.device), C.to(args.device), 
                                            H.to(args.device), sigma_v.to(args.device), sigma_y.to(args.device))
            
            batch_norms = torch.linalg.norm(m, dim=1)
            current_min_norm = batch_norms.min().item()
            current_max_norm = batch_norms.max().item()
            min_m_norm = min(min_m_norm, current_min_norm)
            max_m_norm = max(max_m_norm, current_max_norm)
            
            B, D, D_obs = m.shape[0], args.ori_dim, args.obs_dim
            
            # --- Initialize Ground Truth and Ensemble States ---
            gt_v_a = m.unsqueeze(1)
            # Initial analysis error covariance is the provided C
            P_a = C @ C.transpose(-1, -2) 
            
            # Initial ensemble
            ens_v_a = m.unsqueeze(1).repeat(1, N, 1)
            ens_v_a += torch.bmm(torch.randn_like(ens_v_a, device=args.device), C.transpose(-1,-2))

            # --- Lists to Store Trajectory Data for the Current Batch ---
            ens_list = [ens_v_a]
            gt_list = [gt_v_a.squeeze(1)]
            loc_records = []
            cov_diff_list = []
            rcov_diff_list = []
            w2_diff_list = []

            # --- Main Assimilation Loop ---
            for i in range(args.test_steps -1):
                # --- Ground Truth Evolution ---
                Q = (sigma_v.view(B, 1, 1) ** 2) * torch.eye(D, device=args.device).unsqueeze(0).repeat(B, 1, 1)
                R = (sigma_y.view(B, 1, 1) ** 2) * torch.eye(D_obs, device=args.device).unsqueeze(0).repeat(B, 1, 1)
                gt_v_f = forward_fun(gt_v_a, A)
                gt_v_a = gt_v_f + torch.bmm(torch.randn_like(gt_v_f), Q.sqrt())
                obs_y = H_fun(gt_v_a, H) + torch.bmm(torch.randn_like(H_fun(gt_v_a,H)), R.sqrt())

                # --- Kalman Filter Covariance Update (Ground Truth) ---
                P_f = A @ P_a @ A.transpose(-1, -2) + Q
                S = H @ P_f @ H.transpose(-1, -2) + R
                K = P_f @ H.transpose(-1, -2) @ torch.inverse(S)
                P_a = (torch.eye(D, device=args.device).unsqueeze(0).repeat(B,1,1) - K @ H) @ P_f

                # --- Ensemble Forecast ---
                ens_v_f = forward_fun(ens_v_a, A)
                # Add model error (process noise)
                ens_v_f += torch.bmm(torch.randn_like(ens_v_f), Q.sqrt())
                
                # --- Prepare for Analysis ---
                hv = H_fun(ens_v_f, H)
                r_noise = mean0(torch.bmm(torch.randn_like(hv), R.sqrt()))
                ens_i_innov = obs_y - hv - r_noise
                mean_hv = torch.mean(hv, dim=1, keepdim=True)
                mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True)
                
                # --- Analysis Step (Refactored) ---
                ens_v_a, loc_nn_output = _process_analysis_step(
                    args, model_list, ens_v_f, hv, obs_y, ens_i_innov, mean_ens_v_f, mean_hv, sigma_y=sigma_y,
                    infl=infl, loc_radius=loc_radius, 
                )

                # --- Calculate and Store Covariance Difference ---
                P_ens_a = get_ens_cov(ens_v_a)
                cov_diff = torch.norm(P_ens_a - P_a, p='fro', dim=(-2, -1))
                rcov_diff =  cov_diff / torch.norm(P_a, p='fro', dim=(-2, -1))
                w2_diff = wasserstein2_multivariate_gaussian(mean_true=gt_v_a, cov_true=P_a, mean_sample=ens_v_a.mean(dim=1), cov_sample=P_ens_a)
                cov_diff_list.append(cov_diff)
                rcov_diff_list.append(rcov_diff)
                w2_diff_list.append(w2_diff)
                
                if loc_nn_output is not None:
                    loc_records.append(loc_nn_output)
                
                # --- Store results for this step ---
                ens_list.append(ens_v_a)
                gt_list.append(gt_v_a.squeeze(1))
            
            # --- Process and Store Batch Results ---
            ens_tensor = torch.stack(ens_list, dim=0)
            gt_tensor = torch.stack(gt_list, dim=0)
            
            # --- Metric Calculation for the Batch ---
            # Note: We compute mean over time steps (dim=0) first, then over batch items
            rmse_val = torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - gt_tensor) ** 2, dim=-1)).mean(dim=0)
            rms_val = torch.sqrt(torch.mean(gt_tensor ** 2, dim=-1)).mean(dim=0)
            rrmse_val = rmse_val / rms_val
            rmv_val = torch.sqrt(torch.mean((ens_tensor - gt_tensor.unsqueeze(2))**2, dim=(2,-1))).mean(dim=0)
            crps_val = torch.mean(compute_es(ens_states=ens_tensor, true_states=gt_tensor, norm_p=1), dim=0) 
            rcrps_val = crps_val / torch.mean(torch.norm(gt_tensor, p=2, dim=-1), dim=0)
            
            cov_diff_tensor = torch.stack(cov_diff_list, dim=0)
            rcov_diff_tensor = torch.stack(rcov_diff_list, dim=0)
            w2_diff_tensor = torch.stack(w2_diff_list, dim=0)
            cov_diff_val = cov_diff_tensor.mean(dim=0)
            rcov_diff_val = rcov_diff_tensor.mean(dim=0)
            w2_diff_val = w2_diff_tensor.mean(dim=0)

            # --- Aggregate Results ---
            all_results['rmse'] = torch.cat((all_results['rmse'], rmse_val))
            all_results['rmv'] = torch.cat((all_results['rmv'], rmv_val))
            all_results['rrmse'] = torch.cat((all_results['rrmse'], rrmse_val))
            all_results['crps'] = torch.cat((all_results['crps'], crps_val))
            all_results['rcrps'] = torch.cat((all_results['rcrps'], rcrps_val))
            all_results['cov_diff'] = torch.cat((all_results['cov_diff'], cov_diff_val))
            all_results['rcov_diff'] = torch.cat((all_results['rcov_diff'], rcov_diff_val))
            all_results['w2_diff'] = torch.cat((all_results['w2_diff'], w2_diff_val))
            
            # Handle localization tensor if it exists
            if args.v != "EtE" and not args.no_localization and loc_records:
                current_loc_tensor = torch.stack(loc_records).unsqueeze(0) # Add batch dim
                if loc_tensor_all_batches is None:
                    loc_tensor_all_batches = current_loc_tensor
                else:
                    loc_tensor_all_batches = torch.cat((loc_tensor_all_batches, current_loc_tensor), dim=0)
                    
        if plot_figures: 
            time_idx_plot = -2 # Example: Plot second to last time state
            num_dims_plot = 4 # Example
            dim_indices_plot = list(range(min(args.ori_dim, num_dims_plot)))

            plot_particle_trajectories_with_histograms(particles=ens_tensor[:,time_idx_plot,:,:], 
                                                    true_traj=gt_tensor[:,time_idx_plot,:], 
                                                    observation=None, #observations[:,time_idx_plot,:],
                                                    dim_indices=dim_indices_plot,
                                                    start_time=0, # Adjust as needed
                                                    end_time=ens_tensor.shape[0], #Trajectory length
                                                    mode='quantile',
                                                    save_fig=True,
                                                    save_pdf=save_pdf,
                                                    save_name=fig_name + "_hist",
                                                    hist_step=1,
                                                    fontsize=None)

    # --- Final Metrics Calculation ---
    final_metrics = {}
    
    # Check for NaN values in a reference metric (e.g., rrmse)
    nan_mask = torch.isnan(all_results['rrmse'])
    valid_B_mask = ~nan_mask
    
    if not valid_B_mask.any():
         # Return NaNs if all results are invalid
        metrics_keys = ['mean_rmse', 'std_rmse', 'mean_rmv', 'std_rmv', 
                        'mean_rrmse', 'std_rrmse', 'mean_crps', 'std_crps',
                        'mean_rcrps', 'std_rcrps', 'mean_cov_diff', 'std_cov_diff',
                        'mean_rcov_diff', 'std_rcov_diff', 'mean_w2_diff', 'std_w2_diff',
                        'min_m_norm', 'max_m_norm']
        final_metrics = {key: float('nan') for key in metrics_keys}
        final_metrics['no_nan_percent'] = 0.0
    else:
        # Calculate mean and std for valid results
        final_metrics['mean_rrmse'], final_metrics['std_rrmse'] = get_mean_std(all_results['rrmse'][valid_B_mask])
        final_metrics['mean_rmse'], final_metrics['std_rmse'] = get_mean_std(all_results['rmse'][valid_B_mask])
        final_metrics['mean_rmv'], final_metrics['std_rmv'] = get_mean_std(all_results['rmv'][valid_B_mask])
        final_metrics['mean_crps'], final_metrics['std_crps'] = get_mean_std(all_results['crps'][valid_B_mask])
        final_metrics['mean_rcrps'], final_metrics['std_rcrps'] = get_mean_std(all_results['rcrps'][valid_B_mask])
        final_metrics['mean_cov_diff'], final_metrics['std_cov_diff'] = get_mean_std(all_results['cov_diff'][valid_B_mask])
        final_metrics['mean_rcov_diff'], final_metrics['std_rcov_diff'] = get_mean_std(all_results['rcov_diff'][valid_B_mask])
        final_metrics['mean_w2_diff'], final_metrics['std_w2_diff'] = get_mean_std(all_results['w2_diff'][valid_B_mask])
        final_metrics['min_m_norm'], final_metrics['max_m_norm'] = min_m_norm, max_m_norm
        final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_results['rrmse'].numel() * 100.0

    final_loc_tensor = loc_tensor_all_batches[0] if loc_tensor_all_batches is not None else torch.empty(1, device=args.device)

    final_metrics['loc_tensor'] = final_loc_tensor
    return final_metrics

def print_test_results_v2(results):
    """
    Print formatted test results from a dictionary.

    Args:
        results (dict): A dictionary containing metric names as keys
                        and their calculated values. It checks for the
                        existence of keys before printing.
    """
    if 'mean_rmse' in results and 'std_rmse' in results:
        print(f"RMSE: {results['mean_rmse']:.3f} ± {results['std_rmse']:.3f}")
        
    if 'mean_rrmse' in results and 'std_rrmse' in results:
        print(f"RRMSE: {results['mean_rrmse']:.3f} ± {results['std_rrmse']:.3f}")
        
    if 'mean_rmv' in results and 'std_rmv' in results:
        print(f"RMV: {results['mean_rmv']:.3f} ± {results['std_rmv']:.3f}")
        
    if 'mean_crps' in results and 'std_crps' in results:
        print(f"CRPS: {results['mean_crps']:.3f} ± {results['std_crps']:.3f}")
        
    if 'mean_rcrps' in results and 'std_rcrps' in results:
        print(f"RCRPS: {results['mean_rcrps']:.3f} ± {results['std_rcrps']:.3f}")
        
    if 'mean_w2_diff' in results and 'std_w2_diff' in results:
        print(f"W2: {results['mean_w2_diff']:.3f} ± {results['std_w2_diff']:.3f}")
        
    if 'no_nan_percent' in results:
        print(f"No NAN Percentage: {results['no_nan_percent']:.2f}%")
        
    if 'min_m_norm' in results and 'max_m_norm' in results:
        print(f"Min initial norm: {results['min_m_norm']:.2f}, Max initial norm: {results['max_m_norm']:.2f}")
    

### test gt uncertainty 
def test_linear_sampling_error(loader, args, num_resamples):
    """
    Computes the sampling error of an ensemble in a linear-Gaussian system.

    This function runs a Kalman Filter to get the true posterior mean and
    covariance. At each step, it repeatedly samples from this true posterior
    to measure the statistical error (RMSE, CRPS, W2) introduced by a
    finite ensemble size N.

    Args:
        loader (DataLoader): DataLoader for initial conditions.
        args (Namespace): Experiment parameters (N, device, test_steps, etc.).
        num_resamples (int): Number of times to resample at each step for averaging.

    Returns:
        dict: A dictionary of final, averaged metrics and their standard deviation.
    """
    N = args.N
    D = args.ori_dim
    D_obs = args.obs_dim

    # --- Forward and Observation Function Initialization ---
    forward_fun = lambda x, A: torch.bmm(x, A.transpose(-1, -2))
    H_fun = lambda x, H: torch.bmm(x, H.transpose(-1, -2))

    # --- Initialize Result Tensors for All Batches---
    all_results = {
        'rmse': torch.empty(0, device=args.device),
        'rmv': torch.empty(0, device=args.device),
        'rrmse': torch.empty(0, device=args.device),
        'crps': torch.empty(0, device=args.device),
        'rcrps': torch.empty(0, device=args.device),
        'w2_diff': torch.empty(0, device=args.device),
    }

    with torch.no_grad():
        for batch_ind, batch_info in enumerate(loader):
            # --- Unpack Batch Data ---
            m, A, C, H, sigma_v, sigma_y = (batch_info['m'], batch_info['A'], batch_info['C'], 
                                             batch_info['H'], batch_info['sigma_v'].squeeze(), 
                                             batch_info['sigma_y'].squeeze())
            m, A, C, H, sigma_v, sigma_y = (m.to(args.device), A.to(args.device), C.to(args.device), 
                                             H.to(args.device), sigma_v.to(args.device), sigma_y.to(args.device))
            
            B = m.shape[0]
            
            # --- Initialize Ground Truth State and Covariance ---
            gt_v_a = m.unsqueeze(1) # Shape: (B, 1, D)
            P_a = C @ C.transpose(-1, -2) 
            
            # --- Lists to Store Trajectory-Averaged Data ---
            gt_list = [gt_v_a.squeeze(1)]
            rmse_list, rmv_list, crps_list, w2_diff_list = [], [], [], []

            # --- Main Assimilation Loop ---
            for i in range(args.test_steps - 1):
                print(i, args.test_steps-1)
                # --- Ground Truth Evolution (Kalman Filter) ---
                Q = (sigma_v.view(B, 1, 1) ** 2) * torch.eye(D, device=args.device)
                R = (sigma_y.view(B, 1, 1) ** 2) * torch.eye(D_obs, device=args.device)
                
                gt_v_f = forward_fun(gt_v_a, A)
                gt_v_a = gt_v_f + torch.bmm(torch.randn_like(gt_v_f), Q.sqrt())
                obs_y = H_fun(gt_v_a, H) + torch.bmm(torch.randn_like(H_fun(gt_v_a, H)), R.sqrt())

                # --- Kalman Filter Update (Ground Truth) ---
                P_f = A @ P_a @ A.transpose(-1, -2) + Q
                S = H @ P_f @ H.transpose(-1, -2) + R
                K = P_f @ H.transpose(-1, -2) @ torch.inverse(S)
                P_a = (torch.eye(D, device=args.device) - K @ H) @ P_f
                gt_v_a = (gt_v_f.transpose(-1,-2) + K @ (obs_y - H_fun(gt_v_f, H)).transpose(-1,-2)).transpose(-1,-2)

                # --- Resampling and Metric Calculation Loop ---
                step_rmses, step_rmvs, step_crpses, step_w2s = [], [], [], []
                L_a = torch.linalg.cholesky(P_a)
                
                for _ in range(num_resamples):
                    noise = torch.randn(B, N, D, device=args.device)
                    ens_v_a = gt_v_a + torch.bmm(noise, L_a.transpose(-1,-2))
                    
                    ens_mean = ens_v_a.mean(dim=1)
                    P_ens_a = get_ens_cov(ens_v_a) 

                    # --- Calculate Metrics for this Sample Set ---
                    rmse = torch.sqrt(torch.mean((ens_mean - gt_v_a.squeeze(1)) ** 2, dim=-1))
                    step_rmses.append(rmse)

                    rmv = torch.sqrt(torch.mean((ens_v_a - gt_v_a)**2, dim=(1,-1)))
                    step_rmvs.append(rmv)
                    
                    crps = compute_es(ens_v_a.unsqueeze(0), gt_v_a.squeeze(1).unsqueeze(0)).squeeze(0)
                    step_crpses.append(crps)

                    w2 = wasserstein2_multivariate_gaussian(
                        mean_true=gt_v_a.squeeze(1), cov_true=P_a, 
                        mean_sample=ens_mean, cov_sample=P_ens_a
                    )
                    step_w2s.append(w2)

                # --- Average metrics over the resamples for this time step ---
                rmse_list.append(torch.stack(step_rmses).mean(dim=0))
                rmv_list.append(torch.stack(step_rmvs).mean(dim=0))
                crps_list.append(torch.stack(step_crpses).mean(dim=0))
                w2_diff_list.append(torch.stack(step_w2s).mean(dim=0))
                
                gt_list.append(gt_v_a.squeeze(1))

            # --- Process and Store Batch Results (Averaged over trajectory) ---
            gt_tensor = torch.stack(gt_list, dim=0)
            
            rmse_val = torch.stack(rmse_list).mean(dim=0)
            rmv_val = torch.stack(rmv_list).mean(dim=0)
            crps_val = torch.stack(crps_list).mean(dim=0)
            w2_diff_val = torch.stack(w2_diff_list).mean(dim=0)

            rms_val = torch.sqrt(torch.mean(gt_tensor ** 2, dim=-1)).mean(dim=0)
            rrmse_val = rmse_val / rms_val
            rcrps_val = crps_val / torch.mean(torch.norm(gt_tensor, p=2, dim=-1), dim=0)

            # --- Aggregate Results ---
            all_results['rmse'] = torch.cat((all_results['rmse'], rmse_val))
            all_results['rmv'] = torch.cat((all_results['rmv'], rmv_val))
            all_results['rrmse'] = torch.cat((all_results['rrmse'], rrmse_val))
            all_results['crps'] = torch.cat((all_results['crps'], crps_val))
            all_results['rcrps'] = torch.cat((all_results['rcrps'], rcrps_val))
            all_results['w2_diff'] = torch.cat((all_results['w2_diff'], w2_diff_val))

    # --- Final Metrics Calculation ---
    final_metrics = {}
    nan_mask = torch.isnan(all_results['rrmse'])
    valid_B_mask = ~nan_mask
    
    if not valid_B_mask.any():
        metrics_keys = ['mean_rrmse', 'std_rrmse', 'mean_rmse', 'std_rmse', 
                        'mean_rmv', 'std_rmv', 'mean_crps', 'std_crps', 
                        'mean_rcrps', 'std_rcrps', 'mean_w2_diff', 'std_w2_diff']
        final_metrics = {key: float('nan') for key in metrics_keys}
        final_metrics['no_nan_percent'] = 0.0
    else:
        final_metrics['mean_rrmse'], final_metrics['std_rrmse'] = get_mean_std(all_results['rrmse'][valid_B_mask])
        final_metrics['mean_rmse'], final_metrics['std_rmse'] = get_mean_std(all_results['rmse'][valid_B_mask])
        final_metrics['mean_rmv'], final_metrics['std_rmv'] = get_mean_std(all_results['rmv'][valid_B_mask])
        final_metrics['mean_crps'], final_metrics['std_crps'] = get_mean_std(all_results['crps'][valid_B_mask])
        final_metrics['mean_rcrps'], final_metrics['std_rcrps'] = get_mean_std(all_results['rcrps'][valid_B_mask])
        final_metrics['mean_w2_diff'], final_metrics['std_w2_diff'] = get_mean_std(all_results['w2_diff'][valid_B_mask])
        final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_results['rrmse'].numel() * 100.0

    return final_metrics