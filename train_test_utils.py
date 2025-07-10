import numpy as np
import time
import os

import torch
import torch.nn as nn

from utils import L63, L96, rk4, etd_rk4_wrapper, CircleODE, DoubleWellODE
from utils import AverageMeter, mystery_operator, get_mean_std
from utils import post_process, mean0
from visualization import plot_particle_trajectories_with_histograms, plot_particle_trajectories
from localization import dist2coeff, create_loc_mat
from loss import compute_loss, compute_es, wasserstein2_multivariate_gaussian
from networks import NaiveNetwork, SetTransformer, Simple_MLP, ConditionTransformerNetwork
from benchmark_analysis import ensemble_kalman_filter_analysis, bootstrap_particle_filter_analysis


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
                                        hidden_dim=args.hidden_dim, num_layers=3, freeze_WQ=not args.unfreeze_WQ).to(args.device)
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

def _process_analysis_step(args, model_list, ens_v_f, hv, obs_y, ens_i_innov, mean_ens_v_f, mean_hv, sigma_y=None):
    """
    Process the analysis ensemble step using neural networks.
    Returns the analyzed ensemble and the localization neural network output.
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
    
    # Check if sigma_y is a scalar (0-dimensional tensor)
    if sigma_y.ndim == 0:
        # If it's a scalar, expand it to match the batch size before reshaping
        sigma_y = sigma_y.expand(B).view(B, 1, 1)
    else:
        # If it's already a 1D tensor of size B, just reshape it
        sigma_y = sigma_y.view(B, 1, 1)
    
    # Initialize loc_nn_output to None to prevent UnboundLocalError
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
        # modified to learning the residual 
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
        K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) + R_cov * (N_ens - 1)
        K = torch.bmm(K1, torch.inverse(K2))
        current_analyzed_ens_v_a = Vnn1 + torch.bmm(ens_i_innov, K.transpose(1, 2))
    else:
        raise NotImplementedError(f"args.v = {args.v} is not implemented.")

    if args.clamp is not None:
        ens_v_a = torch.clamp(current_analyzed_ens_v_a, min=-args.clamp, max=args.clamp)
    else:
        ens_v_a = current_analyzed_ens_v_a
    
    # Return both the analyzed ensemble and the localization output
    return ens_v_a, loc_nn_output

# def train_model(epoch, loader, model_list, optimizer, scheduler, args, H_info=None):
#     model, infl_model, local_model, st_model1, st_model2 = model_list
#     m = args.N
#     losses = AverageMeter()
#     batch_time = AverageMeter()
    
#     if args.dataset == "lorenz63":
#         forward_fun = L63.forward
#     elif args.dataset == "lorenz96":
#         forward_fun = L96.forward
#     elif args.dataset == "ks":
#         if args.dt_iter <= 0:
#             raise ValueError("args.dt_iter must be positive for KS model.")
#         forward_fun = etd_rk4_wrapper(device=args.device, dt=args.dt / args.dt_iter)
#     else:
#         raise NotImplementedError(f"Dataset {args.dataset} not implemented.")
    
#     if H_info is None:
#         H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
#     else:
#         H_fun, H = H_info

#     model.train()
#     if hasattr(infl_model, 'train'): infl_model.train()
#     if hasattr(local_model, 'train'): local_model.train()
#     if hasattr(st_model1, 'train'): st_model1.train()
#     if hasattr(st_model2, 'train'): st_model2.train()

#     all_trainable_params = []
#     for m_ in [model, infl_model, local_model, st_model1, st_model2]:
#         if hasattr(m_, 'parameters') and not isinstance(m_, NaiveNetwork):
#             all_trainable_params.extend(list(filter(lambda p: p.requires_grad, m_.parameters())))

#     num_batches_all_nan = 0

#     for batch_ind, batch_v_trajectory in enumerate(loader):
#         t_start = time.time()
#         batch_v_trajectory = batch_v_trajectory.to(device=args.device)
#         current_actual_batch_size = batch_v_trajectory.shape[1]
#         optimizer.zero_grad()
#         ens_v_a = batch_v_trajectory[0].unsqueeze(1).repeat(1, m, 1)
#         ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens
#         accumulated_loss_for_batch_load = 0.0
#         num_valid_loss_contributions = 0

#         end_ind_t = min(epoch + 1, len(batch_v_trajectory) - 1) if args.loss_warm_up else len(batch_v_trajectory) - 1
#         if end_ind_t <= 0:
#             if (batch_ind + 1) % args.print_batch == 0:
#                 print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
#                       f'Skipped batch due to end_ind_t <=0 (Warm-up or short trajectory)')
#             batch_time.update(time.time() - t_start)
#             continue

#         for i in range(end_ind_t):
#             obs_y = H_fun(batch_v_trajectory[i + 1].unsqueeze(1))
#             obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)
#             ens_v_a_forecast_input = ens_v_a.reshape(-1, args.ori_dim)
            
#             for j_iter in range(args.dt_iter):
#                 if args.dataset == 'ks':
#                     ens_v_a_forecast_input = forward_fun(ens_v_a_forecast_input, None, args.dt / args.dt_iter)
#                 else:
#                     current_time_for_rk4 = i * args.dt + j_iter * (args.dt / args.dt_iter)
#                     ens_v_a_forecast_input = rk4(forward_fun, ens_v_a_forecast_input, current_time_for_rk4, args.dt / args.dt_iter)
            
#             ens_v_f = ens_v_a_forecast_input.view(-1, m, args.ori_dim)
#             ens_v_f = ens_v_f + torch.randn_like(ens_v_f, device=args.device) * args.sigma_v
#             hv = H_fun(ens_v_f)
#             r_noise = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
#             ens_i_innov = obs_y - hv - r_noise
#             mean_hv = torch.mean(hv, dim=1, keepdim=True)
#             mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True)

#             # ======================= [Refactored Core Call] =======================
#             # Replace the complex analysis logic with a single call to the new function.
#             current_analyzed_ens_v_a, _ = _process_analysis_step(
#                 args, model_list, ens_v_f, hv, obs_y,
#                 ens_i_innov, mean_ens_v_f, mean_hv,
#             )
#             # ======================================================================

#             if (i + 1) > args.ignore_first:
#                 ens_tensor_step = current_analyzed_ens_v_a.unsqueeze(0)
#                 batch_v_step = batch_v_trajectory[i + 1].unsqueeze(0)
#                 nan_mask_this_step = torch.isnan(ens_tensor_step).any(dim=(0, 2, 3)).squeeze(0) 
#                 valid_B_mask_this_step = ~nan_mask_this_step

#                 if valid_B_mask_this_step.any():
#                     step_loss_sum_over_valid_batch_items = 0
#                     for loss_type_val in args.loss_type:
#                         step_loss_sum_over_valid_batch_items += compute_loss(
#                             ens_tensor=ens_tensor_step, batch_v=batch_v_step,
#                             loss_type=loss_type_val, ignore_first=0, end_ind=None,
#                             valid_B_mask=valid_B_mask_this_step.unsqueeze(0),
#                             norm_p=args.es_p, kes_sigma=args.kes_sigma, return_sum=True 
#                         )
                    
#                     accumulated_loss_for_batch_load += step_loss_sum_over_valid_batch_items
#                     num_valid_in_step = torch.sum(valid_B_mask_this_step)
#                     num_valid_loss_contributions += num_valid_in_step.item()
                    
#                     if num_valid_in_step > 0:
#                         losses.update(step_loss_sum_over_valid_batch_items.item() / num_valid_in_step.item(), 
#                                       num_valid_in_step.item())
            
#             ens_v_a = current_analyzed_ens_v_a
#             if epoch <= args.detach_training_epoch and args.detach_steps > 0 and (i + 1) % args.detach_steps == 0 and (i + 1) < end_ind_t:
#                 ens_v_a = ens_v_a.detach()
        
#         if num_valid_loss_contributions > 0:
#             average_loss_for_loaded_batch = accumulated_loss_for_batch_load / num_valid_loss_contributions
#             average_loss_for_loaded_batch.backward()
            
#             if all_trainable_params:
#                 nn.utils.clip_grad_norm_(all_trainable_params, max_norm=getattr(args, 'grad_clip_norm', 1.0))
#             optimizer.step()
#         else:
#             num_batches_all_nan += 1

#         batch_time.update(time.time() - t_start)
#         total_possible_contributions = current_actual_batch_size * max(0, end_ind_t - args.ignore_first)
#         current_no_nan_percentage = (num_valid_loss_contributions / total_possible_contributions * 100) if total_possible_contributions > 0 else 100.0

#         if (batch_ind + 1) % args.print_batch == 0:
#             print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
#                   f'Batch time {batch_time.val:.3f} (Avg: {batch_time.avg:.3f})\t'
#                   f'Loss {losses.val:.3f} (Avg: {losses.avg:.3f})\t'
#                   f'LR: {optimizer.param_groups[0]["lr"]:.2e}\t'
#                   f'No NAN % (batch): {current_no_nan_percentage:.2f}%')

#     if num_batches_all_nan == len(loader) and len(loader) > 0:
#         print(f"Warning: All {len(loader)} batches in epoch {epoch} resulted in NaN or no valid loss.")
#         if losses.count == 0:
#             return float('nan')

#     scheduler.step()
#     return losses.avg

def train_model(epoch, loader, model_list, optimizer, scheduler, args, H_info=None):
    """
    Function to train the model for one epoch.

    Args:
        epoch (int): The current epoch number.
        loader (DataLoader): The data loader for training data.
        model_list (list): A list of models to be trained.
        optimizer (Optimizer): The optimizer for updating model weights.
        scheduler (Scheduler): The learning rate scheduler.
        args (Namespace): A namespace containing all arguments/hyperparameters.
        H_info (tuple, optional): A tuple containing the observation operator and its matrix form. Defaults to None.

    Returns:
        float: The average loss for the epoch.
    """
    model, infl_model, local_model, st_model1, st_model2 = model_list
    m = args.N
    losses = AverageMeter()
    batch_time = AverageMeter()

    # --- Forward Function Selection ---
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
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
                      f'Skipped batch due to end_ind_t <=0 (Warm-up or short trajectory)')
            batch_time.update(time.time() - t_start)
            continue
        
        # --- Initialize variables for loss calculation ---
        accumulated_loss_for_batch_load = 0.0
        num_valid_loss_contributions = 0
        collected_ens_v_a = [] if not args.running_loss else None

        # ======================= [REFACTORED UNIFIED TIME-STEP LOOP] =======================
        for i in range(end_ind_t):
            # --- Common Forecast and Analysis Step ---
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
            hv = H_fun(ens_v_f)
            r_noise = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
            ens_i_innov = obs_y - hv - r_noise
            mean_hv = torch.mean(hv, dim=1, keepdim=True)
            mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True)

            current_analyzed_ens_v_a, _ = _process_analysis_step(
                args, model_list, ens_v_f, hv, obs_y,
                ens_i_innov, mean_ens_v_f, mean_hv,
            )

            # --- Strategy-Dependent Action inside the loop ---
            if args.running_loss:
                if (i + 1) > args.ignore_first:
                    ens_tensor_step = current_analyzed_ens_v_a.unsqueeze(0)
                    batch_v_step = batch_v_trajectory[i + 1].unsqueeze(0)
                    nan_mask_this_step = torch.isnan(ens_tensor_step).any(dim=(0, 2, 3)).squeeze(0) 
                    valid_B_mask_this_step = ~nan_mask_this_step

                    if valid_B_mask_this_step.any():
                        step_loss_sum_over_valid_batch_items = 0
                        for loss_type_val in args.loss_type:
                            step_loss_sum_over_valid_batch_items += compute_loss(
                                ens_tensor=ens_tensor_step, batch_v=batch_v_step,
                                loss_type=loss_type_val, ignore_first=0, end_ind=None,
                                valid_B_mask=valid_B_mask_this_step.unsqueeze(0),
                                norm_p=args.es_p, kes_sigma=args.kes_sigma, return_sum=True 
                            )
                        
                        accumulated_loss_for_batch_load += step_loss_sum_over_valid_batch_items
                        num_valid_in_step = torch.sum(valid_B_mask_this_step)
                        num_valid_loss_contributions += num_valid_in_step.item()
                        
                        if num_valid_in_step > 0:
                            losses.update(step_loss_sum_over_valid_batch_items.item() / num_valid_in_step.item(), 
                                          num_valid_in_step.item())
            else: # Trajectory loss: collect tensors
                collected_ens_v_a.append(current_analyzed_ens_v_a)
            
            # --- Common State Update and Detach Logic ---
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
        else: # Trajectory loss
            if len(collected_ens_v_a) > args.ignore_first:
                ens_tensor = torch.stack(collected_ens_v_a, dim=0)[args.ignore_first:]
                batch_v = batch_v_trajectory[1:end_ind_t + 1][args.ignore_first:]
                
                if ens_tensor.shape[0] > 0:
                    valid_B_mask = ~torch.isnan(ens_tensor).any(dim=(2, 3))
                    num_valid_loss_contributions = torch.sum(valid_B_mask).item()

                    if num_valid_loss_contributions > 0:
                        total_loss = sum(compute_loss(
                                ens_tensor=ens_tensor, batch_v=batch_v, loss_type=loss_type_val,
                                ignore_first=0, end_ind=None, valid_B_mask=valid_B_mask,
                                norm_p=args.es_p, kes_sigma=args.kes_sigma, return_sum=True
                            ) for loss_type_val in args.loss_type)
                        
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

def generate_and_cache_pf_results(loader, args, H_info, check_disk=True, calculate_crps=True):
    """
    Runs a particle filter, saves results (means, covariances) to a cache file,
    and computes performance metrics (RMSE, optional CRPS).

    This version is memory-efficient and conditionally calculates metrics.
    If calculate_crps is False, CRPS keys are omitted from the output.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader providing the dataset batches.
        args (argparse.Namespace): A namespace object containing script arguments.
        H_info (tuple): A tuple containing the observation operator function and matrix (H_fun, H).
        check_disk (bool): If True, checks if the cache file already exists and skips computation.
        calculate_crps (bool): If True, calculates CRPS and RCRPS metrics.

    Returns:
        dict: A dictionary containing performance metrics. CRPS-related keys are
              only present if calculate_crps is True.
    """

    # --- Model and Observation Initialization ---
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
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
        # Return empty metrics if cache exists
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

    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)

            # --- Particle Filter Initialization ---
            pf_ens_v_a = batch_v[0].unsqueeze(1).repeat(1, args.pf_N, 1)
            pf_ens_v_a += torch.randn_like(pf_ens_v_a, device=args.device) * args.sigma_ens

            batch_pf_means_to_cache, batch_pf_covs_to_cache = [], []
            batch_rmse_steps = []
            if calculate_crps:
                batch_crps_steps = []

            # --- Calculate metrics for the initial state (t=0) ---
            true_state_t0 = batch_v[0]
            rmse_t0 = torch.sqrt(torch.mean((pf_ens_v_a.mean(dim=1) - true_state_t0) ** 2, dim=1))
            batch_rmse_steps.append(rmse_t0)
            if calculate_crps:
                crps_t0 = compute_es(pf_ens_v_a.unsqueeze(0), true_state_t0.unsqueeze(0), norm_p=1)
                batch_crps_steps.append(crps_t0)

            # --- Generate Observations ---
            obs_y_list = [H_fun(batch_v[i].unsqueeze(1)) + args.sigma_y * torch.randn_like(H_fun(batch_v[i].unsqueeze(1))) for i in range(len(batch_v))]

            # --- Main Particle Filter Assimilation Loop ---
            for i in range(len(batch_v) - 1):
                # Forecast Step
                pf_ens_v_a_forecast_input = pf_ens_v_a.view(-1, args.ori_dim)
                for j_iter in range(args.dt_iter):
                    if args.dataset == 'ks':
                        pf_ens_v_a_forecast_input = forward_fun(pf_ens_v_a_forecast_input, None, args.dt / args.dt_iter)
                    else:
                        current_time_for_rk4 = i * args.dt + j_iter * (args.dt / args.dt_iter)
                        pf_ens_v_a_forecast_input = rk4(forward_fun, pf_ens_v_a_forecast_input, current_time_for_rk4, args.dt / args.dt_iter)
                pf_ens_v_f = pf_ens_v_a_forecast_input.view(-1, args.pf_N, args.ori_dim)
                pf_ens_v_f += torch.randn_like(pf_ens_v_f) * args.sigma_v

                # Analysis Step
                pf_ens_v_a = bootstrap_particle_filter_analysis(
                    pf_ens_v_f, obs_y_list[i + 1].squeeze(1), H_fun, args.sigma_y,
                    resampling_method="multinomial", sigma_reg=args.sigma_reg,
                    max_chunk_size=500000,
                )

                # Store results for caching and metrics
                pf_mean_a = torch.mean(pf_ens_v_a, dim=1)
                batch_pf_means_to_cache.append(pf_mean_a)
                batch_pf_covs_to_cache.append(get_ens_cov(pf_ens_v_a))
                
                true_state_ti = batch_v[i + 1]
                rmse_ti = torch.sqrt(torch.mean((pf_mean_a - true_state_ti) ** 2, dim=1))
                batch_rmse_steps.append(rmse_ti)
                if calculate_crps:
                    crps_ti = compute_es(pf_ens_v_a.unsqueeze(0), true_state_ti.unsqueeze(0), norm_p=1)
                    batch_crps_steps.append(crps_ti)

            # --- Aggregate and Cache Batch Results ---
            all_pf_results_to_cache.append({'means': torch.stack(batch_pf_means_to_cache), 'covs': torch.stack(batch_pf_covs_to_cache)})
            
            # --- Calculate and Aggregate Average Metrics for the Batch ---
            rmse_val = torch.mean(torch.stack(batch_rmse_steps), dim=0)
            rms_val = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0)
            all_pf_metrics['rmse'] = torch.cat((all_pf_metrics['rmse'], rmse_val))
            all_pf_metrics['rrmse'] = torch.cat((all_pf_metrics['rrmse'], rmse_val / rms_val))
            if calculate_crps:
                crps_val = torch.mean(torch.stack(batch_crps_steps), dim=0)
                rcrps_val = crps_val / torch.mean(torch.norm(batch_v, p=2, dim=2), dim=0)
                all_pf_metrics['crps'] = torch.cat((all_pf_metrics['crps'], crps_val))
                all_pf_metrics['rcrps'] = torch.cat((all_pf_metrics['rcrps'], rcrps_val))
            
            print("update results")

    # --- Save All Results to Cache File ---
    print(f"Saving PF results to: {cache_filepath}")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(all_pf_results_to_cache, cache_filepath)

    # --- Final Metrics Calculation ---
    if all_pf_metrics['rrmse'].numel() == 0:
        return {} # Return empty dict if no results were produced

    valid_B_mask = ~torch.isnan(all_pf_metrics['rrmse'])
    if not valid_B_mask.any():
        return {} # Return empty dict if all results are invalid

    # Calculate final metrics only for valid results
    mean_rrmse, std_rrmse = get_mean_std(all_pf_metrics['rrmse'][valid_B_mask])
    mean_rmse, std_rmse = get_mean_std(all_pf_metrics['rmse'][valid_B_mask])
    
    final_metrics = {
        'mean_rrmse': mean_rrmse,
        'std_rrmse': std_rrmse,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
    }

    if calculate_crps:
        mean_crps, std_crps = get_mean_std(all_pf_metrics['crps'][valid_B_mask])
        mean_rcrps, std_rcrps = get_mean_std(all_pf_metrics['rcrps'][valid_B_mask])
        final_metrics.update({
            'mean_crps': mean_crps,
            'std_crps': std_crps,
            'mean_rcrps': mean_rcrps,
            'std_rcrps': std_rcrps,
        })
        
    final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_pf_metrics['rrmse'].numel() * 100.0

    return final_metrics


def test_model(loader, model_list, args, infl=1, H_info=None, plot_figures=True, fig_name='example_fig', save_pdf=False):
    model, infl_model, local_model, st_model1, st_model2 = model_list
    m = args.N
    
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
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
    
    model.eval()
    if hasattr(infl_model, 'eval'): infl_model.eval()
    if hasattr(local_model, 'eval'): local_model.eval()
    if hasattr(st_model1, 'eval'): st_model1.eval()
    if hasattr(st_model2, 'eval'): st_model2.eval()
    
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

    all_results = {
        'rmse': torch.empty(0, device=args.device),
        'rmv': torch.empty(0, device=args.device),
        'rrmse': torch.empty(0, device=args.device),
        'crps': torch.empty(0, device=args.device),
        'rcrps': torch.empty(0, device=args.device),
        'cov_diff': torch.empty(0, device=args.device),
        'rcov_diff': torch.empty(0, device=args.device),
        'pf_rmse': torch.empty(0, device=args.device),
    }
    loc_tensor_all_batches = None

    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)
            ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a += torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens
            
            cov_diff_list, rcov_diff_list, pf_rmse_list = [], [], []
            ens_list = [ens_v_a]
            loc_records = []
            
            obs_y_list = []
            for i in range(len(batch_v)):
                obs_y_step = H_fun(batch_v[i].unsqueeze(1))
                obs_y_step += args.sigma_y * torch.randn_like(obs_y_step, device=args.device)
                obs_y_list.append(obs_y_step)

            for i in range(len(batch_v) - 1):
                ens_v_a_forecast_input = ens_v_a.view(-1, args.ori_dim)
                for j_iter in range(args.dt_iter):
                    if args.dataset == 'ks':
                        ens_v_a_forecast_input = forward_fun(ens_v_a_forecast_input, None, args.dt / args.dt_iter)
                    else:
                        current_time_for_rk4 = i * args.dt + j_iter * (args.dt / args.dt_iter)
                        ens_v_a_forecast_input = rk4(forward_fun, ens_v_a_forecast_input, current_time_for_rk4, args.dt / args.dt_iter)
                ens_v_f = ens_v_a_forecast_input.view(-1, m, args.ori_dim)
                ens_v_f += torch.randn_like(ens_v_f, device=args.device) * args.sigma_v
                
                hv = H_fun(ens_v_f)
                obs_y = obs_y_list[i + 1]
                r_noise = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
                ens_i_innov = obs_y - hv - r_noise
                
                mean_hv = torch.mean(hv, dim=1, keepdim=True)
                mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True)
                
                ens_v_a, loc_nn_output = _process_analysis_step(
                    args, model_list, ens_v_f, hv, obs_y, ens_i_innov, mean_ens_v_f, mean_hv
                )
                
                if loc_nn_output is not None:
                    loc_records.append(loc_nn_output)
                    
                ens_v_a = post_process(ens_v_a, infl=infl)
                ens_list.append(ens_v_a)
                
                if args.pf_verification:
                    pf_mean_a = cached_pf_data[batch_ind]['means'][i]
                    pf_cov_ens_a = cached_pf_data[batch_ind]['covs'][i]

                    our_method_mean_a = torch.mean(ens_v_a, dim=1)
                    pf_rmse = torch.sqrt(torch.mean((our_method_mean_a - pf_mean_a)**2, dim=-1))
                    pf_rmse_list.append(pf_rmse)
                    
                    cov_ens_a = get_ens_cov(ens_v_a)
                    cov_diff = torch.norm(cov_ens_a - pf_cov_ens_a, p='fro', dim=(-2, -1)) 
                    rcov_diff = cov_diff / torch.norm(pf_cov_ens_a, p='fro', dim=(-2, -1))
                    cov_diff_list.append(cov_diff)
                    rcov_diff_list.append(rcov_diff)

            ens_tensor = torch.stack(ens_list)
            
            crps_val = torch.mean(compute_es(ens_states=ens_tensor, true_states=batch_v, norm_p=1), dim=0)
            rcrps_val = crps_val / torch.mean(torch.norm(batch_v, p=2, dim=2), dim=0)
            rmse_val = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)), dim=0)
            rms_val = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0)
            rrmse_val = rmse_val / rms_val
            rmv_val = torch.mean(torch.sqrt(args.N / (args.N - 1) * torch.mean((ens_tensor - batch_v.unsqueeze(2)) ** 2, dim=(2,3))),dim=0)
            
            all_results['rmse'] = torch.cat((all_results['rmse'], rmse_val))
            all_results['rmv'] = torch.cat((all_results['rmv'], rmv_val))
            all_results['rrmse'] = torch.cat((all_results['rrmse'], rrmse_val))
            all_results['crps'] = torch.cat((all_results['crps'], crps_val))
            all_results['rcrps'] = torch.cat((all_results['rcrps'], rcrps_val))
            if args.pf_verification:
                all_results['cov_diff'] = torch.cat((all_results['cov_diff'], torch.stack(cov_diff_list).mean(0)))
                all_results['rcov_diff'] = torch.cat((all_results['rcov_diff'], torch.stack(rcov_diff_list).mean(0)))
                all_results['pf_rmse'] = torch.cat((all_results['pf_rmse'], torch.stack(pf_rmse_list).mean(0)))
            
            if args.v != "EtE" and not args.no_localization and loc_records:
                current_loc_tensor = torch.stack(loc_records).unsqueeze(0)
                if loc_tensor_all_batches is None:
                    loc_tensor_all_batches = current_loc_tensor
                else:
                    try:
                        loc_tensor_all_batches = torch.cat((loc_tensor_all_batches, current_loc_tensor), dim=0)
                    except Exception as e:
                        print(f"Warning: Could not concatenate loc_tensors due to shape mismatch or other error: {e}")

            obs_tensor = torch.stack(obs_y_list).squeeze(2)
            observations = torch.full_like(batch_v, float('nan'), device=args.device)
            if hasattr(args, 'obs_inds') and args.obs_inds is not None:
                observations[:, :, args.obs_inds] = obs_tensor
            else:
                print("Warning: args.obs_inds not defined. Observations tensor might be all NaNs.")
    
    if plot_figures: 
        time_idx_plot = -2
        num_dims_plot = 4
        dim_indices_plot = list(range(min(args.ori_dim, num_dims_plot)))

        if args.pf_verification:
            batch_v = cached_pf_data[-1]['means']
            ens_tensor = ens_tensor[1:]
            observations = observations[1:]
        plot_particle_trajectories_with_histograms(particles=ens_tensor[:,time_idx_plot,:,:], 
                                                true_traj=batch_v[:,time_idx_plot,:], 
                                                observation=None, 
                                                dim_indices=dim_indices_plot,
                                                start_time=0,
                                                end_time=ens_tensor.shape[0],
                                                mode='quantile',
                                                save_fig=True,
                                                save_pdf=save_pdf,
                                                save_name=fig_name + "_hist",
                                                hist_step=1,
                                                fontsize=None)
        plot_particle_trajectories(particles=ens_tensor[:,time_idx_plot,:,:], 
                                true_traj=batch_v[:,time_idx_plot,:], 
                                observation=observations[:,time_idx_plot,:],
                                cmap_name='bwr',
                                start_time=0,
                                end_time=ens_tensor.shape[0], 
                                main_fig_size=(5, 2), 
                                save_fig=True,
                                save_pdf=save_pdf,
                                save_name=fig_name + "_traj",
                                colorbar_range=args.colorbar_range if hasattr(args, 'colorbar_range') else None,
                                plot_vertical_colorbar=False,
                                plot_horizontal_colorbar=True)

    final_metrics = {}
    
    if all_results['rrmse'].numel() == 0:
        metrics_keys = ['mean_rmse', 'std_rmse', 'mean_rmv', 'std_rmv', 
                        'mean_rrmse', 'std_rrmse', 'mean_crps', 'std_crps',
                        'mean_rcrps', 'std_rcrps', 'mean_cov_diff', 'std_cov_diff',
                        'mean_rcov_diff', 'std_rcov_diff', 'mean_pf_rmse', 'std_pf_rmse']
        final_metrics = {key: float('nan') for key in metrics_keys}
        final_metrics['no_nan_percent'] = 0.0
    else:
        nan_mask = torch.isnan(all_results['rrmse'])
        valid_B_mask = ~nan_mask
        
        if not valid_B_mask.any():
            metrics_keys = ['mean_rmse', 'std_rmse', 'mean_rmv', 'std_rmv', 
                            'mean_rrmse', 'std_rrmse', 'mean_crps', 'std_crps',
                            'mean_rcrps', 'std_rcrps', 'mean_cov_diff', 'std_cov_diff',
                            'mean_rcov_diff', 'std_rcov_diff', 'mean_pf_rmse', 'std_pf_rmse']
            final_metrics = {key: float('nan') for key in metrics_keys}
            final_metrics['no_nan_percent'] = 0.0
        else:
            final_metrics['mean_rrmse'], final_metrics['std_rrmse'] = get_mean_std(all_results['rrmse'][valid_B_mask])
            final_metrics['mean_rmse'], final_metrics['std_rmse'] = get_mean_std(all_results['rmse'][valid_B_mask])
            final_metrics['mean_rmv'], final_metrics['std_rmv'] = get_mean_std(all_results['rmv'][valid_B_mask])
            final_metrics['mean_crps'], final_metrics['std_crps'] = get_mean_std(all_results['crps'][valid_B_mask])
            final_metrics['mean_rcrps'], final_metrics['std_rcrps'] = get_mean_std(all_results['rcrps'][valid_B_mask])
            final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_results['rrmse'].numel() * 100.0
            if args.pf_verification:
                final_metrics['mean_cov_diff'], final_metrics['std_cov_diff'] = get_mean_std(all_results['cov_diff'][valid_B_mask])
                final_metrics['mean_rcov_diff'], final_metrics['std_rcov_diff'] = get_mean_std(all_results['rcov_diff'][valid_B_mask])
                final_metrics['mean_pf_rmse'], final_metrics['std_pf_rmse'] = get_mean_std(all_results['pf_rmse'][valid_B_mask])

    final_loc_tensor_to_return = loc_tensor_all_batches[0] if loc_tensor_all_batches is not None and loc_tensor_all_batches.shape[0] > 0 else torch.empty(1, device=args.device)
    final_metrics['loc_tensor'] = final_loc_tensor_to_return

    return final_metrics

def print_test_results(results):
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
    if 'mean_cov_diff' in results and 'std_cov_diff' in results:
        print(f"Cov-Diff: {results['mean_cov_diff']:.3f} ± {results['std_cov_diff']:.3f}")
    if 'mean_rcov_diff' in results and 'std_rcov_diff' in results:
        print(f"RCov-Diff: {results['mean_rcov_diff']:.3f} ± {results['std_rcov_diff']:.3f}")
    if 'mean_pf_rmse' in results and 'std_pf_rmse' in results:
        print(f"PF-RMSE: {results['mean_pf_rmse']:.3f} ± {results['std_pf_rmse']:.3f}")
    if 'no_nan_percent' in results:
        print(f"No NAN Percentage: {results['no_nan_percent']:.2f}%")


def test_ClassicFilter(loader, args, infl=1, H_info=None, plot_figures=True, fig_name='example_fig', loc_radius=None, save_pdf=False):
    m = args.N
    
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
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
        
    if args.no_localization:
        print("Do not use localization")
        loc_radius = None # Ensure loc_radius is None if no_localization is True
    
    rmse_tensor_all = None
    rmv_tensor_all = None
    rrmse_tensor_all = None
    crps_tensor_all = None

    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)

            ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

            ens_list = [ens_v_a]
            
            obs_y_step = H_fun(batch_v[0].unsqueeze(1))
            obs_y_step += args.sigma_y * torch.randn_like(obs_y_step, device=args.device)
            obs_y_list = [obs_y_step]

            for i in range(len(batch_v) - 1):
                obs_y = H_fun(batch_v[i + 1].unsqueeze(1))
                obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)
                obs_y_list.append(obs_y)

                ens_v_a_forecast_input = ens_v_a.view(-1, args.ori_dim)
                for j_iter in range(args.dt_iter):
                    if args.dataset == 'ks':
                        ens_v_a_forecast_input = forward_fun(ens_v_a_forecast_input, None, args.dt / args.dt_iter)
                    else:
                        current_time_for_rk4 = i * args.dt + j_iter * (args.dt / args.dt_iter)
                        ens_v_a_forecast_input = rk4(forward_fun, ens_v_a_forecast_input, current_time_for_rk4, args.dt / args.dt_iter)
                ens_v_f = ens_v_a_forecast_input.view(-1, m, args.ori_dim)
                
                ens_v_f = ens_v_f + torch.randn_like(ens_v_f, device=args.device) * args.sigma_v
                
                B_shape, N_ens, D_state = ens_v_f.shape
                d_obs_shape = obs_y.shape[2] # H_fun(ens_v_f).shape[2]
                                
                common_enkf_args = {
                    "observation_y": obs_y.squeeze(1),
                    "observation_operator_H_fun": H_fun, # Pass H_fun directly
                    "sigma_y": args.sigma_y,
                    "inflation_factor": infl
                }
                
                current_analyzed_ens_v_a = None
                if args.v == 'EnKF':
                    current_loc_mat_vy = dist2coeff(args.Lvy, radius=loc_radius, dim1=D_state, dim2=d_obs_shape, device=args.device).unsqueeze(0) if loc_radius is not None else None
                    current_loc_mat_yy = dist2coeff(args.Lyy, radius=loc_radius, dim1=d_obs_shape, dim2=d_obs_shape, device=args.device).unsqueeze(0) if loc_radius is not None else None
                    
                    current_analyzed_ens_v_a, _ = ensemble_kalman_filter_analysis(
                        ens_v_f, **common_enkf_args,
                        method='EnKF_PertObs', # Corrected method name based on typical EnKF implementations
                        localization_matrix_Lxy=current_loc_mat_vy, 
                        localization_matrix_Lyy=current_loc_mat_yy 
                    )
                elif args.v == 'ESRF':
                    current_loc_mat_vy = dist2coeff(args.Lvy, radius=loc_radius, dim1=D_state, dim2=d_obs_shape, device=args.device).unsqueeze(0) if loc_radius is not None else None
                    current_loc_mat_yy = dist2coeff(args.Lyy, radius=loc_radius, dim1=d_obs_shape, dim2=d_obs_shape, device=args.device).unsqueeze(0) if loc_radius is not None else None
                    current_analyzed_ens_v_a, _ = ensemble_kalman_filter_analysis(
                        ens_v_f, **common_enkf_args,
                        method='ESRF',
                        localization_matrix_Lxy=current_loc_mat_vy, # ESRF can also use localization
                        localization_matrix_Lyy=current_loc_mat_yy
                    )
                elif args.v == 'LETKF':
                    coords_state = torch.arange(D_state, device=args.device, dtype=batch_v.dtype).unsqueeze(1)
                    # Assuming obs_inds gives the indices of observed state variables
                    if hasattr(args, 'obs_inds') and args.obs_inds is not None:
                        coords_obs = torch.tensor(args.obs_inds, device=args.device, dtype=batch_v.dtype).unsqueeze(1)
                    else: # Fallback if obs_inds not defined, assume regularly spaced observations
                        coords_obs = torch.arange(0, D_state, int(D_state/d_obs_shape) if d_obs_shape > 0 else 1, device=args.device, dtype=batch_v.dtype).unsqueeze(1)
                        if coords_obs.shape[0] != d_obs_shape: # Adjust if division is not perfect
                            coords_obs = torch.linspace(0, D_state -1, steps=d_obs_shape,  device=args.device, dtype=batch_v.dtype).long().unsqueeze(1)


                    domain = torch.tensor([D_state], device=args.device, dtype=batch_v.dtype)

                    current_analyzed_ens_v_a, _ = ensemble_kalman_filter_analysis(
                        ens_v_f, **common_enkf_args,
                        method='LETKF',
                        localization_radius_letkf=loc_radius,
                        coords_state_letkf=coords_state,
                        coords_obs_letkf=coords_obs,
                        domain_letkf=domain
                    )
                else:
                    raise NotImplementedError(f"The filter {args.v} is not implemented")

                ens_v_a = torch.clamp(current_analyzed_ens_v_a, min=-args.clamp, max=args.clamp)
                ens_list.append(ens_v_a)

            ens_tensor = torch.stack(ens_list)
            obs_tensor = torch.stack(obs_y_list).squeeze(2)
            observations = torch.full_like(batch_v, float('nan'), device=args.device)

            if hasattr(args, 'obs_inds') and args.obs_inds is not None:
                observations[:,:,args.obs_inds] = obs_tensor
            else:
                print("Warning: args.obs_inds not defined for test_ClassicFilter. Observations tensor might be all NaNs.")
            
            crps_val = torch.mean(compute_es(ens_states=ens_tensor, true_states=batch_v, norm_p=1), dim=0) / torch.mean(torch.norm(batch_v, p=1, dim=2), dim=0)
            rmse_val = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)), dim=0)
            rms_val = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0)
            rrmse_val = rmse_val / rms_val
            rmv_val = torch.mean(torch.sqrt(N_ens / (N_ens-1) * torch.mean((ens_tensor - batch_v.unsqueeze(2)) ** 2, dim=(2,3))),dim=0)
            
            if batch_ind == 0:
                rmse_tensor_all = rmse_val
                rmv_tensor_all = rmv_val
                rrmse_tensor_all = rrmse_val
                crps_tensor_all = crps_val
            else:
                rmse_tensor_all = torch.cat((rmse_tensor_all, rmse_val))
                rmv_tensor_all = torch.cat((rmv_tensor_all, rmv_val))
                rrmse_tensor_all = torch.cat((rrmse_tensor_all, rrmse_val))
                crps_tensor_all = torch.cat((crps_tensor_all, crps_val))
            
        if plot_figures: 
            time_idx_plot = -2 
            num_dims_plot = 4
            dim_indices_plot = list(range(min(args.ori_dim, num_dims_plot)))

            plot_particle_trajectories_with_histograms(particles=ens_tensor[:,time_idx_plot,:,:], 
                                                    true_traj=batch_v[:,time_idx_plot,:], 
                                                    observation=None, #observations[:,time_idx_plot,:],
                                                    dim_indices=dim_indices_plot,
                                                    start_time=0,
                                                    end_time=ens_tensor.shape[0], 
                                                    mode='quantile',
                                                    save_fig=True,
                                                    save_pdf=save_pdf,
                                                    save_name=fig_name + "_hist_classic",
                                                    hist_step=1,
                                                    fontsize=None)
            plot_particle_trajectories(particles=ens_tensor[:,time_idx_plot,:,:], 
                                        true_traj=batch_v[:,time_idx_plot,:], 
                                        observation=observations[:,time_idx_plot,:],
                                        cmap_name='bwr',
                                        start_time=0,
                                        end_time=ens_tensor.shape[0], 
                                        main_fig_size=(5, 2), 
                                        save_fig=True,
                                        save_pdf=save_pdf,
                                        save_name=fig_name + "_traj_classic",
                                        colorbar_range=args.colorbar_range if hasattr(args, 'colorbar_range') else None,
                                        plot_vertical_colorbar=False,
                                        plot_horizontal_colorbar=True)

    if rrmse_tensor_all is None:
        return (float('nan'), float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'), float('nan'), 0.0)

    nan_mask = torch.isnan(rrmse_tensor_all) 
    valid_B_mask = ~nan_mask
    
    if not valid_B_mask.any():
        mean_rrmse, std_rrmse = float('nan'), float('nan')
        mean_rmse, std_rmse = float('nan'), float('nan')
        mean_rmv, std_rmv = float('nan'), float('nan')
        mean_crps, std_crps = float('nan'), float('nan')
        no_nan_percent = 0.0
    else:
        mean_rrmse, std_rrmse = get_mean_std(rrmse_tensor_all[valid_B_mask])
        mean_rmse, std_rmse = get_mean_std(rmse_tensor_all[valid_B_mask])
        mean_rmv, std_rmv = get_mean_std(rmv_tensor_all[valid_B_mask])
        mean_crps, std_crps = get_mean_std(crps_tensor_all[valid_B_mask])
        no_nan_percent = torch.sum(valid_B_mask).float() / rrmse_tensor_all.numel() * 100.0
        
    return mean_rmse, std_rmse, mean_rmv, std_rmv, mean_rrmse, std_rrmse, mean_crps, std_crps, no_nan_percent


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
                        normalize_val = torch.std(batch_v, dim=0)
                        # normalize_val = None
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

def test_model_v2(loader, model_list, args, plot_figures=True, fig_name='example_fig_v2', save_pdf=False):
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
                    args, model_list, ens_v_f, hv, obs_y, ens_i_innov, mean_ens_v_f, mean_hv, sigma_y=sigma_y
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