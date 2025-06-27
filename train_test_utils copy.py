import numpy as np
import time

import torch
import torch.nn as nn

from utils import L63, L96, rk4, etd_rk4_wrapper
from utils import AverageMeter, mystery_operator, get_mean_std
from utils import post_process, mean0
from visualization import plot_particle_trajectories_with_histograms, plot_particle_trajectories
from localization import dist2coeff, create_loc_mat
from loss import compute_loss, compute_es
from networks import NaiveNetwork, SetTransformer, Simple_MLP
from benchmark_analysis import ensemble_kalman_filter_analysis

def train_model(epoch, loader, model_list, optimizer, scheduler, args, H_info=None):
    model, infl_model, local_model, st_model1, st_model2 = model_list
    
    m = args.N

    losses = AverageMeter()
    batch_time = AverageMeter()
    
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
    elif args.dataset == "lorenz96":
        forward_fun = L96.forward
    elif args.dataset == "ks":
        forward_fun = etd_rk4_wrapper(device = args.device, dt = args.dt / args.dt_iter)
    else:
        raise NotImplementedError
    
    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info

    model.train()
    
    success_count = 0
    total_count = 0
    num_all_nan_batch = 0
    for batch_ind, batch_v in enumerate(loader):
        t_start = time.time()
        batch_v = batch_v.to(device=args.device)

        # Sample from prior
        ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
        ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

        # Store everything
        ens_list = [ens_v_a]

        # Iterate over timesteps in batch
        if args.loss_warm_up:
            end_ind = np.minimum(epoch + 1, len(batch_v) - 1)
        else:
            end_ind = len(batch_v) - 1
        
        for i in range(end_ind):
            # get next observation
            obs_y = H_fun(batch_v[i + 1].unsqueeze(1))
            obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)

            # forecast step
            ens_v_a = ens_v_a.view(-1, args.ori_dim)
            for j in range(args.dt_iter):
                if args.dataset == 'ks':
                    ens_v_a = forward_fun(ens_v_a, None, args.dt / args.dt_iter)
                else:
                    ens_v_a = rk4(forward_fun, ens_v_a, i * args.dt + j * args.dt / args.dt_iter,
                                args.dt / args.dt_iter)
            ens_v_f = ens_v_a.view(-1, m, args.ori_dim)
            # add forward noise
            ens_v_f = ens_v_f + torch.randn_like(ens_v_f, device=args.device) * args.sigma_v

            # preparation for individual ensemble data
            hv = H_fun(ens_v_f)
            B, N, D = ens_v_f.shape
            d = hv.shape[2]
            
            # generate a random variable for the observation noise
            r = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
            
            ens_i = obs_y - hv - r
            
            # for the ensemble dataset and observations
            mean_hv = torch.mean(hv, dim=1, keepdim=True).expand(-1, N, -1)
            mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True).expand(-1, N, -1)
            
            if args.v == 'CorrTerms':
                # st and following nn inputs
                if args.st_type == 'state_only':
                    ens_nn_output = st_model1(ens_v_f)
                    nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                        ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                    if args.obs_in_loc:
                        local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output], dim=-1)
                    else:
                        local_nn_input = ens_nn_output
                    infl_nn_input = torch.cat([ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.ori_dim + args.st_output_dim)
                elif args.st_type == 'separate':
                    ens_nn_output = st_model1(ens_v_f)
                    ens_o_nn_output = st_model2(hv)                
                    nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                        ens_nn_output.unsqueeze(1).expand(-1, N, -1), ens_o_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                    if args.obs_in_loc:
                        local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output, ens_o_nn_output], dim=-1)
                    else:
                        local_nn_input = torch.cat([ens_nn_output, ens_o_nn_output], dim=-1)
                    infl_nn_input = torch.cat([ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.ori_dim + args.st_output_dim)
                elif args.st_type == 'joint':
                    ens_nn_output = st_model1(torch.cat([ens_v_f, hv], dim=-1))
                    nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                        ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                    if args.obs_in_loc:
                        local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output], dim=-1)
                    else:
                        local_nn_input = ens_nn_output
                    infl_nn_input = torch.cat([ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.ori_dim + 2 * args.st_output_dim)

                # execute model
                nn_output = model(nn_input).view(hv.shape[0], hv.shape[1], -1)
                infl_output = infl_model(infl_nn_input).view(B, N, -1)
                Vnn1 = ens_v_f + infl_output
                Vnn2 = ens_v_f - mean_ens_v_f + nn_output[:, :, :D]
                Ynn = hv - mean_hv + nn_output[:, :, D:]
                R = args.sigma_y ** 2 * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
                
                # get localization matrices
                if args.no_localization:
                    K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) 
                    K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) + R * (N - 1)
                else:
                    loc_nn_output = torch.sigmoid(local_model(local_nn_input)) * args.loc_max_val
                    loc_mat_vy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lvy)
                    loc_mat_yy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lyy)
                    
                    K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) * loc_mat_vy
                    K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) * loc_mat_yy + R * (N - 1)
                
                # Kalman Gain
                K = torch.bmm(K1, torch.inverse(K2))
                ens_v_a = Vnn1 + torch.bmm(ens_i, K.transpose(1, 2))
            elif args.v == 'EtE':
                if args.st_type == 'state_only':
                    ens_nn_output = st_model1(ens_v_f)
                    nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                        ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                elif args.st_type == 'separate':
                    ens_nn_output = st_model1(ens_v_f)
                    ens_o_nn_output = st_model2(hv)                
                    nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                        ens_nn_output.unsqueeze(1).expand(-1, N, -1), ens_o_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                elif args.st_type == 'joint':
                    ens_nn_output = st_model1(torch.cat([ens_v_f, hv], dim=-1))
                    nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                        ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                    
                # execute model
                nn_output = model(nn_input).view(B, N, -1)
                ens_v_a = nn_output
                

            ens_v_a = torch.clamp(ens_v_a, min=-args.clamp, max=args.clamp)

            ens_list.append(ens_v_a)
            
            if epoch <= args.detach_training_epoch:
                ens_v_a = ens_v_a.detach()

        # Concat outputs
        ens_tensor = torch.stack(ens_list)

        # Loss functions
        if args.loss_warm_up:
            if epoch >= len(batch_v) - 1:
                ignore_first = args.ignore_first
            else:
                ignore_first = 0
        else:
            ignore_first = args.ignore_first
        
        # remove nan batch indices
        nan_mask = torch.isnan(ens_tensor).any(dim=(0, 2, 3))  
        valid_B_mask = ~nan_mask

        # loss
        if not valid_B_mask.any():
            num_all_nan_batch += 1
        else:
            loss = 0
            for loss_type in args.loss_type:
                loss += compute_loss(ens_tensor=ens_tensor, 
                                    batch_v=batch_v, 
                                    loss_type=loss_type, 
                                    ignore_first=ignore_first, 
                                    end_ind=None, 
                                    valid_B_mask=valid_B_mask,
                                    norm_p=args.es_p,
                                    kes_sigma=args.kes_sigma)

            success_count += torch.sum(valid_B_mask)
            
            losses.update(loss.item(), torch.sum(valid_B_mask))

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_count += batch_v.shape[1]

        # batch time
        batch_time.update(time.time() - t_start)
        
        if num_all_nan_batch == len(loader):
            # raise RuntimeError("All batches resulted in NaN loss. Stopping training.")
            loss = torch.tensor(float('nan')) 
            losses.update(loss.item(), 1)

        if (batch_ind + 1) % args.print_batch == 0:
            print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
                f'Batch time {batch_time.val:.3f} (Avg: {batch_time.avg:.3f})\t'
                f'Loss {losses.val:.3f} (Avg: {losses.avg:.3f})\t'
                f'Current learning rate: {optimizer.param_groups[0]["lr"]:.2e}\t'
                f'No NAN Percentage: {success_count / total_count * 100: .2f}%\t'
                )

    scheduler.step()
    return losses.avg

def test_model(loader, model_list, args, infl=1, H_info=None, plot_figures=True, fig_name='example_fig', save_pdf=False):
    model, infl_model, local_model, st_model1, st_model2 = model_list

    m = args.N
    
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
    elif args.dataset == "lorenz96":
        forward_fun = L96.forward
    elif args.dataset == "ks":
        forward_fun = etd_rk4_wrapper(device = args.device, dt = args.dt / args.dt_iter)
    else:
        raise NotImplementedError

    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info
        
    # loc_mat_vy = dist2coeff(args.Lvy, radius=4).unsqueeze(0)
    # loc_mat_yy = dist2coeff(args.Lyy, radius=4).unsqueeze(0)

    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)

            # Sample from prior
            ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

            # Store everything
            ens_list = [ens_v_a]
            loc_records = []
            
            obs_y = H_fun(batch_v[0].unsqueeze(1))
            obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)
            obs_y_list = [obs_y]

            # Iterate over timesteps in batch
            for i in range(len(batch_v) - 1):
                t_start = time.time()
                # get next observation
                obs_y = H_fun(batch_v[i + 1].unsqueeze(1))
                obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)
                obs_y_list.append(obs_y)

                # forecast step
                ens_v_a = ens_v_a.view(-1, args.ori_dim)
                for j in range(args.dt_iter):
                    if args.dataset == 'ks':
                        ens_v_a = forward_fun(ens_v_a, None, args.dt / args.dt_iter)
                    else:
                        ens_v_a = rk4(forward_fun, ens_v_a, i * args.dt + j * args.dt / args.dt_iter,
                                args.dt / args.dt_iter)
                ens_v_f = ens_v_a.view(-1, m, args.ori_dim)
                
                # add forward noise
                ens_v_f = ens_v_f + torch.randn_like(ens_v_f, device=args.device) * args.sigma_v

                # preparation for individual ensemble data
                hv = H_fun(ens_v_f)
                
                # ens_v_a, K = EnKF_analysis(ens_v_f, hv, obs_y, args.sigma_y, a_method="PertObs")
                B, N, D = ens_v_f.shape
                d = hv.shape[2]
                
                # generate a random variable for the observation noise
                r = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
                
                ens_i = obs_y - hv - r
                
                # for the ensemble dataset and observations
                mean_hv = torch.mean(hv, dim=1, keepdim=True).expand(-1, N, -1)
                mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True).expand(-1, N, -1)
                
                if args.v == 'CorrTerms':
                    # st and following nn inputs
                    if args.st_type == 'state_only':
                        ens_nn_output = st_model1(ens_v_f)
                        nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                            ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                        if args.obs_in_loc:
                            local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output], dim=-1)
                        else:
                            local_nn_input = ens_nn_output
                        infl_nn_input = torch.cat([ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.ori_dim + args.st_output_dim)
                    elif args.st_type == 'separate':
                        ens_nn_output = st_model1(ens_v_f)
                        ens_o_nn_output = st_model2(hv)                
                        nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                            ens_nn_output.unsqueeze(1).expand(-1, N, -1), ens_o_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                        if args.obs_in_loc:
                            local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output, ens_o_nn_output], dim=-1)
                        else:
                            local_nn_input = torch.cat([ens_nn_output, ens_o_nn_output], dim=-1)
                        infl_nn_input = torch.cat([ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.ori_dim + args.st_output_dim)
                    elif args.st_type == 'joint':
                        ens_nn_output = st_model1(torch.cat([ens_v_f, hv], dim=-1))
                        nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                            ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                        if args.obs_in_loc:
                            local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output], dim=-1)
                        else:
                            local_nn_input = ens_nn_output
                        infl_nn_input = torch.cat([ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.ori_dim + 2 * args.st_output_dim)
                        
                    # execute model
                    nn_output = model(nn_input).view(hv.shape[0], hv.shape[1], -1)
                    infl_output = infl_model(infl_nn_input).view(B, N, -1)
                    if args.zero_infl:
                        Vnn1 = ens_v_f
                    else:
                        Vnn1 = ens_v_f + infl_output
                    
                    # Vnn1 = ens_v_f
                    Vnn2 = ens_v_f - mean_ens_v_f + nn_output[:, :, :D]
                    Ynn = hv - mean_hv + nn_output[:, :, D:]
                    R = args.sigma_y ** 2 * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
                    
                    # get localization matrices
                    if args.no_localization:
                        loc_mat_vy = torch.ones(B, D, d, device=args.device)
                        loc_mat_yy = torch.ones(B, d, d, device=args.device)
                    else:
                        loc_nn_output = torch.sigmoid(local_model(local_nn_input)) * args.loc_max_val
                        loc_mat_vy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lvy)
                        loc_mat_yy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lyy)
                        loc_records.append(loc_nn_output)
                    
                    # Kalman Gain
                    K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) * loc_mat_vy
                    K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) * loc_mat_yy + R * (N - 1)
                    K = torch.bmm(K1, torch.inverse(K2))
                    ens_v_a = Vnn1 + torch.bmm(ens_i, K.transpose(1, 2))
                    
                    ens_v_a = post_process(ens_v_a, infl=infl)
                elif args.v == 'EtE':
                    if args.st_type == 'state_only':
                        ens_nn_output = st_model1(ens_v_f)
                        nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                            ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                    elif args.st_type == 'separate':
                        ens_nn_output = st_model1(ens_v_f)
                        ens_o_nn_output = st_model2(hv)                
                        nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                            ens_nn_output.unsqueeze(1).expand(-1, N, -1), ens_o_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                    elif args.st_type == 'joint':
                        ens_nn_output = st_model1(torch.cat([ens_v_f, hv], dim=-1))
                        nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
                                            ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                        
                    # execute model
                    nn_output = model(nn_input).view(B, N, -1)
                    ens_v_a = nn_output
                    
                ens_v_a = torch.clamp(ens_v_a, min=-args.clamp, max=args.clamp)

                ens_list.append(ens_v_a)

            # Concat outputs
            ens_tensor = torch.stack(ens_list)
            if args.v == "EtE":
                loc_tensor = None
            else:
                if args.no_localization:
                    loc_tensor = torch.empty(1)
                else:
                    loc_tensor = torch.stack(loc_records)
                    
            obs_tensor = torch.stack(obs_y_list).squeeze(2)
            observations = torch.full(batch_v.shape, float('nan')).to(args.device)
            print(observations.shape)
            observations[:,:,args.obs_inds] = obs_tensor
            
            # Loss functions
            # absolute rmse
            crps_tensor = torch.mean(compute_es(ens_states=ens_tensor, true_states=batch_v, norm_p=1), dim=0) / torch.mean(torch.norm(batch_v, p=1, dim=2), dim=0)
            rmse_tensor = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)), dim=0)
            rms_tensor = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0)
            rrmse_tensor = rmse_tensor / rms_tensor
            # relative rmse
            # rmse_tensor = torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)) / torch.sqrt(torch.mean((batch_v) ** 2, dim=2))
            rmv_tensor = torch.mean(torch.sqrt(N / (N-1) * torch.mean((ens_tensor - batch_v.unsqueeze(2)) ** 2, dim=(2,3))),dim=0)
            
            if batch_ind == 0:
                rmse_tensor_all, rmv_tensor_all, rrmse_tensor_all, crps_tensor_all = rmse_tensor, rmv_tensor, rrmse_tensor, crps_tensor
            else:
                # Concatenate tensors along the first dimension
                rmse_tensor_all = torch.cat((rmse_tensor_all, rmse_tensor))
                rmv_tensor_all = torch.cat((rmv_tensor_all, rmv_tensor))
                rrmse_tensor_all = torch.cat((rrmse_tensor_all, rrmse_tensor))
                crps_tensor_all = torch.cat((crps_tensor_all, crps_tensor))
            
        if plot_figures:
            plot_particle_trajectories_with_histograms(particles=ens_tensor[:,-2,:,:], 
                                                    true_traj=batch_v[:,-2,:], 
                                                    # observation=observations[:,-2,:],
                                                    observation=None,
                                                    dim_indices=[0, 1, 2, 3],
                                                    start_time=args.test_steps-100,
                                                    end_time=args.test_steps, 
                                                    mode='quantile',
                                                    save_fig=True,
                                                    save_pdf=save_pdf,
                                                    save_name=fig_name,
                                                    hist_step=1,
                                                    fontsize=None)
            plot_particle_trajectories(particles=ens_tensor[:,-2,:,:], 
                                        true_traj=batch_v[:,-2,:], 
                                        observation=observations[:,-2,:],
                                        cmap_name='bwr',
                                        start_time=args.test_steps-100,
                                        end_time=args.test_steps, 
                                        main_fig_size=(5, 2), 
                                        save_fig=True,
                                        save_pdf=save_pdf,
                                        save_name=fig_name,
                                        colorbar_range=args.colorbar_range,
                                        plot_vertical_colorbar=False,
                                        plot_horizontal_colorbar=True)
        # non-nan trajs
        nan_mask = torch.isnan(rrmse_tensor_all) 
        valid_B_mask = ~nan_mask
        
        mean_rrmse, std_rrmse = get_mean_std(rrmse_tensor_all[valid_B_mask])
        mean_rmse, std_rmse = get_mean_std(rmse_tensor_all[valid_B_mask])
        mean_rmv, std_rmv = get_mean_std(rmv_tensor_all[valid_B_mask])
        mean_crps, std_crps = get_mean_std(crps_tensor_all[valid_B_mask])
        
        no_nan_percent = torch.sum(valid_B_mask) / args.test_traj_num

    return mean_rmse, std_rmse, mean_rmv, std_rmv, mean_rrmse, std_rrmse, mean_crps, std_crps, no_nan_percent, loc_tensor

def set_models(args):
    # set models
    if args.v == 'CorrTerms':
        model = Simple_MLP(d_input=args.input_dim, d_output=args.obs_dim + args.ori_dim, num_hidden_layers=2).to(args.device)
    elif args.v == 'EtE':
        model = Simple_MLP(d_input=args.input_dim, d_output=args.ori_dim, num_hidden_layers=3).to(args.device)
    if args.no_localization or args.v == 'EtE':
        local_model = NaiveNetwork(1)
    else:
        local_model = Simple_MLP(d_input=args.local_input_dim, d_output=args.num_dist, num_hidden_layers=2).to(args.device)
    if args.st_type == 'separate':
        st_model1 = SetTransformer(input_dim=args.ori_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                    hidden_dim=args.hidden_dim, num_layers=1, freeze_WQ=not args.unfreeze_WQ).to(args.device)
        st_model2 = SetTransformer(input_dim=args.obs_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                    hidden_dim=args.hidden_dim, num_layers=1, freeze_WQ=not args.unfreeze_WQ).to(args.device)
        if args.v == 'EtE':
            infl_model = NaiveNetwork(1)
        else:
            infl_model = Simple_MLP(d_input=args.ori_dim + args.st_output_dim, d_output=args.ori_dim, num_hidden_layers=2).to(args.device)
    elif args.st_type == 'state_only':
        st_model1 = SetTransformer(input_dim=args.ori_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                    hidden_dim=args.hidden_dim, num_layers=2, freeze_WQ=not args.unfreeze_WQ).to(args.device)
        st_model2 = NaiveNetwork(1)
        if args.v == 'EtE':
            infl_model = NaiveNetwork(1)
        else:
            infl_model = Simple_MLP(d_input=args.ori_dim + args.st_output_dim, d_output=args.ori_dim, num_hidden_layers=2).to(args.device)
    elif args.st_type == 'joint':
        st_model1 = SetTransformer(input_dim=args.ori_dim + args.obs_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim * 2, 
                                    hidden_dim=args.hidden_dim, num_layers=2, freeze_WQ=not args.unfreeze_WQ).to(args.device)
        st_model2 = NaiveNetwork(1)
        if args.v == 'EtE':
            infl_model = NaiveNetwork(1)
        else:
            infl_model = Simple_MLP(d_input=args.ori_dim + 2 * args.st_output_dim, d_output=args.ori_dim, num_hidden_layers=2).to(args.device)
    if args.use_data_parallel:
        model, infl_model, local_model, st_model1, st_model2 = \
            nn.DataParallel(model), nn.DataParallel(infl_model), nn.DataParallel(local_model), nn.DataParallel(st_model1), nn.DataParallel(st_model2)
    model_list = [model, infl_model, local_model, st_model1, st_model2]
    total_params = sum(sum(p.numel() for p in model.parameters()) for model in model_list)
    print(f'Total number of parameters: {total_params}')
    
    return model_list



def test_ClassicFilter(loader, args, infl=1, H_info=None, plot_figures=True, fig_name='example_fig', loc_radius=None, save_pdf=False):

    m = args.N
    
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
    elif args.dataset == "lorenz96":
        forward_fun = L96.forward
    elif args.dataset == "ks":
        forward_fun = etd_rk4_wrapper(device = args.device, dt = args.dt / args.dt_iter)
    else:
        raise NotImplementedError

    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info
        
    if args.no_localization:
        print("Do not use localization")
        loc_radius = None
        
    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)

            # Sample from prior
            ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

            # Store everything
            ens_list = [ens_v_a]
            
            obs_y = H_fun(batch_v[0].unsqueeze(1))
            obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)
            obs_y_list = [obs_y]

            # Iterate over timesteps in batch
            for i in range(len(batch_v) - 1):
                t_start = time.time()
                # get next observation
                obs_y = H_fun(batch_v[i + 1].unsqueeze(1))
                obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)
                obs_y_list.append(obs_y)

                # forecast step
                ens_v_a = ens_v_a.view(-1, args.ori_dim)
                for j in range(args.dt_iter):
                    if args.dataset == 'ks':
                        ens_v_a = forward_fun(ens_v_a, None, args.dt / args.dt_iter)
                    else:
                        ens_v_a = rk4(forward_fun, ens_v_a, i * args.dt + j * args.dt / args.dt_iter,
                                args.dt / args.dt_iter)
                ens_v_f = ens_v_a.view(-1, m, args.ori_dim)
                
                # add forward noise
                ens_v_f = ens_v_f + torch.randn_like(ens_v_f, device=args.device) * args.sigma_v

                # preparation for individual ensemble data
                hv = H_fun(ens_v_f)
                
                # ens_v_a, K = EnKF_analysis(ens_v_f, hv, obs_y, args.sigma_y, a_method="PertObs")
                B, N, D = ens_v_f.shape
                d = hv.shape[2]
                
                
                common_enkf_args = {
                        "observation_y": obs_y.squeeze(1),
                        "observation_operator_ens": H_fun,
                        "sigma_y": args.sigma_y,
                        "inflation_factor": infl
                    }
                
                if args.v == 'EnKF':
                    loc_mat_vy = dist2coeff(args.Lvy, radius=loc_radius).unsqueeze(0)
                    loc_mat_yy = dist2coeff(args.Lyy, radius=loc_radius).unsqueeze(0)
                    
                    ens_v_a, _ = ensemble_kalman_filter_analysis(
                        ens_v_f, **common_enkf_args,
                        method='EnKF-PertObs',
                        localization_matrix_Lxy=loc_mat_vy, 
                        localization_matrix_Lyy=loc_mat_yy 
                    )
                elif args.v == 'ESRF':
                    ens_v_a, _ = ensemble_kalman_filter_analysis(
                        ens_v_f, **common_enkf_args,
                        method='ESRF'
                    )
                elif args.v == 'LETKF':
                    coords_state = torch.arange(D, device=args.device, dtype=batch_v.dtype).unsqueeze(1) # (D, 1)
                    coords_obs = torch.arange(0, D, int(D/d), device=args.device, dtype=batch_v.dtype).unsqueeze(1) # (d, 1)
                    domain = torch.tensor([D], device=args.device, dtype=batch_v.dtype) # (1,)

                    ens_v_a, _ = ensemble_kalman_filter_analysis(
                        ens_v_f, **common_enkf_args,
                        method='LETKF',
                        localization_radius_letkf=loc_radius,
                        coords_state_letkf=coords_state,
                        coords_obs_letkf=coords_obs,
                        domain_letkf=domain
                    )
                else:
                    raise NotImplementedError(f"The filter {args.v} is not implemented")

                    
                ens_v_a = torch.clamp(ens_v_a, min=-args.clamp, max=args.clamp)

                ens_list.append(ens_v_a)

            # Concat outputs
            ens_tensor = torch.stack(ens_list)
                    
            obs_tensor = torch.stack(obs_y_list).squeeze(2)
            observations = torch.full(batch_v.shape, float('nan')).to(args.device)
            print(observations.shape)
            observations[:,:,args.obs_inds] = obs_tensor
            
            # Loss functions
            # absolute rmse
            crps_tensor = torch.mean(compute_es(ens_states=ens_tensor, true_states=batch_v, norm_p=1), dim=0) / torch.mean(torch.norm(batch_v, p=1, dim=2), dim=0)
            rmse_tensor = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)), dim=0)
            rms_tensor = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0)
            rrmse_tensor = rmse_tensor / rms_tensor
            # relative rmse
            # rmse_tensor = torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)) / torch.sqrt(torch.mean((batch_v) ** 2, dim=2))
            rmv_tensor = torch.mean(torch.sqrt(N / (N-1) * torch.mean((ens_tensor - batch_v.unsqueeze(2)) ** 2, dim=(2,3))),dim=0)
            
            if batch_ind == 0:
                rmse_tensor_all, rmv_tensor_all, rrmse_tensor_all, crps_tensor_all = rmse_tensor, rmv_tensor, rrmse_tensor, crps_tensor
            else:
                # Concatenate tensors along the first dimension
                rmse_tensor_all = torch.cat((rmse_tensor_all, rmse_tensor))
                rmv_tensor_all = torch.cat((rmv_tensor_all, rmv_tensor))
                rrmse_tensor_all = torch.cat((rrmse_tensor_all, rrmse_tensor))
                crps_tensor_all = torch.cat((crps_tensor_all, crps_tensor))
            
        if plot_figures:
            print(ens_tensor.shape, batch_v.shape)
            plot_particle_trajectories_with_histograms(particles=ens_tensor[:,-2,:,:], 
                                                    true_traj=batch_v[:,-2,:], 
                                                    # observation=observations[:,-2,:],
                                                    observation=None,
                                                    dim_indices=[0, 1, 2, 3],
                                                    start_time=args.test_steps-100,
                                                    end_time=args.test_steps, 
                                                    mode='quantile',
                                                    save_fig=True,
                                                    save_pdf=save_pdf,
                                                    save_name=fig_name,
                                                    hist_step=1,
                                                    fontsize=None)
            plot_particle_trajectories(particles=ens_tensor[:,-2,:,:], 
                                        true_traj=batch_v[:,-2,:], 
                                        observation=observations[:,-2,:],
                                        cmap_name='bwr',
                                        start_time=args.test_steps-100,
                                        end_time=args.test_steps, 
                                        main_fig_size=(5, 2), 
                                        save_fig=True,
                                        save_pdf=save_pdf,
                                        save_name=fig_name,
                                        colorbar_range=args.colorbar_range,
                                        plot_vertical_colorbar=False,
                                        plot_horizontal_colorbar=True)
        # non-nan trajs
        nan_mask = torch.isnan(rrmse_tensor_all) 
        valid_B_mask = ~nan_mask
        
        mean_rrmse, std_rrmse = get_mean_std(rrmse_tensor_all[valid_B_mask])
        mean_rmse, std_rmse = get_mean_std(rmse_tensor_all[valid_B_mask])
        mean_rmv, std_rmv = get_mean_std(rmv_tensor_all[valid_B_mask])
        mean_crps, std_crps = get_mean_std(crps_tensor_all[valid_B_mask])
        
        no_nan_percent = torch.sum(valid_B_mask) / args.test_traj_num

    return mean_rmse, std_rmse, mean_rmv, std_rmv, mean_rrmse, std_rrmse, mean_crps, std_crps, no_nan_percent



