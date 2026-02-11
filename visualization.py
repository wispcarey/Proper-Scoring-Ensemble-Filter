import torch
import numpy as np
from typing import Optional, List, Tuple
import pingouin as pg

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os


from argparse import Namespace 

def plot_particle_trajectories_with_histograms(
    particles: torch.Tensor,
    true_traj: torch.Tensor,
    observation: Optional[torch.Tensor],
    dim_indices: List[int],
    start_time: int = 0,
    end_time: Optional[int] = None,
    mode: str = 'width',
    save_fig: bool = False,
    save_pdf: bool = False,
    save_name: str = 'example_fig',
    hist_step: int = 1,
    fontsize: Optional[int] = 20,
    figsize: Tuple[float, float] = (14, 4)
):
    """
    Plots particle trajectories along specified dimensions with overlaid histograms
    showing the distribution of particles at different time steps.

    Args:
        particles (torch.Tensor): Tensor of shape (J, N, d) representing the
            particle trajectories, where J is the number of time steps, N is the
            number of particles, and d is the number of dimensions.
        true_traj (torch.Tensor): Tensor of shape (J, d) representing the true
            trajectory.
        observation (Optional[torch.Tensor]): Tensor of shape (J, d) representing
            the observations. If None, observations will not be plotted.
        dim_indices (List[int]): List of integer indices specifying the dimensions
            to plot.
        start_time (int, optional): The starting time step for plotting. Defaults to 0.
        end_time (Optional[int], optional): The ending time step for plotting.
            If None, plots up to the last time step. Defaults to None.
        mode (str, optional): The mode for displaying the particle distribution.
            Can be 'width' (histogram width proportional to density), 'color'
            (histogram colored by density), 'std' (mean ± 1 standard deviation),
            or 'quantile' (95% confidence interval assuming Gaussian distribution).
            Defaults to 'width'.
        save_fig (bool, optional): If True, saves the plot as a PNG file.
            Defaults to False.
        save_pdf (bool, optional): If True, saves the plot as a PDF file.
            Defaults to False.
        save_name (str, optional): The base name for the saved figure files.
            Defaults to 'example_fig'.
        hist_step (int, optional): The time step interval for plotting histograms.
            Defaults to 1.
        fontsize (Optional[int], optional): The font size for plot labels and
            ticks. If None, grid, ticks, and labels will be removed from the
            main plots (legend is unaffected). Defaults to 20.
        figsize (Tuple[float, float], optional): The size of the figure (width, height)
            in inches. Defaults to (13, 5).
    """

    J_p, N, d_p = particles.shape
    J_t, d_t = true_traj.shape
    if observation is not None:
        J_o, d_o = observation.shape
        if not (J_p == J_t == J_o):
            raise ValueError("All input tensors (particles, true_traj, observation) must have the same number of time steps (J).")
        if not (d_p == d_t == d_o):
            raise ValueError("All input tensors (particles, true_traj, observation) must have the same number of dimensions (d).")
    else:
        if not (J_p == J_t):
            raise ValueError("particles and true_traj must have the same number of time steps (J).")
        if not (d_p == d_t):
            raise ValueError("particles and true_traj must have the same number of dimensions (d).")

    J, d_model = J_p, d_p

    plot_start_time = start_time
    plot_end_time = end_time if end_time is not None else J

    if plot_start_time < 0:
        print(f"Warning: start_time ({plot_start_time}) is negative. Using 0.")
        plot_start_time = 0

    if J == 0:
        print("Error: No data available (J=0). Cannot plot.")
        return

    if plot_start_time >= J:
        print(f"Error: start_time ({plot_start_time}) out of bounds (>= {J}).")
        return

    if plot_end_time > J:
        print(f"Warning: end_time ({plot_end_time}) exceeds total time steps ({J}). Using {J}.")
        plot_end_time = J

    if plot_end_time <= plot_start_time:
        print(f"Error: plot_end_time ({plot_end_time}) must be greater than plot_start_time ({plot_start_time}).")
        return

    particles_cpu = particles.detach().cpu()
    true_traj_cpu = true_traj.detach().cpu()
    observation_cpu = observation.detach().cpu() if observation is not None else None

    time_steps_for_plot_np = torch.arange(plot_start_time, plot_end_time).numpy()
    step_width = 1.0
    all_handles = []
    all_labels = []

    for dim_idx in dim_indices:
        if not (0 <= dim_idx < d_model):
            print(f"Warning: Dimension index {dim_idx} out of bounds (0 to {d_model-1}). Skipping.")
            continue

        fig, ax = plt.subplots(figsize=figsize)
        ensemble_spread_labeled = False
        handles = []
        labels = []

        line_true, = ax.plot(time_steps_for_plot_np,
                                 true_traj_cpu[plot_start_time:plot_end_time, dim_idx].numpy(),
                                 label='True Trajectory', color='blue', linewidth=1.5)
        if 'True Trajectory' not in labels:
            handles.append(line_true)
            labels.append('True Trajectory')

        if observation_cpu is not None:
            marker_obs, = ax.plot(time_steps_for_plot_np,
                                     observation_cpu[plot_start_time:plot_end_time, dim_idx].numpy(),
                                     '*', label='Observation', color='green', markersize=8)
            if 'Observation' not in labels:
                handles.append(marker_obs)
                labels.append('Observation')

        if N > 0:
            particle_mean_slice = particles_cpu[plot_start_time:plot_end_time, :, dim_idx].mean(dim=1)
            line_mean, = ax.plot(time_steps_for_plot_np, particle_mean_slice.numpy(),
                                 label='Ensemble Mean', color='red', linestyle='--', linewidth=1.5)
            if 'Ensemble Mean' not in labels:
                handles.append(line_mean)
                labels.append('Ensemble Mean')

        if N > 0:
            if mode == 'std':
                # Calculate mean and standard deviation for each time step
                particle_mean = particles_cpu[plot_start_time:plot_end_time, :, dim_idx].mean(dim=1)
                particle_std = particles_cpu[plot_start_time:plot_end_time, :, dim_idx].std(dim=1)
                
                # Plot mean ± 1 std
                upper_bound = particle_mean + particle_std
                lower_bound = particle_mean - particle_std
                
                ax.fill_between(time_steps_for_plot_np, 
                               lower_bound.numpy(), 
                               upper_bound.numpy(),
                               alpha=0.3, color='red', 
                               label='Mean ± 1 STD')
                
                if 'Mean ± 1 STD' not in labels:
                    red_fill_handle = mpatches.Patch(facecolor='red', alpha=0.3, edgecolor='red')
                    handles.append(red_fill_handle)
                    labels.append('Mean ± 1 STD')
                    
            elif mode == 'quantile':
                # Calculate mean and standard deviation for each time step
                particle_mean = particles_cpu[plot_start_time:plot_end_time, :, dim_idx].mean(dim=1)
                particle_std = particles_cpu[plot_start_time:plot_end_time, :, dim_idx].std(dim=1)
                
                # For 95% confidence interval assuming Gaussian distribution:
                # The z-score for 95% CI is approximately 1.96
                z_score = 1.96
                
                # Calculate upper and lower bounds of 95% CI
                upper_bound = particle_mean + z_score * particle_std
                lower_bound = particle_mean - z_score * particle_std
                
                ax.fill_between(time_steps_for_plot_np, 
                               lower_bound.numpy(), 
                               upper_bound.numpy(),
                               alpha=0.3, color='red', 
                               label='95% Confidence Interval')
                
                if '95% Confidence Interval' not in labels:
                    red_fill_handle = mpatches.Patch(facecolor='red', alpha=0.3, edgecolor='red')
                    handles.append(red_fill_handle)
                    labels.append('95% Confidence Interval')
                    
            elif mode in ['width', 'color']:
                # Original histogram-based visualization code
                global_max_mass = 1.0
                if mode == 'color':
                    all_bin_masses_list = []
                    for t_actual in range(plot_start_time, plot_end_time, hist_step):
                        data_t = particles_cpu[t_actual, :, dim_idx]
                        if data_t.numel() > 0:
                            hist_t, bins_t = torch.histogram(data_t, bins=15, density=True)
                            if hist_t.numel() > 0:
                                bin_widths_t = bins_t[1:] - bins_t[:-1]
                                current_bin_masses_t = hist_t * bin_widths_t
                                all_bin_masses_list.extend(current_bin_masses_t.tolist())

                    if all_bin_masses_list:
                        calculated_max_mass = max(all_bin_masses_list)
                        if calculated_max_mass > 1e-9:
                            global_max_mass = calculated_max_mass

                for t_actual in range(plot_start_time, plot_end_time, hist_step):
                    current_particles_dim_t = particles_cpu[t_actual, :, dim_idx]
                    if current_particles_dim_t.numel() == 0:
                        continue

                    hist, bins = torch.histogram(current_particles_dim_t, bins=15, density=True)
                    if hist.numel() == 0:
                        continue

                    bin_centers_torch = 0.5 * (bins[:-1] + bins[1:])
                    current_label_for_spread = 'Ensemble Spread' if not ensemble_spread_labeled else None

                    if mode == 'width':
                        hist_max_val = hist.max().item()
                        if hist_max_val > 1e-9:
                            hist_norm_torch = (hist / hist_max_val) * 0.8
                        else:
                            hist_norm_torch = torch.zeros_like(hist)

                        ax.fill_betweenx(bin_centers_torch.numpy(),
                                         (t_actual - hist_norm_torch).numpy(),
                                         (t_actual + hist_norm_torch).numpy(),
                                         facecolor='orange', edgecolor='none', alpha=0.5,
                                         label=current_label_for_spread)
                        if current_label_for_spread and 'Ensemble Spread' not in labels:
                            handles.append(plt.Rectangle((0, 0), 1, 1, fc='orange', alpha=0.5))
                            labels.append('Ensemble Spread')
                            ensemble_spread_labeled = True

                    elif mode == 'color':
                        cmap = plt.colormaps.get_cmap('Oranges')
                        bin_widths = bins[1:] - bins[:-1]
                        current_bin_masses = hist * bin_widths
                        norm_vals_torch = current_bin_masses / global_max_mass

                        for k in range(len(hist)):
                            segment_label = 'Ensemble Spread' if not ensemble_spread_labeled and k == 0 else None
                            norm_val_k = norm_vals_torch[k].item()
                            norm_val_k = max(0.0, min(1.0, norm_val_k))
                            color_val = cmap(0.2 + 0.6 * norm_val_k)
                            ax.fill_between([t_actual - step_width / 2, t_actual + step_width / 2],
                                            bins[k].item(), bins[k+1].item(),
                                            color=color_val, linewidth=0, alpha=0.7,
                                            label=segment_label)
                            if segment_label and 'Ensemble Spread' not in labels:
                                handles.append(plt.Rectangle((0, 0), 1, 1, fc=cmap(0.5), alpha=0.7))
                                labels.append('Ensemble Spread')
                                ensemble_spread_labeled = True

        if fontsize is not None:
            ax.set_xlabel('Time Step', fontsize=fontsize)
            ax.set_ylabel(f'Dimension {dim_idx} Values', fontsize=fontsize)
            ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        if fontsize is not None:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            ax.grid(True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

        if plot_end_time > plot_start_time :
            ax.set_xlim(plot_start_time - 0.5, plot_end_time - 0.5)
        fig.tight_layout()

        if save_fig:
            plt.savefig(f"{save_name}_dim_{dim_idx}_{mode}.png", dpi=150)
            if save_pdf:
                plt.savefig(f"{save_name}_dim_{dim_idx}_{mode}.pdf", bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

        for h, l in zip(handles, labels):
            if l not in [label[1] for label in all_labels]:
                all_handles.append(h)
                all_labels.append((len(all_labels), l)) # Keep track of order

    # Create custom legend with proxy artists
    if all_labels:
        all_labels.sort(key=lambda x: x[0]) # Sort by insertion order
        
        # Create proxy artists for the legend
        legend_handles = []
        legend_labels = []
        
        for label_tuple in all_labels:
            label = label_tuple[1]
            if observation_cpu is None and 'Observation' in label:
                continue
                
            legend_labels.append(label)
            
            # Create appropriate proxy artist based on label
            if label == 'True Trajectory':
                legend_handles.append(plt.Line2D([0], [0], color='blue', linewidth=1.5))
            elif label == 'Observation':
                legend_handles.append(plt.Line2D([0], [0], marker='*', color='green', markersize=8, linestyle=''))
            elif label == 'Ensemble Mean':
                legend_handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5))
            elif label == 'Mean ± 1 STD':
                legend_handles.append(mpatches.Patch(facecolor='red', alpha=0.3, edgecolor='red'))
            elif label == '95% Confidence Interval':
                legend_handles.append(mpatches.Patch(facecolor='red', alpha=0.3, edgecolor='red'))
            elif label == 'Ensemble Spread':
                if mode == 'width':
                    legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc='orange', alpha=0.5))
                elif mode == 'color':
                    cmap = plt.colormaps.get_cmap('Oranges')
                    legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=cmap(0.5), alpha=0.7))

        if legend_handles:
            fig_legend = plt.figure(figsize=(figsize[0], 0.5)) # Narrow height
            ax_legend = fig_legend.add_subplot(111)
            ax_legend.axis('off')
            legend = ax_legend.legend(legend_handles, legend_labels, loc='center', ncol=len(legend_handles), fontsize=24, frameon=False)
            fig_legend.tight_layout(pad=0.1) # Remove extra padding

            if save_fig:
                plt.savefig(f"{save_name}_legend.png", dpi=150, bbox_inches='tight')
                if save_pdf:
                    plt.savefig(f"{save_name}_legend.pdf", bbox_inches='tight')
            else:
                plt.show()
            plt.close(fig_legend)
        
        

def plot_particle_trajectories(
    particles: torch.Tensor,
    true_traj: torch.Tensor,
    observation: torch.Tensor,
    cmap_name: str = 'bwr',
    start_time: int = None,
    end_time: int = None,
    main_fig_size: tuple = (7, 3),
    save_fig: bool = False,
    save_pdf: bool = False,
    save_name: str = 'example_fig',
    colorbar_range: Optional[Tuple[float, float]] = None,
    colorbar_center: float = 0.0,
    plot_vertical_colorbar: bool = True,
    plot_horizontal_colorbar: bool = False
):
    """
    Visualizes particle trajectories, a true trajectory, an observation, their difference, and particle spread.

    Args:
        particles (torch.Tensor): A 3D tensor of shape (J, N, d)
                                  J: number of time steps
                                  N: number of particles
                                  d: dimension of particle state
        true_traj (torch.Tensor): A 2D tensor of shape (J, d)
                                  J: number of time steps
                                  d: dimension of state
        observation (torch.Tensor): A 2D tensor of shape (J, d_obs)
                                    J: number of time steps
                                    d_obs: dimension of observation
        cmap_name (str): Name of the Matplotlib colormap to use.
        start_time (int, optional): The starting time step index for plotting. Defaults to 0.
        end_time (int, optional): The ending time step index (exclusive) for plotting.
                                  Defaults to the total number of time steps.
        main_fig_size (tuple): Figure size (width, height) for the main plots.
                               Defaults to (7, 3).
        save_fig (bool): If True, saves the plots. Defaults to False.
        save_name (str): Base name for saved figures. Suffixes will be added.
                         Defaults to 'example_fig'.
        colorbar_range (Optional[Tuple[float, float]], optional): A tuple (min, max) for the
                                                                  colorbar range. If None, the range
                                                                  is determined automatically from data.
                                                                  Defaults to None.
        colorbar_center (float, optional): The data value that should correspond to the center color
                                           of the colormap (e.g., white for 'bwr'). Defaults to 0.0.
        plot_vertical_colorbar (bool, optional): If True, generates and saves a vertical colorbar.
                                                 Defaults to True.
        plot_horizontal_colorbar (bool, optional): If True, generates and saves a horizontal colorbar.
                                                   Defaults to False.
    """

    cpu_device = torch.device("cpu")
    particles = particles.detach().to(cpu_device)
    true_traj = true_traj.detach().to(cpu_device)
    observation = observation.detach().to(cpu_device)

    # 1. Validate input shapes
    if not isinstance(particles, torch.Tensor) or particles.ndim != 3:
        raise ValueError(f"Particles tensor must be 3-dimensional (J, N, d), but got ndim={particles.ndim}")
    J_p, N, d_p = particles.shape

    if not isinstance(true_traj, torch.Tensor) or true_traj.ndim != 2:
        raise ValueError(f"True trajectory tensor must be 2-dimensional (J, d), but got ndim={true_traj.ndim}")
    J_t, d_t = true_traj.shape

    if not isinstance(observation, torch.Tensor) or observation.ndim != 2:
        raise ValueError(f"Observation tensor must be 2-dimensional (J, d_obs), but got ndim={observation.ndim}")
    J_obs, d_obs = observation.shape

    if not (J_p == J_t == J_obs):
        raise ValueError(f"Time steps mismatch: particles have {J_p}, true_traj has {J_t}, and observation has {J_obs}")
    if d_p != d_t:
        raise ValueError(f"State dimensions mismatch for particles and true_traj: particles have {d_p} and true_traj has {d_t}")

    J_orig, d = J_p, d_p
    _start_time = 0 if start_time is None else int(start_time)
    _end_time = J_orig if end_time is None else int(end_time)

    if not (0 <= _start_time < J_orig and 0 < _end_time <= J_orig and _start_time < _end_time):
        raise ValueError(f"Invalid start_time ({_start_time}) or end_time ({_end_time}) for J_orig={J_orig}")

    particles_sliced = particles[_start_time:_end_time, :, :]
    true_traj_sliced = true_traj[_start_time:_end_time, :]
    observation_sliced = observation[_start_time:_end_time, :]
    
    current_J = true_traj_sliced.shape[0]
    if current_J == 0:
        print("Warning: Time slice is empty. No plots will be generated.")
        return

    # 2. Prepare data for plots
    mean_particles = torch.mean(particles_sliced, dim=1)
    abs_diff = torch.abs(mean_particles - true_traj_sliced)
    # abs_diff = mean_particles - true_traj_sliced
    particle_spread = torch.std(particles_sliced, dim=1) 

    plot_data_list = [
        mean_particles.T, true_traj_sliced.T, observation_sliced.T,
        abs_diff.T, particle_spread.T
    ]
    plot_suffixes = [
        "_mean_particles", "_true_trajectory", "_observation",
        "_absolute_difference", "_particle_spread"
    ]

    # 3. Determine plot value range (vmin_plot, vmax_plot)
    vmin_plot: float
    vmax_plot: float

    if colorbar_range is not None:
        vmin_plot, vmax_plot = float(colorbar_range[0]), float(colorbar_range[1])
        if vmin_plot > vmax_plot:
            raise ValueError(f"Invalid colorbar_range: min value {vmin_plot} cannot be greater than max value {vmax_plot}.")
    else:
        all_plot_data_values_list = [data.flatten() for data in plot_data_list if data.numel() > 0]
        if not all_plot_data_values_list:
            print("Warning: All plot data is empty. No plots will be generated.")
            return
        all_plot_data_values_torch = torch.cat(all_plot_data_values_list)
        if all_plot_data_values_torch.numel() == 0:
            print("Warning: Concatenated plot data is empty. No plots will be generated.")
            return
        all_plot_data_numpy = all_plot_data_values_torch.numpy()
        auto_min_val = np.nanmin(all_plot_data_numpy)
        auto_max_val = np.nanmax(all_plot_data_numpy)
        if np.isnan(auto_min_val) or np.isnan(auto_max_val):
            print("Warning: Min/max over plot data is NaN. Defaulting color range to [0,1].")
            vmin_plot, vmax_plot = 0.0, 1.0
        else:
            vmin_plot, vmax_plot = float(auto_min_val), float(auto_max_val)
    
    if vmin_plot == vmax_plot: 
        offset = 1e-6 if vmax_plot == 0 else abs(vmax_plot * 0.01)
        vmax_plot += offset
        vmin_plot -= offset
        if vmin_plot == vmax_plot: 
            vmin_plot = vmax_plot - 1.0

    # 4. Set up colormap and normalization for linear ticks and centered mid-color
    norm = mcolors.Normalize(vmin=vmin_plot, vmax=vmax_plot)
    base_cmap = plt.get_cmap(cmap_name)
    
    # p_norm_center is the normalized position of colorbar_center within [vmin_plot, vmax_plot]
    # This point 'p_norm_center' in the new (shifted) colormap should get the color from base_cmap(0.5)
    p_norm_center = (colorbar_center - vmin_plot) / (vmax_plot - vmin_plot) # This division is safe due to prior checks

    x_map_coords = np.linspace(0, 1, 256) 
    base_cmap_sample_points = np.zeros_like(x_map_coords)

    if np.isclose(p_norm_center, 0.5):
        base_cmap_sample_points = x_map_coords
    elif np.isclose(p_norm_center, 0):
        base_cmap_sample_points = 0.5 + 0.5 * x_map_coords
    elif np.isclose(p_norm_center, 1):
        base_cmap_sample_points = 0.5 * x_map_coords
    elif 0 < p_norm_center < 1:
        mask_first = x_map_coords <= p_norm_center
        base_cmap_sample_points[mask_first] = 0.5 * x_map_coords[mask_first] / p_norm_center
        mask_second = x_map_coords > p_norm_center
        base_cmap_sample_points[mask_second] = 0.5 + 0.5 * (x_map_coords[mask_second] - p_norm_center) / (1 - p_norm_center)
    elif p_norm_center < 0:
        base_cmap_sample_points = 0.5 + 0.5 * (x_map_coords - p_norm_center) / (1 - p_norm_center)
    elif p_norm_center > 1:
        base_cmap_sample_points = 0.5 * x_map_coords / p_norm_center

    base_cmap_sample_points = np.clip(base_cmap_sample_points, 0, 1)
    final_colors_for_cmap = base_cmap(base_cmap_sample_points)
    current_cmap = mcolors.ListedColormap(final_colors_for_cmap, name=base_cmap.name + "_shifted")

    # 5. Generate the main plots
    for i, data_to_plot_torch in enumerate(plot_data_list):
        fig, ax = plt.subplots(figsize=main_fig_size)
        if data_to_plot_torch.numel() > 0:
            ax.imshow(data_to_plot_torch.numpy(), aspect='auto', cmap=current_cmap, norm=norm, interpolation='nearest')
        ax.axis('off')
        if save_fig:
            plt.savefig(f"{save_name}{plot_suffixes[i]}.png", bbox_inches='tight', pad_inches=0)
            if save_pdf:
                plt.savefig(f"{save_name}{plot_suffixes[i]}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # 6. Generate and save colorbar(s) using the SAME scalar_mappable
    scalar_mappable = cm.ScalarMappable(cmap=current_cmap, norm=norm)
    scalar_mappable.set_array([]) 

    colorbar_label_size = 20
    colorbar_tick_size = 20
    if plot_vertical_colorbar:
        v_cbar_fig_height = main_fig_size[1] 
        v_cbar_fig_width = 0.3 
        fig_cbar_v, ax_cbar_v = plt.subplots(figsize=(v_cbar_fig_width, v_cbar_fig_height))
        cbar_v = plt.colorbar(scalar_mappable, cax=ax_cbar_v, orientation='horizontal')
        cbar_v.set_label('State Value / Obs / Error / Std', fontsize=colorbar_label_size)
        cbar_v.ax.tick_params(labelsize=colorbar_tick_size)
        if save_fig:
            plt.savefig(f"{save_name}_colorbar_vertical.png", bbox_inches='tight', pad_inches=0.05)
            if save_pdf:
                plt.savefig(f"{save_name}_colorbar_vertical.pdf", bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_cbar_v)

    if plot_horizontal_colorbar:
        h_cbar_fig_width = 2 * main_fig_size[0] 
        h_cbar_fig_height = 0.3 
        fig_cbar_h, ax_cbar_h = plt.subplots(figsize=(h_cbar_fig_width, h_cbar_fig_height))
        cbar_h = plt.colorbar(scalar_mappable, cax=ax_cbar_h, orientation='horizontal')
        cbar_h.set_label('State Value / Obs / Error / Std', fontsize=colorbar_label_size)
        cbar_h.ax.tick_params(labelsize=colorbar_tick_size)
        if save_fig:
            plt.savefig(f"{save_name}_colorbar_horizontal.png", bbox_inches='tight', pad_inches=0.05)
            if save_pdf:
                plt.savefig(f"{save_name}_colorbar_horizontal.pdf", bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_cbar_h)

    if not save_fig:
        if not plot_vertical_colorbar and not plot_horizontal_colorbar:
            print("Plot generation complete (no colorbars requested). If in a script, ensure plt.show() is called.")
        else:
            print("Plot generation complete including colorbar(s). If in a script, ensure plt.show() is called.")
            
# def plot_and_test_point_clouds(
#     args: Namespace,
#     tensor: torch.Tensor,
#     num_samples_plot: int,
#     num_samples_test: int,
#     prefix: str,
#     num_repeats: int = 10,
#     plot_indices: Optional[List[int]] = None,
#     history_traj: Optional[torch.Tensor] = None,
# ):
#     """
#     Plots point clouds and optional trajectories, and tests for Gaussianity.

#     Controlled by `args`, it can generate both adaptive and fixed-range plots.

#     Args:
#         args (Namespace): Configuration object with attributes like `dataset` and `dt`.
#         tensor (torch.Tensor): Point cloud data. Shape: (B, N, 3).
#         history_traj (Optional[torch.Tensor]): Historical trajectory data. Shape: (T, B, 3).
#         num_samples_plot (int): The number of points to sample for PLOTTING.
#         num_samples_test (int): The number of points to sample for STATISTICAL TESTS.
#         prefix (str): The filename prefix for the saved images.
#         num_repeats (int): The number of times to repeat the sampling and testing.
#         plot_indices (Optional[List[int]]): A list of specific batch indices to plot.
#                                             If None, all items in the batch are plotted.
#     """
#     # --- Initial Setup and Validation ---
#     if tensor.is_cuda:
#         tensor = tensor.cpu()
#     if history_traj is not None and history_traj.is_cuda:
#         history_traj = history_traj.cpu()

#     B, N, D = tensor.shape
#     assert D >= 3, f"Input tensor must have at least 3 dimensions, but found shape {tensor.shape}"
#     tensor = tensor[..., :3]
    
#     if history_traj is not None:
#         T_h, B_h, D_h = history_traj.shape
#         if B_h != B or D_h != D:
#             print(f"Warning: history_traj shape is incompatible. Trajectory will be ignored.")
#             history_traj = None

#     if plot_indices is None:
#         indices_to_process = range(B)
#     else:
#         indices_to_process = [idx for idx in plot_indices if 0 <= idx < B]

#     if num_samples_test > N:
#         num_samples_test = N

#     # --- Main loop over selected indices ---
#     for i in indices_to_process:
#         full_points_tensor = tensor[i, :, :]

#         # 1. --- STATISTICAL TESTING ---
#         hz_str = ""
#         if num_samples_test < D + 1:
#             hz_str = "HZ Test: Skipped (sample size too small)"
#         else:
#             p_values, normal_flags = [], []
#             for _ in range(num_repeats):
#                 test_indices = torch.randperm(N)[:num_samples_test]
#                 points_for_test = full_points_tensor[test_indices, :].numpy()
#                 hz_results = pg.multivariate_normality(points_for_test, alpha=0.05)
#                 p_values.append(hz_results.pval)
#                 normal_flags.append(hz_results.normal)
            
#             avg_pval = np.mean(p_values)
#             normal_percentage = np.mean(normal_flags) * 100
#             hz_str = f"HZ: {normal_percentage:.0f}% Normal ({num_repeats} runs, avg p-val={avg_pval:.3f})"

#         # 2. --- PLOTTING ---
#         n_to_plot = min(N, num_samples_plot)
#         plot_indices_for_vis = torch.randperm(N)[:n_to_plot]
#         points_to_plot = full_points_tensor[plot_indices_for_vis, :].numpy()
        
#         # --- Plot 1: Adaptive Axes (Always created) ---
#         adaptive_title = f"Cloud {i} (Test on {num_samples_test} points)\n{hz_str}"
#         fig_adaptive = plt.figure(figsize=(8, 8))
#         ax_adaptive = fig_adaptive.add_subplot(111, projection='3d')
#         ax_adaptive.scatter(points_to_plot[:, 0], points_to_plot[:, 1], points_to_plot[:, 2], s=5, alpha=0.7, label='filtering distribution')
#         ax_adaptive.set_xlabel("X-axis"); ax_adaptive.set_ylabel("Y-axis"); ax_adaptive.set_zlabel("Z-axis")
#         ax_adaptive.set_title(adaptive_title, fontsize=12)
#         ax_adaptive.legend()
#         fig_adaptive.savefig(f"{prefix}_{i}_adaptive.png", bbox_inches='tight', dpi=150)
#         plt.close(fig_adaptive)

#         # --- Plot 2: Fixed Axes (Conditional on dataset) ---
#         if args.dataset == 'lorenz63':
#             _limits = {'xlim': (-25, 25), 'ylim': (-35, 35), 'zlim': (0, 60)}
#         elif args.dataset == 'rossler':
#             _limits = {'xlim': (-15, 15), 'ylim': (-15, 15), 'zlim': (0, 30)}
#         elif args.dataset == 'lorenz96':
#             _limits = {'xlim': (-13, 13), 'ylim': (-13, 13), 'zlim': (-13, 13)}
#         else:
#             _limits = None
            
#         if _limits is not None:
#             fixed_title = rf"{args.dataset}: $\Delta t$ = {args.dt}, time step = {len(history_traj)}, T = {round(args.dt*len(history_traj)*100)/100:.2f}" 
            
#             fig_fixed = plt.figure(figsize=(8, 8))
#             ax_fixed = fig_fixed.add_subplot(111, projection='3d')
#             ax_fixed.scatter(points_to_plot[:, 0], points_to_plot[:, 1], points_to_plot[:, 2], s=5, alpha=0.7, label='filtering distribution')
            
#             if history_traj is not None:
#                 traj_to_plot = history_traj[:, i, :].numpy()
#                 ax_fixed.plot(traj_to_plot[:, 0], traj_to_plot[:, 1], traj_to_plot[:, 2], color='red', linewidth=1.5, label='History')

#             ax_fixed.set_xlabel("X-axis"); ax_fixed.set_ylabel("Y-axis"); ax_fixed.set_zlabel("Z-axis")
#             ax_fixed.set_title(fixed_title, fontsize=12)
#             ax_fixed.set_xlim(_limits['xlim']); ax_fixed.set_ylim(_limits['ylim']); ax_fixed.set_zlim(_limits['zlim'])
#             ax_fixed.legend()
#             fig_fixed.savefig(f"{prefix}_{i}_fixed.png", bbox_inches='tight', dpi=150)
#             plt.close(fig_fixed)
        

#     # --- Final Print Statement ---
#     num_processed = len(indices_to_process) if isinstance(indices_to_process, list) else B
#     plot_types_str = "2 plots (adaptive/fixed)" if _limits is not None else "1 plot (adaptive)"
#     print(f"Processed {num_processed} point clouds, saving {plot_types_str} for each with prefix '{prefix}'.")

def _save_separate_legend(prefix: str, include_obs: bool = False, dpi: int = 200) -> None:
    """Create and save a standalone legend image with adaptive layout/figsize.

    Args:
        prefix (str): File prefix for the saved legend image "<prefix>_legend.png".
        include_obs (bool): If True, include the Observation legend entry and layout as 2 rows.
                            If False, omit Observation and layout as a single row.
        dpi (int): Dots per inch for the saved PNG.
    """
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    # Proxy artists for legend entries
    handle_prior = Line2D([0], [0], marker='o', linestyle='None',
                          markerfacecolor='blue', markeredgecolor='none', markersize=8,
                          label='Predictive (prior) distribution')
    handle_posterior = Line2D([0], [0], marker='o', linestyle='None',
                              markerfacecolor='red', markeredgecolor='white', markersize=8,
                              label='Filtering (posterior) distribution')
    handle_obs = Line2D([0], [0], marker='*', linestyle='None',
                        markerfacecolor='orange', markeredgecolor='black', markersize=14,
                        label='Observation')
    handle_traj = Line2D([0], [0], linestyle='-', color='black', linewidth=2,
                         label='History trajectory')

    # Assemble handles/labels based on whether observation is included
    if include_obs:
        handles = [handle_prior, handle_posterior, handle_obs, handle_traj]
        ncols = 2
        nrows = 2  # 4 items → 2x2 grid for good balance and readability
    else:
        handles = [handle_prior, handle_posterior, handle_traj]
        ncols = 3
        nrows = 1  # 3 items → single row

    labels = [h.get_label() for h in handles]

    # --- Adaptive figsize heuristic ---
    # Width scales with column count; height scales with row count.
    # Use slightly larger per-column width to accommodate longer labels.
    per_col_w = 3.2   # inches per column
    per_row_h = 1.35  # inches per row
    pad_w = 0.3
    pad_h = 0.2
    fig_w = max(4.5, ncols * per_col_w + pad_w)
    fig_h = max(1.2, nrows * per_row_h + pad_h)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Centered legend; frame on for standalone readability
    ax.legend(handles, labels, ncol=ncols, loc='center', frameon=True)

    fig.savefig(f"{prefix}_legend.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def plot_and_test_point_clouds(
    args: Namespace,
    tensor: torch.Tensor,
    num_samples_plot: int,
    num_samples_test: int,
    prefix: str,
    point_color: str,                          # NEW: "red" or "blue"
    observation: Optional[torch.Tensor] = None,# NEW: 3D vector for the orange star
    num_repeats: int = 10,
    plot_indices: Optional[List[int]] = None,
    history_traj: Optional[torch.Tensor] = None,
):
    """
    Plots point clouds and optional trajectories, tests for Gaussianity (HZ),
    and saves a separate legend image.

    Args:
        args (Namespace): Has fields like `dataset` and `dt`.
        tensor (torch.Tensor): (B, N, 3+) point cloud (only first 3 dims used).
        num_samples_plot (int): Number of points to sample for PLOTTING.
        num_samples_test (int): Number of points to sample for HZ TEST.
        prefix (str): Base filename prefix for saved images.
        point_color (str): Color for the CURRENT cloud: "red" or "blue".
        observation (Optional[torch.Tensor]): 3D vector for orange star marker.
        num_repeats (int): Repeated HZ tests for stability.
        plot_indices (Optional[List[int]]): Which batch items to plot; default all.
        history_traj (Optional[torch.Tensor]): (T, B, 3+) historical trajectory.

    Behavior changes:
        - Trajectory is plotted in BLACK (instead of red).
        - Current cloud is plotted entirely in `point_color` ("red" or "blue").
        - Observation, if provided, is plotted as an ORANGE star.
        - No in-figure legend; a standalone 2-row legend image is saved as "<prefix>_legend.png".
    """
    # --- Validate and move to CPU if needed ---
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if history_traj is not None and history_traj.is_cuda:
        history_traj = history_traj.cpu()
    if observation is not None and observation.is_cuda:
        observation = observation.cpu()

    B, N, D = tensor.shape
    assert D >= 3, f"Input tensor must have at least 3 dims, got {tensor.shape}"
    tensor = tensor[..., :3]

    if history_traj is not None:
        T_h, B_h, D_h = history_traj.shape
        if B_h != B or D_h < 3:
            print("Warning: history_traj shape is incompatible. Trajectory will be ignored.")
            history_traj = None

    # Sanitize `point_color`
    point_color = (point_color or "").lower().strip()
    if point_color not in {"red", "blue"}:
        print("Warning: point_color should be 'red' or 'blue'. Default to 'blue'.")
        point_color = "blue"

    # Indices to process
    if plot_indices is None:
        indices_to_process = list(range(B))
    else:
        indices_to_process = [idx for idx in plot_indices if 0 <= idx < B]

    if num_samples_test > N:
        num_samples_test = N

    # --- Iterate over selected clouds ---
    for i in indices_to_process:
        full_points_tensor = tensor[i, :, :]

        # 1) --- HZ TEST (multivariate normality) ---
        hz_str = ""
        if num_samples_test < 3 + 1:  # D is 3 after slicing
            hz_str = "HZ Test: Skipped (sample size too small)"
        else:
            p_values, normal_flags = [], []
            for _ in range(num_repeats):
                test_indices = torch.randperm(N)[:num_samples_test]
                points_for_test = full_points_tensor[test_indices, :].numpy()
                hz_results = pg.multivariate_normality(points_for_test, alpha=0.05)
                p_values.append(hz_results.pval)
                normal_flags.append(bool(hz_results.normal))
            avg_pval = float(np.mean(p_values))
            normal_percentage = float(np.mean(normal_flags) * 100.0)
            hz_str = f"HZ: {normal_percentage:.0f}% Normal ({num_repeats} runs, avg p-val={avg_pval:.3f})"

        # 2) --- SAMPLE FOR PLOTTING ---
        n_to_plot = min(N, num_samples_plot)
        plot_indices_for_vis = torch.randperm(N)[:n_to_plot]
        points_to_plot = full_points_tensor[plot_indices_for_vis, :].numpy()

        # 3) --- PLOT 1: Adaptive axes (always) ---
        adaptive_title = f"Cloud {i} (Test on {num_samples_test} points)\n{hz_str}"
        fig_adaptive = plt.figure(figsize=(8, 8))
        ax_adaptive = fig_adaptive.add_subplot(111, projection='3d')

        # Current cloud (color-controlled)
        ax_adaptive.scatter(points_to_plot[:, 0], points_to_plot[:, 1], points_to_plot[:, 2],
                            s=5, alpha=0.7, c=point_color, edgecolors='none', zorder=0)

        # History trajectory in black (if available)
        # if history_traj is not None:
        #     traj_to_plot = history_traj[:, i, :3].numpy()
        #     ax_adaptive.plot(traj_to_plot[:, 0], traj_to_plot[:, 1], traj_to_plot[:, 2],
        #                      color='black', linewidth=1.5, zorder=10)

        # Observation (orange star), if provided
        if observation is not None:
            obs_np = observation[:3].numpy() if isinstance(observation, torch.Tensor) else np.asarray(observation)[:3]
            ax_adaptive.scatter(obs_np[0], obs_np[1], obs_np[2],
                                marker='*', s=140, c='orange', edgecolors='black', linewidth=0.6, zorder=10)

        ax_adaptive.set_xlabel("X-axis"); ax_adaptive.set_ylabel("Y-axis"); ax_adaptive.set_zlabel("Z-axis")
        ax_adaptive.set_title(adaptive_title, fontsize=12)
        fig_adaptive.savefig(f"{prefix}_{i}_adaptive.png", bbox_inches='tight', dpi=150)
        plt.close(fig_adaptive)

        # 4) --- PLOT 2: Fixed axes (dataset-conditional) ---
        if args.dataset == 'lorenz63':
            _limits = {'xlim': (-25, 25), 'ylim': (-35, 35), 'zlim': (0, 60)}
        elif args.dataset == 'rossler':
            _limits = {'xlim': (-15, 15), 'ylim': (-15, 15), 'zlim': (0, 30)}
        elif args.dataset == 'lorenz96':
            _limits = {'xlim': (-13, 13), 'ylim': (-13, 13), 'zlim': (-13, 13)}
        else:
            _limits = None

        if _limits is not None:
            # Guard against None history_traj in title
            steps = len(history_traj) if history_traj is not None else 0
            total_T = round(float(getattr(args, "dt", 1.0)) * steps * 100) / 100.0
            fixed_title = rf"{args.dataset}: $\Delta t$ = {getattr(args, 'dt', 1.0)}, time step = {steps}, T = {total_T:.2f}"

            fig_fixed = plt.figure(figsize=(8, 8))
            ax_fixed = fig_fixed.add_subplot(111, projection='3d')

            # Current cloud (color-controlled)
            ax_fixed.scatter(points_to_plot[:, 0], points_to_plot[:, 1], points_to_plot[:, 2],
                             s=5, alpha=0.7, c=point_color, edgecolors='none', zorder=0)

            # Trajectory in black (if available)
            if history_traj is not None:
                traj_to_plot = history_traj[:, i, :3].numpy()
                ax_fixed.plot(traj_to_plot[:, 0], traj_to_plot[:, 1], traj_to_plot[:, 2],
                              color='black', linewidth=1.5, zorder=10)

            # Observation (orange star), if provided
            if observation is not None:
                obs_np = observation[:3].numpy() if isinstance(observation, torch.Tensor) else np.asarray(observation)[:3]
                ax_fixed.scatter(obs_np[0], obs_np[1], obs_np[2],
                                 marker='*', s=140, c='orange', edgecolors='black', linewidth=0.6, zorder=10)

            ax_fixed.set_xlabel("X-axis"); ax_fixed.set_ylabel("Y-axis"); ax_fixed.set_zlabel("Z-axis")
            ax_fixed.set_title(fixed_title, fontsize=12)
            ax_fixed.set_xlim(_limits['xlim']); ax_fixed.set_ylim(_limits['ylim']); ax_fixed.set_zlim(_limits['zlim'])
            fig_fixed.savefig(f"{prefix}_{i}_fixed.png", bbox_inches='tight', dpi=150)
            plt.close(fig_fixed)

    # 5) --- Save a separate legend image (2 rows) ---
    legend_dir = os.path.dirname(prefix) or "."
    legend_prefix = os.path.join(legend_dir, str(getattr(args, "dataset", "")))
    if observation is not None:
        _save_separate_legend(legend_prefix, include_obs=True)
    else:
        _save_separate_legend(legend_prefix, include_obs=False)

    # 6) --- Final summary print ---
    num_processed = len(indices_to_process)
    plot_types_str = "2 plots (adaptive/fixed)" if _limits is not None else "1 plot (adaptive)"
    print(f"Processed {num_processed} point clouds, saving {plot_types_str} for each with prefix '{prefix}'.")


def _map_state_to_ring_xy(points: torch.Tensor) -> torch.Tensor:
    """
    Map state points (N, D) to unit-circle coordinates (N, 2).

    - D == 1: theta = 2*pi*x
    - D >= 2: theta = atan2(y, x), where x=points[:,0], y=points[:,1]
    """
    if points.ndim != 2:
        raise ValueError(f"Expected points shape (N, D), got {tuple(points.shape)}")
    d = points.shape[1]
    if d < 1:
        raise ValueError("State dimension must be >= 1 for ring mapping.")

    if d == 1:
        theta = 2.0 * np.pi * points[:, 0]
    else:
        theta = torch.atan2(points[:, 1], points[:, 0])

    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def plot_and_test_point_clouds_ring(
    args: Namespace,
    tensor: torch.Tensor,
    num_samples_plot: int,
    prefix: str,
    point_color: str,
    observation: Optional[torch.Tensor] = None,
    true_state: Optional[torch.Tensor] = None,
    plot_history: bool = True,
    plot_indices: Optional[List[int]] = None,
    history_traj: Optional[torch.Tensor] = None,
):
    """
    Visualize 1D/2D point clouds on a unit circle using 2D density estimation.

    - For 1D: state x is mapped by theta=2*pi*x.
    - For 2D: state (u,v) is mapped by theta=atan2(v,u).
    - Observation is interpreted as circle x-coordinate and plotted at (obs_x, 0).
    - True state is mapped to ring and plotted with a black "x" marker if provided.
    - History trajectory is optional and controlled by `plot_history`.
    - If plotted ensemble size < 1000, draw all points directly (scatter).
      Otherwise use 2D hexbin density.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if history_traj is not None and history_traj.is_cuda:
        history_traj = history_traj.cpu()
    if observation is not None and isinstance(observation, torch.Tensor) and observation.is_cuda:
        observation = observation.cpu()
    if true_state is not None and isinstance(true_state, torch.Tensor) and true_state.is_cuda:
        true_state = true_state.cpu()

    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor shape (B, N, D), got {tuple(tensor.shape)}")

    B, N, D = tensor.shape
    if D < 1:
        raise ValueError(f"State dimension must be >=1, got D={D}.")

    if history_traj is not None:
        if history_traj.ndim != 3:
            print("Warning: history_traj must be (T, B, D). Ignoring trajectory.")
            history_traj = None
        else:
            T_h, B_h, D_h = history_traj.shape
            if B_h != B or D_h < 1:
                print("Warning: history_traj shape is incompatible. Ignoring trajectory.")
                history_traj = None

    point_color = (point_color or "").lower().strip()
    if point_color not in {"red", "blue"}:
        point_color = "blue"

    if plot_indices is None:
        indices_to_process = list(range(B))
    else:
        indices_to_process = [idx for idx in plot_indices if 0 <= idx < B]

    # Observation scalar -> x-coordinate on ring axis
    obs_x = None
    if observation is not None:
        if isinstance(observation, torch.Tensor):
            obs_tensor = observation.reshape(-1).detach().cpu()
            if obs_tensor.numel() > 0:
                obs_x = float(obs_tensor[0].item())
        else:
            obs_np = np.asarray(observation).reshape(-1)
            if obs_np.size > 0:
                obs_x = float(obs_np[0])

    for i in indices_to_process:
        full_points = tensor[i, :, :]  # (N, D)
        n_to_plot = min(N, num_samples_plot)
        density_threshold = 1000
        if n_to_plot < density_threshold:
            points_plot = full_points
        else:
            chosen = torch.randperm(N)[:n_to_plot]
            points_plot = full_points[chosen, :]
        xy_plot = _map_state_to_ring_xy(points_plot).numpy()

        fig, ax = plt.subplots(figsize=(7, 7))

        # Unit circle reference
        phi = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(np.cos(phi), np.sin(phi), color='gray', linewidth=1.0, alpha=0.8, label='Unit circle')

        if xy_plot.shape[0] < density_threshold:
            # Small ensembles: plot all points directly.
            ax.scatter(
                xy_plot[:, 0], xy_plot[:, 1],
                s=10, alpha=0.55, c=point_color, edgecolors='none', label='Ensemble'
            )
        else:
            # Large ensembles: density view is clearer.
            cmap = 'Reds' if point_color == 'red' else 'Blues'
            hb = ax.hexbin(
                xy_plot[:, 0], xy_plot[:, 1],
                gridsize=60,
                extent=(-1.25, 1.25, -1.25, 1.25),
                mincnt=1,
                bins='log',
                cmap=cmap,
                linewidths=0.0,
                alpha=0.95,
            )
            cbar = fig.colorbar(hb, ax=ax, shrink=0.82, pad=0.02)
            cbar.set_label("log10(count)")

            # Optional sparse overlay to show geometry without overplotting.
            n_overlay = min(1500, xy_plot.shape[0])
            if n_overlay > 0:
                idx_overlay = np.random.choice(xy_plot.shape[0], size=n_overlay, replace=False)
                ax.scatter(
                    xy_plot[idx_overlay, 0], xy_plot[idx_overlay, 1],
                    s=2, alpha=0.12, c=point_color, edgecolors='none'
                )

        # History trajectory (mapped to ring)
        if plot_history and history_traj is not None:
            traj = history_traj[:, i, :]  # (T, D)
            traj_xy = _map_state_to_ring_xy(traj).numpy()
            ax.plot(traj_xy[:, 0], traj_xy[:, 1], color='black', linewidth=1.5, label='History trajectory')

        # Observation as circle x-coordinate
        if obs_x is not None:
            obs_x_clip = float(np.clip(obs_x, -1.2, 1.2))
            ax.scatter(obs_x_clip, 0.0, marker='*', s=160, c='orange', edgecolors='black',
                       linewidth=0.6, zorder=10, label='Observation x-coordinate')
            ax.axvline(obs_x_clip, color='orange', linestyle='--', linewidth=1.0, alpha=0.7)

        # True state mapped to ring (if provided)
        if true_state is not None:
            if isinstance(true_state, torch.Tensor):
                ts = true_state.detach().cpu()
            else:
                ts = torch.as_tensor(true_state)

            true_xy = None
            if ts.ndim == 1:
                if ts.numel() >= D:
                    true_xy = _map_state_to_ring_xy(ts[:D].reshape(1, D)).numpy()[0]
            elif ts.ndim == 2:
                if ts.shape[0] == B and ts.shape[1] >= D:
                    true_xy = _map_state_to_ring_xy(ts[i, :D].reshape(1, D)).numpy()[0]
                elif ts.shape[1] >= D:
                    true_xy = _map_state_to_ring_xy(ts[0, :D].reshape(1, D)).numpy()[0]

            if true_xy is not None:
                ax.scatter(
                    true_xy[0], true_xy[1],
                    marker='x', s=90, c='black', linewidth=1.8, zorder=11, label='True state'
                )

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_xlabel("ring-x")
        ax.set_ylabel("ring-y")
        mode_str = "scatter" if xy_plot.shape[0] < density_threshold else "hexbin density"
        ax.set_title(f"{args.dataset} (ring map, {mode_str}), cloud {i}")
        ax.legend(loc='upper right', fontsize=9, frameon=True)

        fig.savefig(f"{prefix}_{i}_ring.png", bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f"Processed {len(indices_to_process)} ring-mapped point clouds with prefix '{prefix}'.")
