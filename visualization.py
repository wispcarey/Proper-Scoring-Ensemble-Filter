import os
import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


from argparse import Namespace 


PF_3D_SUPPORTED_DATASETS = {"lorenz63", "lorenz96", "ks"}
PF_MAX_SCATTER_POINTS = 1000000
PF_MAX_CONTOUR_POINTS = 1000000
PF_FIXED_RANGES_3D = {
    # Typical Lorenz-63 attractor envelope is about x in [-20, 20], y in [-27, 27], z in [0, 48.5].
    # We keep a slightly larger plotting box for robustness.
    "lorenz63": {"xlim": (-22.0, 22.0), "ylim": (-30.0, 30.0), "zlim": (0.0, 52.0)},
    "lorenz96": {"xlim": (-15.0, 15.0), "ylim": (-15.0, 15.0), "zlim": (-15.0, 15.0)},
    "ks": {"xlim": (-6.0, 6.0), "ylim": (-6.0, 6.0), "zlim": (-6.0, 6.0)},
}


def _save_horizontal_legend_image(
    prefix: str,
    handles: List,
    labels: List[str],
    save_pdf: bool = False,
    dpi: int = 200,
    fontsize: int = 11,
    frameon: bool = True,
    scale: float = 1.45,
    pad_inches: float = 0.02,
) -> None:
    """Save a standalone horizontal legend image "<prefix>_legend.(png|pdf)"."""
    dedup_handles: List = []
    dedup_labels: List[str] = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        dedup_handles.append(handle)
        dedup_labels.append(label)
        seen.add(label)

    if len(dedup_handles) == 0:
        return

    scale = float(max(scale, 0.5))
    ncols = len(dedup_handles)
    fig_w = max(4.5 * scale, 2.2 * ncols * scale)
    fig_h = 1.35 * scale
    scaled_fontsize = max(1, int(round(float(fontsize) * scale)))
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    ax.axis('off')
    legend = ax.legend(
        dedup_handles,
        dedup_labels,
        loc='center',
        ncol=ncols,
        fontsize=scaled_fontsize,
        markerscale=scale,
        handlelength=2.0 * scale,
        handleheight=0.8 * scale,
        borderpad=0.35 * scale,
        columnspacing=1.2 * scale,
        handletextpad=0.7 * scale,
        labelspacing=0.4 * scale,
        frameon=frameon,
    )
    legend_handle_list = getattr(legend, "legend_handles", None)
    if legend_handle_list is None:
        legend_handle_list = getattr(legend, "legendHandles", [])
    for handle in legend_handle_list:
        if isinstance(handle, Line2D):
            handle.set_linewidth(max(1.0, handle.get_linewidth() * scale))
            if handle.get_markersize() > 0:
                handle.set_markersize(handle.get_markersize() * scale)

    fig.tight_layout(pad=0.02)
    fig.savefig(f"{prefix}_legend.png", dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
    if save_pdf:
        fig.savefig(f"{prefix}_legend.pdf", bbox_inches='tight', pad_inches=pad_inches)
    plt.close(fig)


def _save_two_row_legend_image(
    prefix: str,
    top_handles: List,
    top_labels: List[str],
    bottom_handles: List,
    bottom_labels: List[str],
    save_pdf: bool = False,
    dpi: int = 200,
    fontsize: int = 11,
    frameon: bool = True,
    scale: float = 1.45,
    pad_inches: float = 0.02,
) -> None:
    """Save one standalone legend image with two rows in a single legend frame."""
    def _dedup(handles: List, labels: List[str]) -> Tuple[List, List[str]]:
        out_handles: List = []
        out_labels: List[str] = []
        seen = set()
        for handle, label in zip(handles, labels):
            if not label or label in seen:
                continue
            out_handles.append(handle)
            out_labels.append(label)
            seen.add(label)
        return out_handles, out_labels

    top_h, top_l = _dedup(top_handles, top_labels)
    bot_h, bot_l = _dedup(bottom_handles, bottom_labels)
    if len(top_h) == 0 and len(bot_h) == 0:
        return
    if len(top_h) == 0 or len(bot_h) == 0:
        _save_horizontal_legend_image(
            prefix=prefix,
            handles=top_h if len(top_h) > 0 else bot_h,
            labels=top_l if len(top_l) > 0 else bot_l,
            save_pdf=save_pdf,
            dpi=dpi,
            fontsize=fontsize,
            frameon=frameon,
            scale=scale,
            pad_inches=pad_inches,
        )
        return

    scale = float(max(scale, 0.5))
    ncols = max(len(top_h), len(bot_h))
    fig_w = max(4.5 * scale, 2.2 * ncols * scale)
    fig_h = 1.9 * scale
    scaled_fontsize = max(1, int(round(float(fontsize) * scale)))

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    ax.axis('off')

    blank_handle = Line2D([0], [0], linestyle='None', marker='None', alpha=0.0)
    row_top_handles = [top_h[j] if j < len(top_h) else blank_handle for j in range(ncols)]
    row_top_labels = [top_l[j] if j < len(top_l) else " " for j in range(ncols)]
    row_bot_handles = [bot_h[j] if j < len(bot_h) else blank_handle for j in range(ncols)]
    row_bot_labels = [bot_l[j] if j < len(bot_l) else " " for j in range(ncols)]

    # Matplotlib legend with ncol fills entries in column-major order.
    # Interleave row-1/row-2 items per column so display reads left-to-right by rows.
    handles: List = []
    labels: List[str] = []
    for col in range(ncols):
        handles.append(row_top_handles[col])
        labels.append(row_top_labels[col])
        handles.append(row_bot_handles[col])
        labels.append(row_bot_labels[col])

    legend = ax.legend(
        handles,
        labels,
        loc='center',
        ncol=ncols,
        fontsize=scaled_fontsize,
        markerscale=scale,
        handlelength=2.0 * scale,
        handleheight=0.8 * scale,
        borderpad=0.35 * scale,
        columnspacing=1.2 * scale,
        handletextpad=0.7 * scale,
        labelspacing=0.55 * scale,
        frameon=frameon,
    )

    legend_handle_list = getattr(legend, "legend_handles", None)
    if legend_handle_list is None:
        legend_handle_list = getattr(legend, "legendHandles", [])
    for handle in legend_handle_list:
        if isinstance(handle, Line2D):
            handle.set_linewidth(max(1.0, handle.get_linewidth() * scale))
            if handle.get_markersize() > 0:
                handle.set_markersize(handle.get_markersize() * scale)

    fig.tight_layout(pad=0.02)
    fig.savefig(f"{prefix}_legend.png", dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
    if save_pdf:
        fig.savefig(f"{prefix}_legend.pdf", bbox_inches='tight', pad_inches=pad_inches)
    plt.close(fig)


def _covariance_ellipse_2d(
    cov_2d: np.ndarray,
    center_2d: np.ndarray,
    n_std: float = 2.447746830680816,
    **kwargs,
):
    """
    Build a matplotlib ellipse patch from a 2x2 covariance matrix.

    n_std=2.4477 corresponds to a 95% confidence region in 2D.
    """
    cov = np.asarray(cov_2d, dtype=np.float64)
    center = np.asarray(center_2d, dtype=np.float64)
    if cov.shape != (2, 2):
        raise ValueError(f"cov_2d must be (2,2), got {cov.shape}.")
    if center.shape != (2,):
        raise ValueError(f"center_2d must be (2,), got {center.shape}.")

    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], a_min=0.0, a_max=None)
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2.0 * n_std * np.sqrt(eigvals[0])
    height = 2.0 * n_std * np.sqrt(eigvals[1])
    return mpatches.Ellipse(
        xy=center,
        width=float(width),
        height=float(height),
        angle=float(angle),
        **kwargs,
    )


def plot_linear_kalman_vs_method_2d(
    kalman_means: torch.Tensor,
    kalman_covs: torch.Tensor,
    method_means: torch.Tensor,
    method_covs: torch.Tensor,
    dim_indices: Tuple[int, int] = (0, 1),
    tail_steps: int = 20,
    save_fig: bool = False,
    save_pdf: bool = False,
    save_name: str = "example_fig_linear_2d",
    legend_in_figure: bool = True,
):
    """
    Plot 2D projected trajectories with 95% covariance ellipses:
    Kalman posterior (as reference truth) vs. method posterior.
    """
    if kalman_means.ndim != 2 or method_means.ndim != 2:
        raise ValueError("kalman_means and method_means must be 2D tensors [T, D].")
    if kalman_covs.ndim != 3 or method_covs.ndim != 3:
        raise ValueError("kalman_covs and method_covs must be 3D tensors [T, D, D].")
    if kalman_means.shape != method_means.shape:
        raise ValueError("kalman_means and method_means must have the same shape.")
    if kalman_covs.shape != method_covs.shape:
        raise ValueError("kalman_covs and method_covs must have the same shape.")
    if kalman_means.shape[0] != kalman_covs.shape[0]:
        raise ValueError("Time length mismatch between means and covariances.")

    total_steps, state_dim = kalman_means.shape
    if total_steps <= 0:
        print("Warning: empty trajectory, skipping 2D linear visualization.")
        return

    dim0, dim1 = int(dim_indices[0]), int(dim_indices[1])
    if not (0 <= dim0 < state_dim and 0 <= dim1 < state_dim):
        raise ValueError(
            f"dim_indices={dim_indices} out of range for state_dim={state_dim}."
        )
    if dim0 == dim1:
        raise ValueError("dim_indices must contain two distinct dimensions.")

    tail_steps = max(1, int(tail_steps))
    start = max(0, total_steps - tail_steps)
    end = total_steps

    km = kalman_means[start:end, [dim0, dim1]].detach().cpu().numpy()
    mm = method_means[start:end, [dim0, dim1]].detach().cpu().numpy()
    kc_full = kalman_covs[start:end].detach().cpu().numpy()
    mc_full = method_covs[start:end].detach().cpu().numpy()

    idx = np.array([dim0, dim1], dtype=np.int64)
    kc = kc_full[:, idx[:, None], idx[None, :]]
    mc = mc_full[:, idx[:, None], idx[None, :]]

    valid_k = np.isfinite(km).all(axis=1) & np.isfinite(kc.reshape(kc.shape[0], -1)).all(axis=1)
    valid_m = np.isfinite(mm).all(axis=1) & np.isfinite(mc.reshape(mc.shape[0], -1)).all(axis=1)
    if (not np.any(valid_k)) and (not np.any(valid_m)):
        print("Warning: all projected values are non-finite, skipping 2D linear visualization.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    kalman_color = "tab:blue"
    method_color = "tab:red"
    n_std_95 = 2.447746830680816

    km_plot = km.copy()
    km_plot[~valid_k] = np.nan
    mm_plot = mm.copy()
    mm_plot[~valid_m] = np.nan

    ax.plot(
        km_plot[:, 0],
        km_plot[:, 1],
        color=kalman_color,
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        alpha=0.95,
        label="Kalman mean",
    )
    ax.plot(
        mm_plot[:, 0],
        mm_plot[:, 1],
        color=method_color,
        linewidth=1.6,
        marker="s",
        markersize=3.0,
        alpha=0.9,
        label="Method mean",
    )

    for t in range(km.shape[0]):
        if valid_k[t]:
            ell_k = _covariance_ellipse_2d(
                kc[t],
                km[t],
                n_std=n_std_95,
                edgecolor=kalman_color,
                facecolor=kalman_color,
                alpha=0.12,
                linewidth=1.0,
            )
            ax.add_patch(ell_k)
        if valid_m[t]:
            ell_m = _covariance_ellipse_2d(
                mc[t],
                mm[t],
                n_std=n_std_95,
                edgecolor=method_color,
                facecolor=method_color,
                alpha=0.10,
                linewidth=1.0,
            )
            ax.add_patch(ell_m)

    x_all = np.concatenate([km[:, 0], mm[:, 0]])
    y_all = np.concatenate([km[:, 1], mm[:, 1]])
    k_var_x = np.clip(kc[:, 0, 0], a_min=0.0, a_max=None)
    k_var_y = np.clip(kc[:, 1, 1], a_min=0.0, a_max=None)
    m_var_x = np.clip(mc[:, 0, 0], a_min=0.0, a_max=None)
    m_var_y = np.clip(mc[:, 1, 1], a_min=0.0, a_max=None)
    x_rad = n_std_95 * np.sqrt(np.concatenate([k_var_x, m_var_x]))
    y_rad = n_std_95 * np.sqrt(np.concatenate([k_var_y, m_var_y]))
    finite_x = np.isfinite(x_all) & np.isfinite(x_rad)
    finite_y = np.isfinite(y_all) & np.isfinite(y_rad)
    if np.any(finite_x):
        x_min = np.min(x_all[finite_x] - x_rad[finite_x])
        x_max = np.max(x_all[finite_x] + x_rad[finite_x])
    else:
        x_min, x_max = -1.0, 1.0
    if np.any(finite_y):
        y_min = np.min(y_all[finite_y] - y_rad[finite_y])
        y_max = np.max(y_all[finite_y] + y_rad[finite_y])
    else:
        y_min, y_max = -1.0, 1.0

    x_pad = max(1e-6, 0.08 * (x_max - x_min))
    y_pad = max(1e-6, 0.08 * (y_max - y_min))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.55)
    ax.set_xlabel(f"dim{dim0}")
    ax.set_ylabel(f"dim{dim1}")
    ax.set_title(
        f"Linear test 2D projection (steps {start + 1}-{end}, 95% covariance ellipses)"
    )

    if legend_in_figure:
        ax.legend(loc='best', frameon=True, fontsize=9)

    fig.tight_layout()

    if save_fig:
        fig.savefig(f"{save_name}.png", dpi=150, bbox_inches='tight')
        if save_pdf:
            fig.savefig(f"{save_name}.pdf", bbox_inches='tight')
    else:
        plt.show()

    if not legend_in_figure:
        handles = [
            Line2D([0], [0], color=kalman_color, linewidth=1.8, marker="o", markersize=5),
            Line2D([0], [0], color=method_color, linewidth=1.6, marker="s", markersize=5),
            mpatches.Patch(facecolor=kalman_color, edgecolor=kalman_color, alpha=0.12),
            mpatches.Patch(facecolor=method_color, edgecolor=method_color, alpha=0.10),
        ]
        labels = [
            "Kalman mean",
            "Method mean",
            "Kalman 95% ellipse",
            "Method 95% ellipse",
        ]
        _save_horizontal_legend_image(
            prefix=save_name,
            handles=handles,
            labels=labels,
            save_pdf=save_pdf,
            dpi=150,
            fontsize=11,
            frameon=True,
            scale=1.0,
        )

    plt.close(fig)

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
    figsize: Tuple[float, float] = (14, 4),
    legend_in_figure: bool = True,
    ensemble_color: str = 'red',
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
        legend_in_figure (bool, optional): Whether to keep the original in-figure
            annotation style. If False, hides axis labels while keeping ticks and
            uses standalone legend output.
    """

    J_p, N, d_p = particles.shape
    ensemble_color = str(ensemble_color or 'red')
    ensemble_color_lower = ensemble_color.lower()
    spread_cmap_name = 'Blues' if 'blue' in ensemble_color_lower else 'Reds'
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
                                 label='Ensemble Mean', color=ensemble_color, linestyle='--', linewidth=1.5)
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
                               alpha=0.3, color=ensemble_color, 
                               label='Mean ± 1 STD')
                
                if 'Mean ± 1 STD' not in labels:
                    fill_handle = mpatches.Patch(facecolor=ensemble_color, alpha=0.3, edgecolor=ensemble_color)
                    handles.append(fill_handle)
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
                               alpha=0.3, color=ensemble_color, 
                               label='95% Confidence Interval')
                
                if '95% Confidence Interval' not in labels:
                    fill_handle = mpatches.Patch(facecolor=ensemble_color, alpha=0.3, edgecolor=ensemble_color)
                    handles.append(fill_handle)
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
                                         facecolor=ensemble_color, edgecolor='none', alpha=0.5,
                                         label=current_label_for_spread)
                        if current_label_for_spread and 'Ensemble Spread' not in labels:
                            handles.append(plt.Rectangle((0, 0), 1, 1, fc=ensemble_color, alpha=0.5))
                            labels.append('Ensemble Spread')
                            ensemble_spread_labeled = True

                    elif mode == 'color':
                        cmap = plt.colormaps.get_cmap(spread_cmap_name)
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

        if legend_in_figure and fontsize is not None:
            ax.set_xlabel('Time Step', fontsize=fontsize)
            ax.set_ylabel(f'Dimension {dim_idx} Values', fontsize=fontsize)
            ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        if fontsize is not None:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            ax.grid(True)
            if legend_in_figure:
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
                legend_handles.append(plt.Line2D([0], [0], color=ensemble_color, linestyle='--', linewidth=1.5))
            elif label == 'Mean ± 1 STD':
                legend_handles.append(mpatches.Patch(facecolor=ensemble_color, alpha=0.3, edgecolor=ensemble_color))
            elif label == '95% Confidence Interval':
                legend_handles.append(mpatches.Patch(facecolor=ensemble_color, alpha=0.3, edgecolor=ensemble_color))
            elif label == 'Ensemble Spread':
                if mode == 'width':
                    legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=ensemble_color, alpha=0.5))
                elif mode == 'color':
                    cmap = plt.colormaps.get_cmap(spread_cmap_name)
                    legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=cmap(0.5), alpha=0.7))

        if legend_handles:
            if save_fig:
                _save_horizontal_legend_image(
                    prefix=save_name,
                    handles=legend_handles,
                    labels=legend_labels,
                    save_pdf=save_pdf,
                    dpi=150,
                    fontsize=24,
                    frameon=False,
                )
            else:
                fig_legend = plt.figure(figsize=(figsize[0], 0.5))
                ax_legend = fig_legend.add_subplot(111)
                ax_legend.axis('off')
                ax_legend.legend(
                    legend_handles,
                    legend_labels,
                    loc='center',
                    ncol=len(legend_handles),
                    fontsize=24,
                    frameon=False,
                )
                fig_legend.tight_layout(pad=0.1)
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
    plot_horizontal_colorbar: bool = False,
    legend_in_figure: bool = True,
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

def _save_separate_legend(
    prefix: str,
    include_prior: bool = True,
    include_posterior: bool = True,
    include_obs: bool = False,
    include_history: bool = True,
    include_true_state: bool = True,
    dpi: int = 200,
) -> None:
    """Create and save a standalone horizontal legend image."""
    handle_prior_density = mpatches.Patch(
        facecolor='blue',
        alpha=0.22,
        edgecolor='none',
        label='predictive density',
    )
    handle_prior = Line2D([0], [0], marker='o', linestyle='None',
                          markerfacecolor='blue', markeredgecolor='none', markersize=8,
                          label='predictive ensemble')
    handle_posterior_density = mpatches.Patch(
        facecolor='red',
        alpha=0.22,
        edgecolor='none',
        label='filtering density',
    )
    handle_posterior = Line2D([0], [0], marker='o', linestyle='None',
                              markerfacecolor='red', markeredgecolor='white', markersize=8,
                              label='filtering ensemble')
    handle_obs = Line2D([0], [0], marker='*', linestyle='None',
                        markerfacecolor='orange', markeredgecolor='black', markersize=14,
                        label='Observation')
    handle_traj = Line2D([0], [0], linestyle='-', color='black', linewidth=2,
                         label='History trajectory')
    handle_true = Line2D([0], [0], marker='x', linestyle='None',
                         color='black', markeredgecolor='black', markersize=10,
                         label='True state')

    handles = []
    if include_prior:
        handles.append(handle_prior_density)
        handles.append(handle_prior)
    if include_posterior:
        handles.append(handle_posterior_density)
        handles.append(handle_posterior)
    if include_obs:
        handles.append(handle_obs)
    if include_true_state:
        handles.append(handle_true)
    if include_history:
        handles.append(handle_traj)

    labels = [h.get_label() for h in handles]
    _save_horizontal_legend_image(
        prefix=prefix,
        handles=handles,
        labels=labels,
        save_pdf=False,
        dpi=dpi,
        fontsize=11,
        frameon=True,
    )


def _save_combined_prior_post_legend(
    prefix: str,
    include_obs: bool = False,
    include_history: bool = False,
    include_true_state: bool = True,
    dpi: int = 200,
) -> None:
    """Save one combined legend where predictive/filtering density stays on row 2."""
    top_handles: List = [
        Line2D(
            [0], [0],
            marker='o', linestyle='None',
            markerfacecolor='blue', markeredgecolor='none', markersize=8,
            label='predictive ensemble',
        ),
        Line2D(
            [0], [0],
            marker='o', linestyle='None',
            markerfacecolor='red', markeredgecolor='white', markersize=8,
            label='filtering ensemble',
        ),
    ]
    if include_obs:
        top_handles.append(
            Line2D(
                [0], [0],
                marker='*', linestyle='None',
                markerfacecolor='orange', markeredgecolor='black', markersize=14,
                label='Observation',
            )
        )
    if include_history:
        top_handles.append(
            Line2D([0], [0], linestyle='-', color='black', linewidth=2, label='History trajectory')
        )
    if include_true_state:
        top_handles.append(
            Line2D(
                [0], [0],
                marker='x', linestyle='None',
                color='black', markeredgecolor='black', markersize=10,
                label='True state',
            )
        )

    bottom_handles: List = [
        mpatches.Patch(facecolor='blue', alpha=0.22, edgecolor='none', label='predictive density'),
        mpatches.Patch(facecolor='red', alpha=0.22, edgecolor='none', label='filtering density'),
    ]

    _save_two_row_legend_image(
        prefix=prefix,
        top_handles=top_handles,
        top_labels=[h.get_label() for h in top_handles],
        bottom_handles=bottom_handles,
        bottom_labels=[h.get_label() for h in bottom_handles],
        save_pdf=False,
        dpi=dpi,
        fontsize=11,
        frameon=True,
    )


def _sample_points(points: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Randomly sample up to `sample_size` points without replacement."""
    n = points.shape[0]
    if sample_size <= 0 or sample_size >= n:
        return points
    idx = torch.randperm(n)[:sample_size]
    return points[idx]


def _prepare_scatter_points(points: torch.Tensor, max_points: int = PF_MAX_SCATTER_POINTS) -> Tuple[torch.Tensor, bool]:
    """Downsample points for scatter rendering and return whether downsampling happened."""
    n = points.shape[0]
    if max_points <= 0 or n <= max_points:
        return points, False
    idx = torch.randperm(n)[:max_points]
    return points[idx], True


def _adaptive_scatter_style(
    n_points: int,
    is_3d: bool = False,
    downsampled: bool = False,
) -> Tuple[float, float]:
    """Choose scatter marker size/alpha based on plotted count."""
    n = max(int(n_points), 1)
    if n <= 2000:
        size, alpha = (9.0 if not is_3d else 8.0, 0.38 if not is_3d else 0.34)
    elif n <= 20000:
        size, alpha = (5.0 if not is_3d else 4.5, 0.22 if not is_3d else 0.20)
    else:
        size, alpha = (2.2 if not is_3d else 2.0, 0.11 if not is_3d else 0.10)

    if downsampled:
        # Requested style tweak for hard-capped scatter (<=50k):
        # slightly larger markers and slightly lower opacity.
        size *= 1.25
        alpha *= 0.8
    return float(size), float(alpha)


def _compute_axis_limits(values: np.ndarray, pad_ratio: float = 0.06) -> Tuple[float, float]:
    """Compute robust plotting limits with padding."""
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return (-1.0, 1.0)
    v_min = float(np.min(finite_vals))
    v_max = float(np.max(finite_vals))
    span = v_max - v_min
    if span <= 1e-12:
        delta = max(1.0, abs(v_max) * 0.1 + 1e-3)
        return (v_min - delta, v_max + delta)
    pad = max(span * pad_ratio, 1e-6)
    return (v_min - pad, v_max + pad)


def _zoom_interval_from_fixed_range(
    values: np.ndarray,
    fixed_range: Tuple[float, float],
    num_splits: int = 10,
) -> Tuple[float, float]:
    """
    Build adaptive 1D interval from current data range.

    Rule:
    - Let current range be [min(values), max(values)] (after clipping to fixed range).
    - Expand both sides by `min(0.15 * span, 5)`, where span = max - min.
    - Integerize all bounds: floor for lower bound, ceil for upper bound.
    - Keep interval within fixed range.
    """
    lo_fixed = float(fixed_range[0])
    hi_fixed = float(fixed_range[1])
    if not np.isfinite(lo_fixed) or not np.isfinite(hi_fixed) or hi_fixed <= lo_fixed:
        return (-1.0, 1.0)

    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        lo_out = float(np.floor(lo_fixed))
        hi_out = float(np.ceil(hi_fixed))
        if hi_out <= lo_out:
            lo_out -= 1.0
            hi_out += 1.0
        return (lo_out, hi_out)

    _ = num_splits
    vals = np.clip(vals, lo_fixed, hi_fixed)
    cur_lo = float(np.min(vals))
    cur_hi = float(np.max(vals))
    if not np.isfinite(cur_lo) or not np.isfinite(cur_hi):
        lo_out = float(np.floor(lo_fixed))
        hi_out = float(np.ceil(hi_fixed))
        if hi_out <= lo_out:
            lo_out -= 1.0
            hi_out += 1.0
        return (lo_out, hi_out)

    span = max(cur_hi - cur_lo, 0.0)
    expand = min(0.15 * span, 5.0)

    lo = cur_lo - expand
    hi = cur_hi + expand
    lo = max(lo, lo_fixed)
    hi = min(hi, hi_fixed)
    lo = float(np.floor(lo))
    hi = float(np.ceil(hi))

    if hi <= lo:
        center = 0.5 * (cur_lo + cur_hi)
        lo = float(np.floor(center - 1.0))
        hi = float(np.ceil(center + 1.0))
        lo = max(lo, float(np.floor(lo_fixed)))
        hi = min(hi, float(np.ceil(hi_fixed)))
        if hi <= lo:
            lo = float(np.floor(lo_fixed))
            hi = float(np.ceil(hi_fixed))
            if hi <= lo:
                lo -= 1.0
                hi += 1.0
    return (lo, hi)


def _compute_zoomed_ranges_from_fixed_3d(
    points_xyz: np.ndarray,
    fixed_limits: Dict[str, Tuple[float, float]],
    num_splits: int = 10,
) -> Dict[str, Tuple[float, float]]:
    """Compute per-axis zoom-in ranges from fixed 3D limits."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return {
            "xlim": tuple(fixed_limits["xlim"]),
            "ylim": tuple(fixed_limits["ylim"]),
            "zlim": tuple(fixed_limits["zlim"]),
        }
    pts = pts[:, :3]
    return {
        "xlim": _zoom_interval_from_fixed_range(pts[:, 0], fixed_limits["xlim"], num_splits=num_splits),
        "ylim": _zoom_interval_from_fixed_range(pts[:, 1], fixed_limits["ylim"], num_splits=num_splits),
        "zlim": _zoom_interval_from_fixed_range(pts[:, 2], fixed_limits["zlim"], num_splits=num_splits),
    }


def _adaptive_projection_kde_std(n_samples: int, base_std: float) -> float:
    """
    Adaptive isotropic KDE std for low-sample 2D projections.
    - Uses `base_std` at n=200.
    - Smooths more when n is smaller.
    """
    n = max(int(n_samples), 1)
    std = float(base_std) * np.sqrt(200.0 / float(n))
    return float(np.clip(std, 0.65 * float(base_std), 2.8 * float(base_std)))


def _isotropic_kde_2d(
    xy: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    std: float,
    grid_size: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple isotropic Gaussian KDE on a 2D grid (numpy-only)."""
    n_grid = max(40, int(grid_size))
    xg = np.linspace(float(xlim[0]), float(xlim[1]), n_grid, dtype=np.float64)
    yg = np.linspace(float(ylim[0]), float(ylim[1]), n_grid, dtype=np.float64)
    xx, yy = np.meshgrid(xg, yg)

    samples = np.asarray(xy, dtype=np.float64)
    samples = samples[np.isfinite(samples).all(axis=1)]
    if samples.size == 0:
        return xx, yy, np.zeros_like(xx)

    std = float(max(std, 1e-6))
    dx = (xx[None, :, :] - samples[:, 0][:, None, None]) / std
    dy = (yy[None, :, :] - samples[:, 1][:, None, None]) / std
    kernel = np.exp(-0.5 * (dx * dx + dy * dy))
    density = np.mean(kernel, axis=0) / (2.0 * np.pi * std * std)
    return xx, yy, density


def _empirical_1d_w2(x: np.ndarray, y: np.ndarray) -> float:
    """Approximate 1D Wasserstein-2 distance between empirical samples."""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_sorted = np.sort(x.astype(np.float64))
    y_sorted = np.sort(y.astype(np.float64))
    n_q = int(max(x_sorted.size, y_sorted.size))
    if n_q <= 1:
        return float(abs(x_sorted[0] - y_sorted[0]))
    q = np.linspace(0.0, 1.0, n_q, dtype=np.float64)
    px = np.linspace(0.0, 1.0, x_sorted.size, dtype=np.float64)
    py = np.linspace(0.0, 1.0, y_sorted.size, dtype=np.float64)
    x_q = np.interp(q, px, x_sorted)
    y_q = np.interp(q, py, y_sorted)
    return float(np.sqrt(np.mean((x_q - y_q) ** 2)))


def _sliced_w2_distance(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    num_directions: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute sliced Wasserstein-2 distance in R^d using random projections."""
    if rng is None:
        rng = np.random.default_rng()
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Sliced W2 expects two 2D arrays with matching feature dimensions.")
    d = a.shape[1]
    if d <= 0:
        return float("nan")
    k = max(1, int(num_directions))
    dirs = rng.normal(size=(k, d))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.clip(norms, 1e-12, None)

    proj_a = a @ dirs.T
    proj_b = b @ dirs.T
    w2_list = []
    for j in range(k):
        w2_j = _empirical_1d_w2(proj_a[:, j], proj_b[:, j])
        if np.isfinite(w2_j):
            w2_list.append(w2_j)
    if len(w2_list) == 0:
        return float("nan")
    return float(np.mean(w2_list))


def _fitted_gaussian_swd_ratio(
    points: np.ndarray,
    num_directions: int = 50,
    num_reference_samples: int = 1000000,
) -> Tuple[float, float, float]:
    """Return (ratio, swd(points, gaussian), swd(gaussian, gaussian baseline))."""
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.ndim != 2 or pts.shape[0] < 2:
        return (float("nan"), float("nan"), float("nan"))

    d = pts.shape[1]
    mean = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 0:
        cov = np.eye(d, dtype=np.float64) * float(cov)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    n_ref = max(2000, int(num_reference_samples))
    n_pts = min(pts.shape[0], n_ref)
    rng = np.random.default_rng()

    if pts.shape[0] > n_pts:
        sub_idx = rng.choice(pts.shape[0], size=n_pts, replace=False)
        pts_eval = pts[sub_idx]
    else:
        pts_eval = pts

    gauss_ref = rng.multivariate_normal(mean, cov_psd, size=n_ref)
    gauss_a = rng.multivariate_normal(mean, cov_psd, size=n_ref)
    gauss_b = rng.multivariate_normal(mean, cov_psd, size=n_ref)

    swd_data = _sliced_w2_distance(pts_eval, gauss_ref, num_directions=num_directions, rng=rng)
    swd_base = _sliced_w2_distance(gauss_a, gauss_b, num_directions=num_directions, rng=rng)
    if not np.isfinite(swd_data) or not np.isfinite(swd_base) or abs(swd_base) < 1e-12:
        return (float("nan"), swd_data, swd_base)
    return (float(swd_data / swd_base), swd_data, swd_base)


def _resolve_true_state_xyz(
    true_state: Optional[torch.Tensor],
    batch_idx: int,
    batch_size: int,
) -> Optional[np.ndarray]:
    """Resolve true state for one batch item into a 3D numpy vector."""
    if true_state is None:
        return None
    if isinstance(true_state, torch.Tensor):
        ts = true_state.detach().cpu()
    else:
        ts = torch.as_tensor(true_state)

    out = None
    if ts.ndim == 1:
        if ts.numel() >= 3:
            out = ts[:3]
    elif ts.ndim == 2:
        if ts.shape[0] == batch_size and ts.shape[1] >= 3:
            out = ts[batch_idx, :3]
        elif ts.shape[1] >= 3:
            out = ts[0, :3]
    if out is None:
        return None
    return out.numpy()


def _plot_pf_projection_2d(
    ax,
    points: np.ndarray,
    point_color: str,
    point_label: str,
    dim_x: int,
    dim_y: int,
    marker_size: float,
    marker_alpha: float,
    true_xyz: Optional[np.ndarray] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    legend_in_figure: bool = True,
    dataset: Optional[str] = None,
    kde_threshold: int = 200,
    contour_grid_bins: int = 200,
    draw_scatter: bool = True,
    draw_contour: bool = True,
) -> None:
    """Plot a single PF distribution in one 2D projection with scatter."""
    xy = points[:, [dim_x, dim_y]]

    if xlim is None:
        xlim = _compute_axis_limits(xy[:, 0])
    if ylim is None:
        ylim = _compute_axis_limits(xy[:, 1])

    _ = dataset
    _ = kde_threshold
    scatter_size = float(marker_size)
    scatter_alpha = float(min(0.98, marker_alpha))

    if draw_contour:
        _draw_density_contours_2d(
            ax=ax,
            points_xy=xy,
            xlim=xlim,
            ylim=ylim,
            point_color=point_color,
            grid_bins=contour_grid_bins,
        )

    if draw_scatter:
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=scatter_size, alpha=scatter_alpha, c=point_color, edgecolors='none', zorder=4,
            label=point_label,
        )

    if true_xyz is not None and len(true_xyz) >= 3 and np.all(np.isfinite(true_xyz[:3])):
        ax.scatter(
            true_xyz[dim_x], true_xyz[dim_y],
            marker='x', s=120, c='black', linewidths=2.0, zorder=12,
            label='True state',
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if legend_in_figure:
        ax.legend(loc='best', fontsize=8, frameon=True)


def _draw_density_contours_2d(
    ax,
    points_xy: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    point_color: str,
    grid_bins: int = 200,
    alpha_scale: float = 1.0,
) -> None:
    """Draw filled density contours from a regular 2D histogram grid."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 8:
        return
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 8:
        return

    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    if (not np.isfinite(x0)) or (not np.isfinite(x1)) or (not np.isfinite(y0)) or (not np.isfinite(y1)):
        return
    if x1 <= x0 or y1 <= y0:
        return

    bins = int(max(16, grid_bins))
    try:
        hist, x_edges, y_edges = np.histogram2d(
            pts[:, 0],
            pts[:, 1],
            bins=[bins, bins],
            range=[[x0, x1], [y0, y1]],
            density=True,
        )
    except Exception:
        return

    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    if hist.size == 0 or not np.any(hist > 0):
        return

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    zz = hist.T
    # Light smoothing spreads tiny mass outward so outer contours are visible.
    for _ in range(2):
        pad = np.pad(zz, ((1, 1), (1, 1)), mode="edge")
        zz = (
            pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
            pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
            pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
        ) / 9.0
    positive = zz[zz > 0.0]
    if positive.size == 0:
        return
    z_min = float(np.min(positive))
    z_max = float(np.max(positive))
    if (not np.isfinite(z_min)) or (not np.isfinite(z_max)) or z_max <= z_min:
        return
    # More contour bands with log-spaced levels to emphasize outer low-density structure.
    levels = np.geomspace(max(z_min, 1e-12), z_max, 16, dtype=np.float64)
    levels = np.unique(levels)
    if levels.size < 2:
        return

    cmap_name = "Blues" if "blue" in str(point_color).lower() else "Reds"
    cmap_base = plt.colormaps.get_cmap(cmap_name)
    # Use darker half of the colormap so contours are visually stronger.
    cmap_dark = mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_dark",
        cmap_base(np.linspace(0.35, 1.0, 256)),
    )
    alpha_scale = float(np.clip(alpha_scale, 0.0, 1.0))
    ax.contourf(
        xx,
        yy,
        zz,
        levels=levels,
        cmap=cmap_dark,
        alpha=0.68 * alpha_scale,
        zorder=1,
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        zz,
        levels=levels,
        cmap=cmap_dark,
        linewidths=0.6,
        alpha=0.9 * alpha_scale,
        zorder=2,
    )


def _integer_axis_limits(values: np.ndarray, pad: int = 0) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 1.0)
    lo = float(np.floor(np.min(vals))) - float(max(0, int(pad)))
    hi = float(np.ceil(np.max(vals))) + float(max(0, int(pad)))
    if hi <= lo:
        lo -= 1.0
        hi += 1.0
    return (lo, hi)


def _pca_limits_from_xyz_ranges(
    xyz_limits: Dict[str, Tuple[float, float]],
    mean: np.ndarray,
    basis: np.ndarray,
    pad: int = 0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Convert axis-aligned 3D limits into integer PCA(PC1, PC2) limits.

    For linear PCA projection, extrema over a 3D box are attained at box corners.
    """
    if not all(k in xyz_limits for k in ("xlim", "ylim", "zlim")):
        return ((-1.0, 1.0), (-1.0, 1.0))

    xlim = np.asarray(xyz_limits["xlim"], dtype=np.float64).reshape(-1)
    ylim = np.asarray(xyz_limits["ylim"], dtype=np.float64).reshape(-1)
    zlim = np.asarray(xyz_limits["zlim"], dtype=np.float64).reshape(-1)
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    vec = np.asarray(basis, dtype=np.float64)

    if (
        xlim.size < 2 or ylim.size < 2 or zlim.size < 2 or
        mu.size < 3 or vec.ndim != 2 or vec.shape[0] < 3 or vec.shape[1] < 2
    ):
        return ((-1.0, 1.0), (-1.0, 1.0))

    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    z0, z1 = float(zlim[0]), float(zlim[1])
    if not all(np.isfinite(v) for v in (x0, x1, y0, y1, z0, z1)):
        return ((-1.0, 1.0), (-1.0, 1.0))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if z1 < z0:
        z0, z1 = z1, z0

    corners = np.asarray(
        [
            [x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
            [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1],
        ],
        dtype=np.float64,
    )
    proj = _project_pca_2d(corners, mean=mu[:3], basis=vec[:3, :2])
    return (
        _integer_axis_limits(proj[:, 0], pad=pad),
        _integer_axis_limits(proj[:, 1], pad=pad),
    )


def _fit_pca_2d(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.ndim != 2 or pts.shape[0] < 2:
        mean = np.zeros((pts.shape[1],), dtype=np.float64) if pts.ndim == 2 else np.zeros((3,), dtype=np.float64)
        basis = np.zeros((mean.size, 2), dtype=np.float64)
        basis[0, 0] = 1.0
        if mean.size > 1:
            basis[1, 1] = 1.0
        return mean, basis
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:2]]
    if basis.shape[1] < 2:
        pad = np.zeros((basis.shape[0], 2 - basis.shape[1]), dtype=np.float64)
        basis = np.concatenate([basis, pad], axis=1)
    return mean, basis


def _project_pca_2d(points: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        return np.zeros((0, 2), dtype=np.float64)
    centered = pts - mean.reshape(1, -1)
    return centered @ basis


def _quantile_implied_density_curve(
    quantiles: np.ndarray,
    probs: np.ndarray,
    min_dqdp: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    q = np.asarray(quantiles, dtype=np.float64).reshape(-1)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if q.size == 0 or p.size == 0 or q.size != p.size:
        return np.array([]), np.array([])
    mask = np.isfinite(q) & np.isfinite(p)
    q = q[mask]
    p = p[mask]
    if q.size < 3:
        return np.array([]), np.array([])
    order = np.argsort(p)
    p = p[order]
    q = q[order]
    p_unique, idx_unique = np.unique(p, return_index=True)
    q = q[idx_unique]
    p = p_unique
    if q.size < 3:
        return np.array([]), np.array([])
    eps = 1e-10
    q_mono = np.maximum.accumulate(q + eps * np.arange(q.size, dtype=np.float64))
    dqdp = np.gradient(q_mono, p)
    dqdp = np.clip(dqdp, float(min_dqdp), None)
    density = 1.0 / dqdp
    area = np.trapz(density, q_mono)
    if np.isfinite(area) and area > 1e-12:
        density = density / area
    return q_mono, density


def _save_quantile_density_plot(
    x: np.ndarray,
    density: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    if x.size == 0 or density.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(x, density, color="tab:blue", linewidth=1.8)
    ax.fill_between(x, 0.0, density, color="tab:blue", alpha=0.18)
    ax.set_xlabel("state value")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_and_test_point_clouds(
    args: Namespace,
    prior_tensor: torch.Tensor,
    posterior_tensor: torch.Tensor,
    num_samples_plot: int,
    prefix: str,
    num_swd_reference_samples: int = 1000000,
    num_swd_directions: int = 50,
    plot_indices: Optional[List[int]] = None,
    history_traj: Optional[torch.Tensor] = None,
    true_state: Optional[torch.Tensor] = None,
    legend_in_figure: bool = True,
    prior_range_int: Optional[torch.Tensor] = None,
    post_range_int: Optional[torch.Tensor] = None,
    prior_pca_range_int: Optional[torch.Tensor] = None,
    post_pca_range_int: Optional[torch.Tensor] = None,
    quantile_probs: Optional[torch.Tensor] = None,
    prior_quantiles: Optional[torch.Tensor] = None,
    post_quantiles: Optional[torch.Tensor] = None,
    prior_pca_quantiles: Optional[torch.Tensor] = None,
    post_pca_quantiles: Optional[torch.Tensor] = None,
    scatter_size_scale: float = 1.0,
    prior_pca_source_tensor: Optional[torch.Tensor] = None,
    posterior_pca_source_tensor: Optional[torch.Tensor] = None,
    prior_pca_mean: Optional[torch.Tensor] = None,
    post_pca_mean: Optional[torch.Tensor] = None,
    prior_pca_basis: Optional[torch.Tensor] = None,
    post_pca_basis: Optional[torch.Tensor] = None,
):
    """
    PF visualization for non-ring datasets (dim>=3):
    - For each mode (prior/post) and range mode (fixed/adaptive), save separate 2D views:
      12, 23, 13, and PCA12 (no grid).
    - Save one fixed-range 3D view with fixed camera angle.
    - Save 6 quantile-density figures: dim1-3 and PCA1-3.
    - Adaptive ranges can be provided from cached PF range statistics.
    """
    dataset = str(getattr(args, "dataset", "")).lower()
    scatter_size_scale = float(max(0.1, scatter_size_scale))
    if dataset not in PF_3D_SUPPORTED_DATASETS:
        raise ValueError(
            f"PF 3D plotter supports only {sorted(PF_3D_SUPPORTED_DATASETS)}, got dataset='{dataset}'."
        )

    if prior_tensor.is_cuda:
        prior_tensor = prior_tensor.cpu()
    if posterior_tensor.is_cuda:
        posterior_tensor = posterior_tensor.cpu()
    if prior_pca_source_tensor is not None and prior_pca_source_tensor.is_cuda:
        prior_pca_source_tensor = prior_pca_source_tensor.cpu()
    if posterior_pca_source_tensor is not None and posterior_pca_source_tensor.is_cuda:
        posterior_pca_source_tensor = posterior_pca_source_tensor.cpu()
    if prior_pca_mean is not None and isinstance(prior_pca_mean, torch.Tensor) and prior_pca_mean.is_cuda:
        prior_pca_mean = prior_pca_mean.cpu()
    if post_pca_mean is not None and isinstance(post_pca_mean, torch.Tensor) and post_pca_mean.is_cuda:
        post_pca_mean = post_pca_mean.cpu()
    if prior_pca_basis is not None and isinstance(prior_pca_basis, torch.Tensor) and prior_pca_basis.is_cuda:
        prior_pca_basis = prior_pca_basis.cpu()
    if post_pca_basis is not None and isinstance(post_pca_basis, torch.Tensor) and post_pca_basis.is_cuda:
        post_pca_basis = post_pca_basis.cpu()
    if history_traj is not None and history_traj.is_cuda:
        history_traj = history_traj.cpu()
    if true_state is not None and isinstance(true_state, torch.Tensor) and true_state.is_cuda:
        true_state = true_state.cpu()

    if prior_tensor.ndim != 3 or posterior_tensor.ndim != 3:
        raise ValueError(
            f"Expected prior/posterior shape (B, N, D), got {tuple(prior_tensor.shape)} and {tuple(posterior_tensor.shape)}."
        )

    Bp, _, Dp = prior_tensor.shape
    Bq, _, Dq = posterior_tensor.shape
    if Bp != Bq:
        raise ValueError(f"Batch mismatch between prior and posterior: {Bp} vs {Bq}.")
    if Dp < 3 or Dq < 3:
        raise ValueError(f"PF 3D plotter expects at least 3 dims, got D={Dp} and D={Dq}.")
    if Dp != Dq:
        raise ValueError(f"State dim mismatch between prior and posterior: {Dp} vs {Dq}.")

    prior_pca_source_full = prior_tensor if prior_pca_source_tensor is None else prior_pca_source_tensor
    post_pca_source_full = posterior_tensor if posterior_pca_source_tensor is None else posterior_pca_source_tensor
    if prior_pca_source_full.ndim != 3 or post_pca_source_full.ndim != 3:
        raise ValueError(
            f"PCA source tensors must have shape (B, N, D), got {tuple(prior_pca_source_full.shape)} "
            f"and {tuple(post_pca_source_full.shape)}."
        )
    if prior_pca_source_full.shape[0] != Bp or post_pca_source_full.shape[0] != Bq:
        raise ValueError(
            f"PCA source batch mismatch: {prior_pca_source_full.shape[0]} vs {Bp}, "
            f"{post_pca_source_full.shape[0]} vs {Bq}."
        )

    # For dim>3, this visualization intentionally uses the first 3 dims.
    prior_tensor = prior_tensor[..., :3]
    posterior_tensor = posterior_tensor[..., :3]
    B = Bp

    if history_traj is not None:
        if history_traj.ndim != 3:
            print("Warning: history_traj must be (T, B, D). Ignoring trajectory.")
            history_traj = None
        else:
            _, Bh, Dh = history_traj.shape
            if Bh != B or Dh < 3:
                print("Warning: history_traj shape is incompatible. Ignoring trajectory.")
                history_traj = None
            else:
                history_traj = history_traj[..., :3]

    if plot_indices is None:
        indices_to_process = list(range(B))
    else:
        indices_to_process = [idx for idx in plot_indices if 0 <= idx < B]

    if len(indices_to_process) == 0:
        print("Warning: no valid plot indices were provided.")
        return []

    distance_records: List[Dict[str, Any]] = []
    contour_grid_bins = int(max(16, int(getattr(args, "pf_contour_grid_bins", 200))))

    def _merge_mode_ranges(
        range_a: Optional[torch.Tensor],
        range_b: Optional[torch.Tensor],
        min_axes: int,
    ) -> Optional[np.ndarray]:
        ranges = []
        for r_t in [range_a, range_b]:
            if r_t is None:
                continue
            r_np = torch.as_tensor(r_t).detach().cpu().numpy()
            if (
                r_np.ndim == 2
                and r_np.shape[0] >= min_axes
                and r_np.shape[1] >= 2
                and np.isfinite(r_np[:min_axes, :2]).all()
                and np.all(r_np[:min_axes, 1] > r_np[:min_axes, 0])
            ):
                ranges.append(r_np[:min_axes, :2].astype(np.float64))
        if len(ranges) == 0:
            return None
        stacked = np.stack(ranges, axis=0)  # [M, A, 2]
        lo = np.min(stacked[:, :, 0], axis=0)
        hi = np.max(stacked[:, :, 1], axis=0)
        if not np.all(hi > lo):
            return None
        return np.stack([lo, hi], axis=1)

    def _range2d_from_cached(range_t: Optional[torch.Tensor]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if range_t is None:
            return None
        r_np = torch.as_tensor(range_t).detach().cpu().numpy()
        if (
            r_np.ndim != 2
            or r_np.shape[0] < 2
            or r_np.shape[1] < 2
            or not np.isfinite(r_np[:2, :2]).all()
            or not np.all(r_np[:2, 1] > r_np[:2, 0])
        ):
            return None
        return (
            (float(r_np[0, 0]), float(r_np[0, 1])),
            (float(r_np[1, 0]), float(r_np[1, 1])),
        )

    def _parse_pca_transform(
        mean_t: Optional[torch.Tensor],
        basis_t: Optional[torch.Tensor],
        source_dim: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if mean_t is None or basis_t is None:
            return None
        mean_np = torch.as_tensor(mean_t).detach().cpu().numpy().astype(np.float64).reshape(-1)
        basis_np = torch.as_tensor(basis_t).detach().cpu().numpy().astype(np.float64)
        if mean_np.size != int(source_dim):
            return None
        if basis_np.ndim != 2 or basis_np.shape[0] != int(source_dim) or basis_np.shape[1] < 1:
            return None
        basis_np = basis_np[:, : min(2, basis_np.shape[1])]
        if basis_np.shape[1] < 2:
            pad = np.zeros((basis_np.shape[0], 2 - basis_np.shape[1]), dtype=np.float64)
            basis_np = np.concatenate([basis_np, pad], axis=1)
        return mean_np, basis_np

    def _limits_from_projected_points(points_xy: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        pts = np.asarray(points_xy, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return ((-1.0, 1.0), (-1.0, 1.0))
        return (
            _integer_axis_limits(pts[:, 0], pad=0),
            _integer_axis_limits(pts[:, 1], pad=0),
        )

    fixed_limits = PF_FIXED_RANGES_3D[dataset]
    projection_specs = [
        ("xy", 0, 1, "x", "y", fixed_limits["xlim"], fixed_limits["ylim"]),
        ("yz", 1, 2, "y", "z", fixed_limits["ylim"], fixed_limits["zlim"]),
        ("xz", 0, 2, "x", "z", fixed_limits["xlim"], fixed_limits["zlim"]),
    ]

    for i in indices_to_process:
        prior_full = prior_tensor[i, :, :]
        post_full = posterior_tensor[i, :, :]
        true_xyz = _resolve_true_state_xyz(true_state=true_state, batch_idx=i, batch_size=B)
        adaptive_limits = _compute_zoomed_ranges_from_fixed_3d(
            points_xyz=torch.cat([prior_full, post_full], dim=0).numpy(),
            fixed_limits=fixed_limits,
            num_splits=10,
        )
        shared_mode_range_3d = _merge_mode_ranges(prior_range_int, post_range_int, min_axes=3)
        if shared_mode_range_3d is not None:
            adaptive_limits = {
                "xlim": (float(shared_mode_range_3d[0, 0]), float(shared_mode_range_3d[0, 1])),
                "ylim": (float(shared_mode_range_3d[1, 0]), float(shared_mode_range_3d[1, 1])),
                "zlim": (float(shared_mode_range_3d[2, 0]), float(shared_mode_range_3d[2, 1])),
            }

        prior_for_vis = _sample_points(prior_full, num_samples_plot)
        post_for_vis = _sample_points(post_full, num_samples_plot)

        prior_scatter_ds, prior_downsampled = _prepare_scatter_points(
            prior_for_vis, max_points=PF_MAX_SCATTER_POINTS
        )
        post_scatter_ds, post_downsampled = _prepare_scatter_points(
            post_for_vis, max_points=PF_MAX_SCATTER_POINTS
        )
        prior_contour_ds, _ = _prepare_scatter_points(
            prior_for_vis, max_points=PF_MAX_CONTOUR_POINTS
        )
        post_contour_ds, _ = _prepare_scatter_points(
            post_for_vis, max_points=PF_MAX_CONTOUR_POINTS
        )

        prior_scatter = prior_scatter_ds.numpy()
        post_scatter = post_scatter_ds.numpy()
        prior_contour = prior_contour_ds.numpy()
        post_contour = post_contour_ds.numpy()

        # Legacy shared PCA basis from the displayed 3D state cloud.
        all_full = torch.cat([prior_full, post_full], dim=0).numpy()
        pca_mean, pca_basis = _fit_pca_2d(all_full)
        prior_scatter_pca = _project_pca_2d(prior_scatter, pca_mean, pca_basis)
        post_scatter_pca = _project_pca_2d(post_scatter, pca_mean, pca_basis)
        prior_contour_pca = _project_pca_2d(prior_contour, pca_mean, pca_basis)
        post_contour_pca = _project_pca_2d(post_contour, pca_mean, pca_basis)
        true_pca = None
        if true_xyz is not None and len(true_xyz) >= 3 and np.all(np.isfinite(true_xyz[:3])):
            true_pca = _project_pca_2d(np.asarray(true_xyz[:3], dtype=np.float64).reshape(1, 3), pca_mean, pca_basis)[0]

        # Optional cached PF PCA transform (test-time only): center by cached PF mean
        # and project using cached PF eigenvectors so CRPS and PCA12 plots share the
        # same coordinate system.
        prior_pca_full_item = prior_pca_source_full[i, :, :]
        post_pca_full_item = post_pca_source_full[i, :, :]
        prior_pca_for_vis = _sample_points(prior_pca_full_item, num_samples_plot)
        post_pca_for_vis = _sample_points(post_pca_full_item, num_samples_plot)
        prior_pca_scatter_ds, _ = _prepare_scatter_points(prior_pca_for_vis, max_points=PF_MAX_SCATTER_POINTS)
        post_pca_scatter_ds, _ = _prepare_scatter_points(post_pca_for_vis, max_points=PF_MAX_SCATTER_POINTS)
        prior_pca_contour_ds, _ = _prepare_scatter_points(prior_pca_for_vis, max_points=PF_MAX_CONTOUR_POINTS)
        post_pca_contour_ds, _ = _prepare_scatter_points(post_pca_for_vis, max_points=PF_MAX_CONTOUR_POINTS)
        prior_pca_scatter_src = prior_pca_scatter_ds.numpy()
        post_pca_scatter_src = post_pca_scatter_ds.numpy()
        prior_pca_contour_src = prior_pca_contour_ds.numpy()
        post_pca_contour_src = post_pca_contour_ds.numpy()

        true_state_full_np = None
        if true_state is not None:
            ts_full = torch.as_tensor(true_state).detach().cpu()
            if ts_full.ndim == 1:
                true_state_full_np = ts_full.numpy().astype(np.float64)
            elif ts_full.ndim == 2:
                if ts_full.shape[0] == B:
                    true_state_full_np = ts_full[i].numpy().astype(np.float64)
                elif ts_full.shape[0] > 0:
                    true_state_full_np = ts_full[0].numpy().astype(np.float64)

        fixed_xyz = {
            "xlim": fixed_limits["xlim"],
            "ylim": fixed_limits["ylim"],
            "zlim": fixed_limits["zlim"],
        }

        prior_cached_transform = _parse_pca_transform(
            prior_pca_mean,
            prior_pca_basis,
            source_dim=int(prior_pca_full_item.shape[-1]),
        )
        post_cached_transform = _parse_pca_transform(
            post_pca_mean,
            post_pca_basis,
            source_dim=int(post_pca_full_item.shape[-1]),
        )

        prior_scatter_pca_custom = None
        prior_contour_pca_custom = None
        prior_true_pca_custom = None
        prior_fixed_pca_custom = None
        prior_adapt_pca_custom = _range2d_from_cached(prior_pca_range_int)
        if prior_cached_transform is not None:
            prior_mean_np, prior_basis_np = prior_cached_transform
            prior_scatter_pca_custom = _project_pca_2d(prior_pca_scatter_src, prior_mean_np, prior_basis_np)
            prior_contour_pca_custom = _project_pca_2d(prior_pca_contour_src, prior_mean_np, prior_basis_np)
            if true_state_full_np is not None and true_state_full_np.size == prior_mean_np.size:
                prior_true_pca_custom = _project_pca_2d(
                    true_state_full_np.reshape(1, -1),
                    prior_mean_np,
                    prior_basis_np,
                )[0]
            if prior_mean_np.size >= 3 and prior_basis_np.shape[0] >= 3:
                prior_fixed_pca_custom = _pca_limits_from_xyz_ranges(
                    xyz_limits=fixed_xyz,
                    mean=prior_mean_np[:3],
                    basis=prior_basis_np[:3, :2],
                    pad=0,
                )
            if prior_fixed_pca_custom is None:
                prior_fixed_pca_custom = _limits_from_projected_points(prior_contour_pca_custom)
            if prior_adapt_pca_custom is None:
                prior_adapt_pca_custom = _limits_from_projected_points(prior_contour_pca_custom)

        post_scatter_pca_custom = None
        post_contour_pca_custom = None
        post_true_pca_custom = None
        post_fixed_pca_custom = None
        post_adapt_pca_custom = _range2d_from_cached(post_pca_range_int)
        if post_cached_transform is not None:
            post_mean_np, post_basis_np = post_cached_transform
            post_scatter_pca_custom = _project_pca_2d(post_pca_scatter_src, post_mean_np, post_basis_np)
            post_contour_pca_custom = _project_pca_2d(post_pca_contour_src, post_mean_np, post_basis_np)
            if true_state_full_np is not None and true_state_full_np.size == post_mean_np.size:
                post_true_pca_custom = _project_pca_2d(
                    true_state_full_np.reshape(1, -1),
                    post_mean_np,
                    post_basis_np,
                )[0]
            if post_mean_np.size >= 3 and post_basis_np.shape[0] >= 3:
                post_fixed_pca_custom = _pca_limits_from_xyz_ranges(
                    xyz_limits=fixed_xyz,
                    mean=post_mean_np[:3],
                    basis=post_basis_np[:3, :2],
                    pad=0,
                )
            if post_fixed_pca_custom is None:
                post_fixed_pca_custom = _limits_from_projected_points(post_contour_pca_custom)
            if post_adapt_pca_custom is None:
                post_adapt_pca_custom = _limits_from_projected_points(post_contour_pca_custom)
        prior_marker_size_2d, prior_marker_alpha_2d = _adaptive_scatter_style(
            n_points=prior_scatter.shape[0],
            is_3d=False,
            downsampled=prior_downsampled,
        )
        post_marker_size_2d, post_marker_alpha_2d = _adaptive_scatter_style(
            n_points=post_scatter.shape[0],
            is_3d=False,
            downsampled=post_downsampled,
        )
        prior_marker_size_3d, prior_marker_alpha_3d = _adaptive_scatter_style(
            n_points=prior_scatter.shape[0], is_3d=True, downsampled=prior_downsampled
        )
        post_marker_size_3d, post_marker_alpha_3d = _adaptive_scatter_style(
            n_points=post_scatter.shape[0], is_3d=True, downsampled=post_downsampled
        )

        compute_swd = bool(getattr(args, "pf_swd", False))
        if compute_swd:
            swd_ratio_prior, swd_data_prior, swd_base_prior = _fitted_gaussian_swd_ratio(
                points=prior_full.numpy(),
                num_directions=num_swd_directions,
                num_reference_samples=num_swd_reference_samples,
            )
            swd_ratio_post, swd_data_post, swd_base_post = _fitted_gaussian_swd_ratio(
                points=post_full.numpy(),
                num_directions=num_swd_directions,
                num_reference_samples=num_swd_reference_samples,
            )
        else:
            swd_ratio_prior, swd_data_prior, swd_base_prior = float("nan"), float("nan"), float("nan")
            swd_ratio_post, swd_data_post, swd_base_post = float("nan"), float("nan"), float("nan")
        prior_ratio_str = f"{swd_ratio_prior:.3f}" if np.isfinite(swd_ratio_prior) else "nan"
        post_ratio_str = f"{swd_ratio_post:.3f}" if np.isfinite(swd_ratio_post) else "nan"
        prior_size_scale = scatter_size_scale if prior_scatter.shape[0] < 500 else 1.0
        post_size_scale = scatter_size_scale if post_scatter.shape[0] < 500 else 1.0
        mode_specs = [
            (
                "prior",
                prior_scatter,
                prior_contour,
                prior_scatter_pca,
                prior_contour_pca,
                prior_scatter_pca_custom,
                prior_contour_pca_custom,
                prior_true_pca_custom,
                prior_fixed_pca_custom,
                prior_adapt_pca_custom,
                "blue",
                "predictive ensemble",
                prior_ratio_str,
                prior_marker_size_2d * prior_size_scale,
                prior_marker_alpha_2d,
                prior_marker_size_3d * prior_size_scale,
                prior_marker_alpha_3d,
                prior_quantiles,
                prior_pca_quantiles,
            ),
            (
                "post",
                post_scatter,
                post_contour,
                post_scatter_pca,
                post_contour_pca,
                post_scatter_pca_custom,
                post_contour_pca_custom,
                post_true_pca_custom,
                post_fixed_pca_custom,
                post_adapt_pca_custom,
                "red",
                "filtering ensemble",
                post_ratio_str,
                post_marker_size_2d * post_size_scale,
                post_marker_alpha_2d,
                post_marker_size_3d * post_size_scale,
                post_marker_alpha_3d,
                post_quantiles,
                post_pca_quantiles,
            ),
        ]

        shared_mode_pca_range_2d = _merge_mode_ranges(prior_pca_range_int, post_pca_range_int, min_axes=2)
        shared_adapt_pca_xlim, shared_adapt_pca_ylim = _pca_limits_from_xyz_ranges(
            xyz_limits=adaptive_limits,
            mean=pca_mean,
            basis=pca_basis,
            pad=0,
        )
        if shared_mode_pca_range_2d is not None:
            shared_adapt_pca_xlim = (
                float(shared_mode_pca_range_2d[0, 0]),
                float(shared_mode_pca_range_2d[0, 1]),
            )
            shared_adapt_pca_ylim = (
                float(shared_mode_pca_range_2d[1, 0]),
                float(shared_mode_pca_range_2d[1, 1]),
            )

        for (
            mode_tag,
            mode_points_scatter,
            mode_points_contour,
            mode_points_pca_scatter_default,
            mode_points_pca_contour_default,
            mode_points_pca_scatter_custom,
            mode_points_pca_contour_custom,
            mode_true_pca_custom,
            mode_fixed_pca_custom,
            mode_adapt_pca_custom,
            mode_color,
            mode_label,
            mode_ratio,
            size_2d,
            alpha_2d,
            size_3d,
            alpha_3d,
            mode_quantiles,
            mode_pca_quantiles,
        ) in mode_specs:
            fixed_pca_xlim, fixed_pca_ylim = _pca_limits_from_xyz_ranges(
                xyz_limits=fixed_xyz,
                mean=pca_mean,
                basis=pca_basis,
                pad=0,
            )

            adapt_xyz = adaptive_limits
            adapt_pca_xlim, adapt_pca_ylim = shared_adapt_pca_xlim, shared_adapt_pca_ylim
            mode_points_pca_scatter = mode_points_pca_scatter_default
            mode_points_pca_contour = mode_points_pca_contour_default
            mode_true_pca = true_pca
            if mode_points_pca_scatter_custom is not None and mode_points_pca_contour_custom is not None:
                mode_points_pca_scatter = mode_points_pca_scatter_custom
                mode_points_pca_contour = mode_points_pca_contour_custom
                mode_true_pca = mode_true_pca_custom
                if mode_fixed_pca_custom is not None:
                    fixed_pca_xlim, fixed_pca_ylim = mode_fixed_pca_custom
                if mode_adapt_pca_custom is not None:
                    adapt_pca_xlim, adapt_pca_ylim = mode_adapt_pca_custom

            range_modes = [
                ("fixed", {"xlim": fixed_limits["xlim"], "ylim": fixed_limits["ylim"], "zlim": fixed_limits["zlim"]}, fixed_pca_xlim, fixed_pca_ylim),
                ("adaptive", adapt_xyz, adapt_pca_xlim, adapt_pca_ylim),
            ]

            for range_tag, lim_xyz, lim_pca_x, lim_pca_y in range_modes:
                # 1) 12 / 23 / 13 views as separate files: scatter and contour
                for plane_tag, dim_x, dim_y, x_label, y_label, _, _ in projection_specs:
                    if plane_tag == "xy":
                        xlim_curr, ylim_curr = lim_xyz["xlim"], lim_xyz["ylim"]
                    elif plane_tag == "yz":
                        xlim_curr, ylim_curr = lim_xyz["ylim"], lim_xyz["zlim"]
                    else:
                        xlim_curr, ylim_curr = lim_xyz["xlim"], lim_xyz["zlim"]
                    view_name = "12" if plane_tag == "xy" else ("23" if plane_tag == "yz" else "13")
                    for plot_style in ["scatter", "contour"]:
                        plot_points = mode_points_scatter if plot_style == "scatter" else mode_points_contour
                        fig, ax = plt.subplots(figsize=(7.2, 6.2))
                        _plot_pf_projection_2d(
                            ax=ax,
                            points=plot_points,
                            point_color=mode_color,
                            point_label=mode_label,
                            dim_x=dim_x,
                            dim_y=dim_y,
                            marker_size=size_2d,
                            marker_alpha=alpha_2d,
                            true_xyz=true_xyz,
                            xlim=xlim_curr,
                            ylim=ylim_curr,
                            legend_in_figure=False,
                            dataset=dataset,
                            contour_grid_bins=contour_grid_bins,
                            draw_scatter=(plot_style == "scatter"),
                            draw_contour=(plot_style == "contour"),
                        )
                        if legend_in_figure:
                            ax.set_xlabel(x_label)
                            ax.set_ylabel(y_label)
                            ax.set_title(f"{plane_tag} ({range_tag}, {plot_style})", fontsize=10)
                        else:
                            ax.set_xlabel("")
                            ax.set_ylabel("")
                            ax.set_title("")
                        fig.tight_layout()
                        if plot_style == "scatter":
                            save_path = f"{prefix}_{i}_{mode_tag}_{range_tag}_{view_name}.png"
                        else:
                            save_path = f"{prefix}_{i}_{mode_tag}_{range_tag}_{view_name}_contour.png"
                        fig.savefig(save_path, bbox_inches='tight', dpi=150)
                        plt.close(fig)

                # 2) PCA12 view as separate files: scatter and contour
                for plot_style in ["scatter", "contour"]:
                    fig_pca, ax_pca = plt.subplots(figsize=(7.2, 6.2))
                    if plot_style == "contour":
                        _draw_density_contours_2d(
                            ax=ax_pca,
                            points_xy=mode_points_pca_contour,
                            xlim=lim_pca_x,
                            ylim=lim_pca_y,
                            point_color=mode_color,
                            grid_bins=contour_grid_bins,
                        )
                    else:
                        ax_pca.scatter(
                            mode_points_pca_scatter[:, 0],
                            mode_points_pca_scatter[:, 1],
                            s=float(size_2d),
                            alpha=float(min(0.98, alpha_2d)),
                            c=mode_color,
                            edgecolors='none',
                            zorder=4,
                            label=mode_label,
                        )
                    if mode_true_pca is not None and np.all(np.isfinite(mode_true_pca)):
                        ax_pca.scatter(
                            mode_true_pca[0], mode_true_pca[1],
                            marker='x', s=120, c='black', linewidths=2.0, zorder=12, label='True state'
                        )
                    ax_pca.set_xlim(lim_pca_x)
                    ax_pca.set_ylim(lim_pca_y)
                    if legend_in_figure:
                        ax_pca.set_xlabel("PC1")
                        ax_pca.set_ylabel("PC2")
                        ax_pca.set_title(f"pca12 ({range_tag}, {plot_style})", fontsize=10)
                    else:
                        ax_pca.set_xlabel("")
                        ax_pca.set_ylabel("")
                        ax_pca.set_title("")
                    if legend_in_figure:
                        pca_handles = [
                            Line2D([0], [0], marker='x', linestyle='None', color='black', markeredgecolor='black', markersize=10),
                        ]
                        pca_labels = ["True state"]
                        if plot_style == "scatter":
                            pca_handles.insert(
                                0,
                                Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=mode_color, markeredgecolor='none', markersize=7),
                            )
                            pca_labels.insert(0, mode_label)
                        ax_pca.legend(
                            pca_handles,
                            pca_labels,
                            loc='best',
                            fontsize=8,
                            frameon=True,
                        )
                    fig_pca.tight_layout()
                    if plot_style == "scatter":
                        save_path = f"{prefix}_{i}_{mode_tag}_{range_tag}_PCA12.png"
                    else:
                        save_path = f"{prefix}_{i}_{mode_tag}_{range_tag}_PCA12_contour.png"
                    fig_pca.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.close(fig_pca)

                # 3) mixed contour views (posterior over prior), lighter colors (60%)
                if mode_tag == "post":
                    mixed_alpha_scale = 0.6
                    for plane_tag, dim_x, dim_y, x_label, y_label, _, _ in projection_specs:
                        if plane_tag == "xy":
                            xlim_curr, ylim_curr = lim_xyz["xlim"], lim_xyz["ylim"]
                        elif plane_tag == "yz":
                            xlim_curr, ylim_curr = lim_xyz["ylim"], lim_xyz["zlim"]
                        else:
                            xlim_curr, ylim_curr = lim_xyz["xlim"], lim_xyz["zlim"]
                        view_name = "12" if plane_tag == "xy" else ("23" if plane_tag == "yz" else "13")

                        fig_mix, ax_mix = plt.subplots(figsize=(7.2, 6.2))
                        _draw_density_contours_2d(
                            ax=ax_mix,
                            points_xy=prior_contour[:, [dim_x, dim_y]],
                            xlim=xlim_curr,
                            ylim=ylim_curr,
                            point_color="blue",
                            grid_bins=contour_grid_bins,
                            alpha_scale=mixed_alpha_scale,
                        )
                        _draw_density_contours_2d(
                            ax=ax_mix,
                            points_xy=post_contour[:, [dim_x, dim_y]],
                            xlim=xlim_curr,
                            ylim=ylim_curr,
                            point_color="red",
                            grid_bins=contour_grid_bins,
                            alpha_scale=mixed_alpha_scale,
                        )
                        if true_xyz is not None and len(true_xyz) >= 3 and np.all(np.isfinite(true_xyz[:3])):
                            ax_mix.scatter(
                                true_xyz[dim_x], true_xyz[dim_y],
                                marker='x', s=120, c='black', linewidths=2.0, zorder=12, label='True state',
                            )
                        ax_mix.set_xlim(xlim_curr)
                        ax_mix.set_ylim(ylim_curr)
                        if legend_in_figure:
                            ax_mix.set_xlabel(x_label)
                            ax_mix.set_ylabel(y_label)
                            ax_mix.set_title(f"{plane_tag} ({range_tag}, mixed contour)", fontsize=10)
                            ax_mix.legend(
                                [
                                    Line2D([0], [0], color='blue', linewidth=2.0),
                                    Line2D([0], [0], color='red', linewidth=2.0),
                                    Line2D([0], [0], marker='x', linestyle='None', color='black', markeredgecolor='black', markersize=10),
                                ],
                                ["Prior contour", "Posterior contour", "True state"],
                                loc='best',
                                fontsize=8,
                                frameon=True,
                            )
                        else:
                            ax_mix.set_xlabel("")
                            ax_mix.set_ylabel("")
                            ax_mix.set_title("")
                        fig_mix.tight_layout()
                        fig_mix.savefig(
                            f"{prefix}_{i}_mixed_{range_tag}_{view_name}_contour.png",
                            bbox_inches='tight',
                            dpi=150,
                        )
                        plt.close(fig_mix)

                    fig_mix_pca, ax_mix_pca = plt.subplots(figsize=(7.2, 6.2))
                    mix_prior_pca = prior_contour_pca
                    mix_post_pca = post_contour_pca
                    mix_true_pca = true_pca
                    mix_lim_pca_x, mix_lim_pca_y = lim_pca_x, lim_pca_y
                    if post_cached_transform is not None:
                        post_mean_np, post_basis_np = post_cached_transform
                        mix_prior_pca = _project_pca_2d(prior_pca_contour_src, post_mean_np, post_basis_np)
                        mix_post_pca = _project_pca_2d(post_pca_contour_src, post_mean_np, post_basis_np)
                        mix_true_pca = post_true_pca_custom
                        if range_tag == "fixed" and post_fixed_pca_custom is not None:
                            mix_lim_pca_x, mix_lim_pca_y = post_fixed_pca_custom
                        elif range_tag == "adaptive" and post_adapt_pca_custom is not None:
                            mix_lim_pca_x, mix_lim_pca_y = post_adapt_pca_custom
                    _draw_density_contours_2d(
                        ax=ax_mix_pca,
                        points_xy=mix_prior_pca,
                        xlim=mix_lim_pca_x,
                        ylim=mix_lim_pca_y,
                        point_color="blue",
                        grid_bins=contour_grid_bins,
                        alpha_scale=mixed_alpha_scale,
                    )
                    _draw_density_contours_2d(
                        ax=ax_mix_pca,
                        points_xy=mix_post_pca,
                        xlim=mix_lim_pca_x,
                        ylim=mix_lim_pca_y,
                        point_color="red",
                        grid_bins=contour_grid_bins,
                        alpha_scale=mixed_alpha_scale,
                    )
                    if mix_true_pca is not None and np.all(np.isfinite(mix_true_pca)):
                        ax_mix_pca.scatter(
                            mix_true_pca[0], mix_true_pca[1],
                            marker='x', s=120, c='black', linewidths=2.0, zorder=12, label='True state'
                        )
                    ax_mix_pca.set_xlim(mix_lim_pca_x)
                    ax_mix_pca.set_ylim(mix_lim_pca_y)
                    if legend_in_figure:
                        ax_mix_pca.set_xlabel("PC1")
                        ax_mix_pca.set_ylabel("PC2")
                        ax_mix_pca.set_title(f"pca12 ({range_tag}, mixed contour)", fontsize=10)
                        ax_mix_pca.legend(
                            [
                                Line2D([0], [0], color='blue', linewidth=2.0),
                                Line2D([0], [0], color='red', linewidth=2.0),
                                Line2D([0], [0], marker='x', linestyle='None', color='black', markeredgecolor='black', markersize=10),
                            ],
                            ["Prior contour", "Posterior contour", "True state"],
                            loc='best',
                            fontsize=8,
                            frameon=True,
                        )
                    else:
                        ax_mix_pca.set_xlabel("")
                        ax_mix_pca.set_ylabel("")
                        ax_mix_pca.set_title("")
                    fig_mix_pca.tight_layout()
                    fig_mix_pca.savefig(
                        f"{prefix}_{i}_mixed_{range_tag}_PCA12_contour.png",
                        bbox_inches='tight',
                        dpi=150,
                    )
                    plt.close(fig_mix_pca)

            # 4) fixed-range 3D view (fixed angle, no adaptive range)
            fig3d = plt.figure(figsize=(8.0, 8.0))
            ax3d = fig3d.add_subplot(111, projection='3d')
            ax3d.scatter(
                mode_points_scatter[:, 0],
                mode_points_scatter[:, 1],
                mode_points_scatter[:, 2],
                s=float(size_3d),
                alpha=float(min(0.98, alpha_3d)),
                c=mode_color,
                edgecolors='none',
                label=mode_label,
            )
            if true_xyz is not None and len(true_xyz) >= 3 and np.all(np.isfinite(true_xyz[:3])):
                ax3d.scatter(
                    true_xyz[0], true_xyz[1], true_xyz[2],
                    marker='x', s=130, c='black', linewidths=2.0, zorder=12, label='True state',
                )
            ax3d.set_xlim(fixed_limits["xlim"])
            ax3d.set_ylim(fixed_limits["ylim"])
            ax3d.set_zlim(fixed_limits["zlim"])
            ax3d.view_init(
                elev=float(getattr(args, "pf_3d_elev", 25.0)),
                azim=float(getattr(args, "pf_3d_azim", -55.0)),
            )
            if legend_in_figure:
                ax3d.set_xlabel("x")
                ax3d.set_ylabel("y")
                ax3d.set_zlabel("z")
                title_3d = f"{dataset} {mode_tag} 3D fixed"
                if compute_swd:
                    title_3d += f" | SWD ratio={mode_ratio}"
                ax3d.set_title(title_3d, fontsize=11)
                ax3d.legend(loc='best', fontsize=8, frameon=True)
            else:
                ax3d.set_title("")
                ax3d.set_xlabel("")
                ax3d.set_ylabel("")
                ax3d.set_zlabel("")
            fig3d.savefig(f"{prefix}_{i}_{mode_tag}_3d_fixed.png", bbox_inches='tight', dpi=150)
            plt.close(fig3d)

            # 4) quantile-inversion density plots (6 files)
            q_probs_np = None
            if quantile_probs is not None:
                q_probs_np = torch.as_tensor(quantile_probs).detach().cpu().numpy().reshape(-1)
            mode_q_np = None if mode_quantiles is None else torch.as_tensor(mode_quantiles).detach().cpu().numpy()
            mode_pca_q_np = None if mode_pca_quantiles is None else torch.as_tensor(mode_pca_quantiles).detach().cpu().numpy()
            if q_probs_np is not None and mode_q_np is not None and mode_pca_q_np is not None:
                for d_idx in range(min(3, mode_q_np.shape[0])):
                    x, dens = _quantile_implied_density_curve(mode_q_np[d_idx], q_probs_np)
                    _save_quantile_density_plot(
                        x=x,
                        density=dens,
                        title=f"{dataset} {mode_tag} dim{d_idx+1} implied density",
                        save_path=f"{prefix}_{i}_{mode_tag}_qden_dim{d_idx+1}.png",
                    )
                for p_idx in range(min(3, mode_pca_q_np.shape[0])):
                    x, dens = _quantile_implied_density_curve(mode_pca_q_np[p_idx], q_probs_np)
                    _save_quantile_density_plot(
                        x=x,
                        density=dens,
                        title=f"{dataset} {mode_tag} pca{p_idx+1} implied density",
                        save_path=f"{prefix}_{i}_{mode_tag}_qden_PCA{p_idx+1}.png",
                    )

        if compute_swd:
            print(
                f"[PF Plot] cloud={i}, "
                f"prior: SWD(data,gauss)={swd_data_prior:.6f}, SWD(baseline)={swd_base_prior:.6f}, ratio={prior_ratio_str}; "
                f"post: SWD(data,gauss)={swd_data_post:.6f}, SWD(baseline)={swd_base_post:.6f}, ratio={post_ratio_str}"
            )
            distance_records.append(
                {
                    "cloud_index": int(i),
                    "prior_swd_ratio": float(swd_ratio_prior),
                    "prior_swd_data": float(swd_data_prior),
                    "prior_swd_baseline": float(swd_base_prior),
                    "post_swd_ratio": float(swd_ratio_post),
                    "post_swd_data": float(swd_data_post),
                    "post_swd_baseline": float(swd_base_post),
                }
            )

    if not legend_in_figure:
        legend_dir = os.path.dirname(prefix) or "."
        legend_prefix_prior = os.path.join(legend_dir, f"{dataset}_prior")
        legend_prefix_post = os.path.join(legend_dir, f"{dataset}_post")
        legend_prefix_combined = os.path.join(legend_dir, f"{dataset}_prior_post_combined")
        _save_separate_legend(
            legend_prefix_prior,
            include_prior=True,
            include_posterior=False,
            include_obs=True,
            include_history=False,
            include_true_state=(true_state is not None),
        )
        _save_combined_prior_post_legend(
            legend_prefix_combined,
            include_obs=True,
            include_history=False,
            include_true_state=(true_state is not None),
            dpi=200,
        )
        _save_separate_legend(
            legend_prefix_post,
            include_prior=False,
            include_posterior=True,
            include_obs=True,
            include_history=False,
            include_true_state=(true_state is not None),
        )

    print(f"Processed {len(indices_to_process)} PF point-cloud pairs with prefix '{prefix}'.")
    return distance_records


def plot_pf_initial_distribution(
    args: Namespace,
    tensor: torch.Tensor,
    num_samples_plot: int,
    prefix: str,
    true_state: Optional[torch.Tensor] = None,
    plot_indices: Optional[List[int]] = None,
    legend_in_figure: bool = True,
    plot_cdf: bool = False,
) -> None:
    """
    Save PF initialization figures using the plotting style appropriate for the model.

    - dim >= 3 and supported 3D datasets: PF-style 2D/PCA/3D views for one cloud.
    - dim in {1, 2}: ring-mapped initialization plot, optionally with phase CDF/PDF.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if true_state is not None and isinstance(true_state, torch.Tensor) and true_state.is_cuda:
        true_state = true_state.cpu()

    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor shape (B, N, D), got {tuple(tensor.shape)}")

    dataset = str(getattr(args, "dataset", "")).lower()
    _, _, d_state = tensor.shape
    if d_state >= 3 and dataset in PF_3D_SUPPORTED_DATASETS:
        _plot_pf_initial_distribution_3d(
            args=args,
            tensor=tensor,
            num_samples_plot=num_samples_plot,
            prefix=prefix,
            true_state=true_state,
            plot_indices=plot_indices,
            legend_in_figure=legend_in_figure,
        )
        return

    plot_and_test_point_clouds_ring(
        args=args,
        tensor=tensor,
        num_samples_plot=num_samples_plot,
        prefix=prefix,
        point_color="blue",
        observation=None,
        true_state=true_state,
        plot_history=False,
        plot_indices=plot_indices,
        history_traj=None,
        plot_cdf=plot_cdf,
        legend_in_figure=legend_in_figure,
        standalone_legend_name="init_density",
        save_combined_prior_post_legend=False,
    )


def _plot_pf_initial_distribution_3d(
    args: Namespace,
    tensor: torch.Tensor,
    num_samples_plot: int,
    prefix: str,
    true_state: Optional[torch.Tensor] = None,
    plot_indices: Optional[List[int]] = None,
    legend_in_figure: bool = True,
) -> None:
    """Save PF-style initialization views for one 3D-supported cloud."""
    dataset = str(getattr(args, "dataset", "")).lower()
    if dataset not in PF_3D_SUPPORTED_DATASETS:
        raise ValueError(
            f"PF 3D initialization plotter supports only {sorted(PF_3D_SUPPORTED_DATASETS)}, got dataset='{dataset}'."
        )

    if tensor.is_cuda:
        tensor = tensor.cpu()
    if true_state is not None and isinstance(true_state, torch.Tensor) and true_state.is_cuda:
        true_state = true_state.cpu()

    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor shape (B, N, D), got {tuple(tensor.shape)}")

    batch_size, _, state_dim = tensor.shape
    if state_dim < 3:
        raise ValueError(f"PF 3D initialization plotter expects D >= 3, got D={state_dim}.")

    if plot_indices is None:
        indices_to_process = list(range(batch_size))
    else:
        indices_to_process = [idx for idx in plot_indices if 0 <= idx < batch_size]

    if len(indices_to_process) == 0:
        return

    contour_grid_bins = int(max(16, int(getattr(args, "pf_contour_grid_bins", 200))))
    fixed_limits = PF_FIXED_RANGES_3D[dataset]
    projection_specs = [
        ("xy", 0, 1, "x", "y"),
        ("yz", 1, 2, "y", "z"),
        ("xz", 0, 2, "x", "z"),
    ]

    for i in indices_to_process:
        full_points = tensor[i, :, :]
        display_points = full_points[:, :3]
        true_xyz = _resolve_true_state_xyz(true_state=true_state, batch_idx=i, batch_size=batch_size)
        adaptive_limits = _compute_zoomed_ranges_from_fixed_3d(
            points_xyz=display_points.numpy(),
            fixed_limits=fixed_limits,
            num_splits=10,
        )

        points_for_vis = _sample_points(full_points, num_samples_plot)
        scatter_ds, scatter_downsampled = _prepare_scatter_points(
            points_for_vis,
            max_points=PF_MAX_SCATTER_POINTS,
        )
        contour_ds, _ = _prepare_scatter_points(
            points_for_vis,
            max_points=PF_MAX_CONTOUR_POINTS,
        )

        scatter_xyz = scatter_ds[:, :3].numpy()
        contour_xyz = contour_ds[:, :3].numpy()
        marker_size_2d, marker_alpha_2d = _adaptive_scatter_style(
            n_points=scatter_xyz.shape[0],
            is_3d=False,
            downsampled=scatter_downsampled,
        )
        marker_size_3d, marker_alpha_3d = _adaptive_scatter_style(
            n_points=scatter_xyz.shape[0],
            is_3d=True,
            downsampled=scatter_downsampled,
        )

        pca_mean, pca_basis = _fit_pca_2d(full_points.numpy())
        scatter_pca = _project_pca_2d(scatter_ds.numpy(), pca_mean, pca_basis)
        contour_pca = _project_pca_2d(contour_ds.numpy(), pca_mean, pca_basis)
        true_pca = None
        if true_state is not None:
            ts = torch.as_tensor(true_state).detach().cpu()
            true_full = None
            if ts.ndim == 1 and ts.numel() == state_dim:
                true_full = ts.numpy().astype(np.float64)
            elif ts.ndim == 2:
                if ts.shape[0] == batch_size and ts.shape[1] == state_dim:
                    true_full = ts[i].numpy().astype(np.float64)
                elif ts.shape[0] > 0 and ts.shape[1] == state_dim:
                    true_full = ts[0].numpy().astype(np.float64)
            if true_full is not None:
                true_pca = _project_pca_2d(true_full.reshape(1, -1), pca_mean, pca_basis)[0]

        fixed_pca_xlim, fixed_pca_ylim = _pca_limits_from_xyz_ranges(
            xyz_limits=fixed_limits,
            mean=pca_mean[:3],
            basis=pca_basis[:3, :2],
            pad=0,
        )
        adapt_pca_xlim, adapt_pca_ylim = _pca_limits_from_xyz_ranges(
            xyz_limits=adaptive_limits,
            mean=pca_mean[:3],
            basis=pca_basis[:3, :2],
            pad=0,
        )

        range_modes = [
            ("fixed", fixed_limits, fixed_pca_xlim, fixed_pca_ylim),
            ("adaptive", adaptive_limits, adapt_pca_xlim, adapt_pca_ylim),
        ]
        for range_tag, lim_xyz, lim_pca_x, lim_pca_y in range_modes:
            for plane_tag, dim_x, dim_y, x_label, y_label in projection_specs:
                if plane_tag == "xy":
                    xlim_curr, ylim_curr = lim_xyz["xlim"], lim_xyz["ylim"]
                elif plane_tag == "yz":
                    xlim_curr, ylim_curr = lim_xyz["ylim"], lim_xyz["zlim"]
                else:
                    xlim_curr, ylim_curr = lim_xyz["xlim"], lim_xyz["zlim"]
                view_name = "12" if plane_tag == "xy" else ("23" if plane_tag == "yz" else "13")
                for plot_style in ["scatter", "contour"]:
                    fig, ax = plt.subplots(figsize=(7.2, 6.2))
                    _plot_pf_projection_2d(
                        ax=ax,
                        points=scatter_xyz if plot_style == "scatter" else contour_xyz,
                        point_color="blue",
                        point_label="Initialization distribution",
                        dim_x=dim_x,
                        dim_y=dim_y,
                        marker_size=marker_size_2d,
                        marker_alpha=marker_alpha_2d,
                        true_xyz=true_xyz,
                        xlim=xlim_curr,
                        ylim=ylim_curr,
                        legend_in_figure=False,
                        dataset=dataset,
                        contour_grid_bins=contour_grid_bins,
                        draw_scatter=(plot_style == "scatter"),
                        draw_contour=(plot_style == "contour"),
                    )
                    if legend_in_figure:
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.set_title(f"{dataset} initialization {plane_tag} ({range_tag}, {plot_style})", fontsize=10)
                        handles = []
                        labels = []
                        if plot_style == "scatter":
                            handles.append(
                                Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='blue', markeredgecolor='none', markersize=7)
                            )
                            labels.append("Initialization distribution")
                        else:
                            handles.append(mpatches.Patch(facecolor='blue', alpha=0.68, edgecolor='none'))
                            labels.append("Initialization contour")
                        if true_xyz is not None and np.all(np.isfinite(true_xyz[:3])):
                            handles.append(
                                Line2D([0], [0], marker='x', linestyle='None', color='black', markeredgecolor='black', markersize=10)
                            )
                            labels.append("True state")
                        ax.legend(handles, labels, loc='best', fontsize=8, frameon=True)
                    else:
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        ax.set_title("")
                    fig.tight_layout()
                    save_path = f"{prefix}_{i}_init_{range_tag}_{view_name}.png"
                    if plot_style == "contour":
                        save_path = f"{prefix}_{i}_init_{range_tag}_{view_name}_contour.png"
                    fig.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.close(fig)

            for plot_style in ["scatter", "contour"]:
                fig_pca, ax_pca = plt.subplots(figsize=(7.2, 6.2))
                if plot_style == "contour":
                    _draw_density_contours_2d(
                        ax=ax_pca,
                        points_xy=contour_pca,
                        xlim=lim_pca_x,
                        ylim=lim_pca_y,
                        point_color="blue",
                        grid_bins=contour_grid_bins,
                    )
                else:
                    ax_pca.scatter(
                        scatter_pca[:, 0],
                        scatter_pca[:, 1],
                        s=float(marker_size_2d),
                        alpha=float(min(0.98, marker_alpha_2d)),
                        c='blue',
                        edgecolors='none',
                        zorder=4,
                        label='Initialization distribution',
                    )
                if true_pca is not None and np.all(np.isfinite(true_pca)):
                    ax_pca.scatter(
                        true_pca[0], true_pca[1],
                        marker='x', s=120, c='black', linewidths=2.0, zorder=12, label='True state'
                    )
                ax_pca.set_xlim(lim_pca_x)
                ax_pca.set_ylim(lim_pca_y)
                if legend_in_figure:
                    ax_pca.set_xlabel("PC1")
                    ax_pca.set_ylabel("PC2")
                    ax_pca.set_title(f"{dataset} initialization pca12 ({range_tag}, {plot_style})", fontsize=10)
                    handles = []
                    labels = []
                    if plot_style == "scatter":
                        handles.append(
                            Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='blue', markeredgecolor='none', markersize=7)
                        )
                        labels.append("Initialization distribution")
                    else:
                        handles.append(mpatches.Patch(facecolor='blue', alpha=0.68, edgecolor='none'))
                        labels.append("Initialization contour")
                    if true_pca is not None and np.all(np.isfinite(true_pca)):
                        handles.append(
                            Line2D([0], [0], marker='x', linestyle='None', color='black', markeredgecolor='black', markersize=10)
                        )
                        labels.append("True state")
                    ax_pca.legend(handles, labels, loc='best', fontsize=8, frameon=True)
                else:
                    ax_pca.set_xlabel("")
                    ax_pca.set_ylabel("")
                    ax_pca.set_title("")
                fig_pca.tight_layout()
                save_path = f"{prefix}_{i}_init_{range_tag}_PCA12.png"
                if plot_style == "contour":
                    save_path = f"{prefix}_{i}_init_{range_tag}_PCA12_contour.png"
                fig_pca.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig_pca)

        fig3d = plt.figure(figsize=(8.0, 8.0))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.scatter(
            scatter_xyz[:, 0],
            scatter_xyz[:, 1],
            scatter_xyz[:, 2],
            s=float(marker_size_3d),
            alpha=float(min(0.98, marker_alpha_3d)),
            c='blue',
            edgecolors='none',
            label='Initialization distribution',
        )
        if true_xyz is not None and np.all(np.isfinite(true_xyz[:3])):
            ax3d.scatter(
                true_xyz[0], true_xyz[1], true_xyz[2],
                marker='x', s=130, c='black', linewidths=2.0, zorder=12, label='True state',
            )
        ax3d.set_xlim(fixed_limits["xlim"])
        ax3d.set_ylim(fixed_limits["ylim"])
        ax3d.set_zlim(fixed_limits["zlim"])
        ax3d.view_init(
            elev=float(getattr(args, "pf_3d_elev", 25.0)),
            azim=float(getattr(args, "pf_3d_azim", -55.0)),
        )
        if legend_in_figure:
            ax3d.set_xlabel("x")
            ax3d.set_ylabel("y")
            ax3d.set_zlabel("z")
            ax3d.set_title(f"{dataset} initialization 3D fixed", fontsize=11)
            ax3d.legend(loc='best', fontsize=8, frameon=True)
        else:
            ax3d.set_title("")
            ax3d.set_xlabel("")
            ax3d.set_ylabel("")
            ax3d.set_zlabel("")
        fig3d.savefig(f"{prefix}_{i}_init_3d_fixed.png", bbox_inches='tight', dpi=150)
        plt.close(fig3d)

    if not legend_in_figure:
        handles = [
            Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='blue', markeredgecolor='none', markersize=8),
        ]
        labels = ["Initialization distribution"]
        if true_state is not None:
            handles.append(
                Line2D([0], [0], marker='x', linestyle='None', color='black', markeredgecolor='black', markersize=10)
            )
            labels.append("True state")
        _save_horizontal_legend_image(
            prefix=f"{prefix}_init_legend",
            handles=handles,
            labels=labels,
            save_pdf=False,
            dpi=150,
            fontsize=11,
            frameon=True,
        )


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


def _map_state_to_ring_phase01(points: torch.Tensor) -> torch.Tensor:
    """
    Map state points (N, D) to normalized ring phase in [0, 1).

    - D == 1: theta = 2*pi*x, phase = mod(theta / 2*pi, 1) = mod(x, 1)
    - D >= 2: theta = atan2(y, x), phase = mod(theta / 2*pi, 1)
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

    phase01 = torch.remainder(theta / (2.0 * np.pi), 1.0)
    return phase01


def _adaptive_kde_std(n_samples: int, base_std: float = 0.01) -> float:
    """
    Adaptive KDE std on [0,1):
    - Keep 0.01 as the reference at n=500.
    - Use larger smoothing for smaller n.
    """
    n = max(int(n_samples), 1)
    scaled = base_std * np.sqrt(500.0 / float(n))
    return float(np.clip(scaled, 0.005, 0.08))


def _wrapped_kde_pdf_01(samples: np.ndarray, grid: np.ndarray, std: float) -> np.ndarray:
    """
    Wrapped Gaussian KDE on circular support [0,1).
    """
    if samples.size == 0:
        return np.zeros_like(grid)

    std = float(max(std, 1e-6))
    diff0 = (grid[:, None] - samples[None, :]) / std
    diffm = (grid[:, None] - (samples[None, :] - 1.0)) / std
    diffp = (grid[:, None] - (samples[None, :] + 1.0)) / std

    kernel = np.exp(-0.5 * diff0 * diff0) + np.exp(-0.5 * diffm * diffm) + np.exp(-0.5 * diffp * diffp)
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * std)
    density = norm * np.mean(kernel, axis=1)

    area = np.trapz(density, grid)
    if area > 0:
        density = density / area
    return density


def _smooth_histogram_pdf(pdf_vals: np.ndarray, bin_width: float, window_size: int = 9) -> np.ndarray:
    """
    Smooth histogram-based PDF on circular support [0,1) with wrapped moving average,
    then re-normalize area to 1.
    """
    if pdf_vals.size == 0:
        return pdf_vals

    win = int(max(1, window_size))
    if win % 2 == 0:
        win += 1
    if win == 1:
        return pdf_vals

    kernel = np.ones(win, dtype=np.float64) / float(win)
    half = win // 2
    padded = np.concatenate([pdf_vals[-half:], pdf_vals, pdf_vals[:half]])
    smoothed = np.convolve(padded, kernel, mode='valid')
    smoothed = smoothed[:pdf_vals.size]

    area = float(np.sum(smoothed) * max(float(bin_width), 1e-12))
    if area > 0:
        smoothed = smoothed / area
    return smoothed


def _estimate_phase_pdf_curve_01(phase01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate PDF curve on circular support [0, 1).
    """
    n_phase = int(phase01.size)
    if n_phase < 500:
        phase_grid = np.linspace(0.0, 1.0, 1000)
        kde_std = _adaptive_kde_std(n_phase, base_std=0.01)
        pdf_vals = _wrapped_kde_pdf_01(phase01, phase_grid, kde_std)
    else:
        n_bins = int(np.round(10.0 * float(n_phase) / 500.0))
        n_bins = int(np.clip(n_bins, 10, 2000))
        hist, edges = np.histogram(phase01, bins=n_bins, range=(0.0, 1.0), density=True)
        phase_grid = 0.5 * (edges[:-1] + edges[1:])
        pdf_vals = _smooth_histogram_pdf(
            hist,
            bin_width=(edges[1] - edges[0]) if edges.size >= 2 else 1.0,
            window_size=9,
        )

    pdf_vals = np.nan_to_num(pdf_vals, nan=0.0, posinf=0.0, neginf=0.0)
    return phase_grid, pdf_vals


def _draw_ring_pdf_fill(
    ax: plt.Axes,
    phase_grid: np.ndarray,
    pdf_vals: np.ndarray,
    fill_color: str,
    alpha: float = 0.22,
    label: str = "Phase density",
) -> float:
    """
    Draw ring PDF as a filled band between unit circle and outer PDF radius.
    Returns the max outer radius.
    """
    if phase_grid.size == 0 or pdf_vals.size == 0:
        return 1.0

    theta_curve = 2.0 * np.pi * np.mod(phase_grid, 1.0)
    pdf_safe = np.nan_to_num(pdf_vals, nan=0.0, posinf=0.0, neginf=0.0)
    pdf_safe = np.maximum(pdf_safe, 0.0)
    radius_outer = np.sqrt(1.0 + pdf_safe / np.pi)

    theta_wrap = np.concatenate([theta_curve, np.array([theta_curve[0] + 2.0 * np.pi])])
    outer_wrap = np.concatenate([radius_outer, np.array([radius_outer[0]])])
    x_outer = outer_wrap * np.cos(theta_wrap)
    y_outer = outer_wrap * np.sin(theta_wrap)
    x_inner = np.cos(theta_wrap)
    y_inner = np.sin(theta_wrap)

    poly_x = np.concatenate([x_outer, x_inner[::-1]])
    poly_y = np.concatenate([y_outer, y_inner[::-1]])
    ax.fill(poly_x, poly_y, color=fill_color, alpha=alpha, edgecolor='none', linewidth=0.0, label=label)

    return float(np.max(radius_outer))


def _save_ring_pdf_on_ring(
    args: Namespace,
    prefix: str,
    cloud_idx: int,
    point_color: str,
    phase01: np.ndarray,
    phase_grid: np.ndarray,
    pdf_vals: np.ndarray,
    plot_history: bool,
    history_traj: Optional[torch.Tensor],
    obs_x: Optional[float],
    true_xy: Optional[np.ndarray],
    legend_in_figure: bool = True,
) -> List[str]:
    """
    Save phase PDF mapped onto a ring (angle in [0, 2*pi], positive density outward).
    """
    n_phase = int(phase01.size)
    fig, ax = plt.subplots(figsize=(7, 7))
    legend_labels_used: List[str] = []
    point_color_norm = str(point_color).lower().strip()
    if point_color_norm == "blue":
        density_label = "predictive density"
    elif point_color_norm == "red":
        density_label = "filtering density"
    else:
        density_label = "phase density"

    def _add_label(label: str) -> None:
        if label not in legend_labels_used:
            legend_labels_used.append(label)

    phi = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(phi), np.sin(phi), color='gray', linewidth=1.0, alpha=0.8, label='Unit circle')
    _add_label('Unit circle')

    _draw_ring_pdf_fill(
        ax=ax,
        phase_grid=phase_grid,
        pdf_vals=pdf_vals,
        fill_color=point_color,
        alpha=0.22,
        label=density_label,
    )
    _add_label(density_label)

    history_used = False
    if n_phase < 500 and phase01.size > 0:
        theta_samples = 2.0 * np.pi * np.mod(phase01, 1.0)
        ax.scatter(
            np.cos(theta_samples),
            np.sin(theta_samples),
            s=50,
            alpha=0.55,
            c=point_color,
            edgecolors='none',
            label='Samples',
        )
        _add_label('Samples')

        if plot_history and history_traj is not None:
            traj = history_traj[:, cloud_idx, :]
            traj_xy = _map_state_to_ring_xy(traj).detach().cpu().numpy()
            ax.plot(
                traj_xy[:, 0],
                traj_xy[:, 1],
                color='black',
                linewidth=1.5,
                label='History trajectory',
            )
            history_used = True
            _add_label('History trajectory')

    if obs_x is not None:
        obs_x_clip = float(np.clip(obs_x, -1.2, 1.2))
        ax.axvline(obs_x_clip, color='orange', linestyle='--', linewidth=2.0, alpha=0.7, label='Observation')
        _add_label('Observation')

    if true_xy is not None:
        ax.scatter(
            true_xy[0],
            true_xy[1],
            marker='x',
            s=120,
            c='black',
            linewidths=2.0,
            zorder=11,
            label='True state',
        )
        _add_label('True state')

    mode_tag = "hist" if history_used else "nohist"
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    if legend_in_figure:
        ax.set_xlabel("ring-x")
        ax.set_ylabel("ring-y")
        ax.set_title(f"{args.dataset} phase PDF on ring, cloud {cloud_idx} ({mode_tag})")
        ax.legend(loc='upper right', fontsize=9, frameon=True)

    fig.savefig(f"{prefix}_{cloud_idx}_ring_pdf_on_ring_{mode_tag}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    return legend_labels_used


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
    plot_cdf: bool = True,
    legend_in_figure: bool = True,
    standalone_legend_name: Optional[str] = None,
    save_combined_prior_post_legend: bool = True,
):
    """
    Visualize 1D/2D point clouds on a unit circle using 2D density estimation.

    - For 1D: state x is mapped by theta=2*pi*x.
    - For 2D: state (u,v) is mapped by theta=atan2(v,u).
    - Observation is interpreted as circle x-coordinate and shown as a vertical line.
    - True state is mapped to ring and plotted with a black X marker if provided.
    - History trajectory is optional and controlled by `plot_history`.
    - If plotted ensemble size < 1000, draw all points directly (scatter).
      Otherwise use 2D hexbin density.
    - Optionally saves an empirical CDF of phase=angle/(2*pi) projected to [0, 1)
      when `plot_cdf=True`. The CDF plot intentionally excludes
      observation/history/true-state overlays.
    - If `plot_cdf=True`, also saves a PDF on [0,1):
      * N < 500: wrapped KDE with adaptive std (base 0.01 at N=500).
      * N >= 500: histogram-based PDF with adaptive bins:
        bins = min(2000, round(10 * N / 500)), so N=500 -> 10.
        Then applies moving-average smoothing.
      * For all N, also saves a "PDF on ring" view:
        map [0,1) -> [0,2*pi], and draw positive density outward.
        History is shown only when N < 500 and `plot_history=True`.
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
    if point_color == "blue":
        ensemble_label = "predictive ensemble"
        density_label = "predictive density"
    elif point_color == "red":
        ensemble_label = "filtering ensemble"
        density_label = "filtering density"
    else:
        ensemble_label = "ensemble"
        density_label = "phase density"

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

    legend_labels_used: List[str] = []

    def _add_label(label: str) -> None:
        if label not in legend_labels_used:
            legend_labels_used.append(label)

    for i in indices_to_process:
        full_points = tensor[i, :, :]  # (N, D)
        phase01 = np.array([])
        phase_grid = np.array([])
        pdf_vals = np.array([])
        if plot_cdf:
            phase01 = _map_state_to_ring_phase01(full_points).detach().cpu().numpy()
            phase01 = phase01[np.isfinite(phase01)]
            if phase01.size > 0:
                phase_grid, pdf_vals = _estimate_phase_pdf_curve_01(phase01)

        true_xy = None
        if true_state is not None:
            if isinstance(true_state, torch.Tensor):
                ts = true_state.detach().cpu()
            else:
                ts = torch.as_tensor(true_state)

            if ts.ndim == 1:
                if ts.numel() >= D:
                    true_xy = _map_state_to_ring_xy(ts[:D].reshape(1, D)).numpy()[0]
            elif ts.ndim == 2:
                if ts.shape[0] == B and ts.shape[1] >= D:
                    true_xy = _map_state_to_ring_xy(ts[i, :D].reshape(1, D)).numpy()[0]
                elif ts.shape[1] >= D:
                    true_xy = _map_state_to_ring_xy(ts[0, :D].reshape(1, D)).numpy()[0]

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
        _add_label('Unit circle')

        if xy_plot.shape[0] < density_threshold:
            # Small ensembles: plot all points directly.
            ax.scatter(
                xy_plot[:, 0], xy_plot[:, 1],
                s=50, alpha=0.55, c=point_color, edgecolors='none', label=ensemble_label
            )
            _add_label(ensemble_label)
        else:
            # Large ensembles: density view is clearer.
            cmap = 'Reds' if point_color == 'red' else 'Blues'
            hb = ax.hexbin(
                xy_plot[:, 0], xy_plot[:, 1],
                gridsize=60,
                extent=(-2.0, 2.0, -2.0, 2.0),
                mincnt=1,
                bins='log',
                cmap=cmap,
                linewidths=0.0,
                alpha=0.95,
            )
            cbar = fig.colorbar(hb, ax=ax, shrink=0.82, pad=0.02)
            if legend_in_figure:
                cbar.set_label("log10(count)")
            _add_label('Ensemble density')

            # Optional sparse overlay to show geometry without overplotting.
            n_overlay = min(1500, xy_plot.shape[0])
            if n_overlay > 0:
                idx_overlay = np.random.choice(xy_plot.shape[0], size=n_overlay, replace=False)
                ax.scatter(
                    xy_plot[idx_overlay, 0], xy_plot[idx_overlay, 1],
                    s=2, alpha=0.12, c=point_color, edgecolors='none'
                )

        # Overlay phase PDF as a filled ring band (for both scatter and density modes).
        if phase01.size > 0 and phase_grid.size > 0 and pdf_vals.size > 0:
            _draw_ring_pdf_fill(
                ax=ax,
                phase_grid=phase_grid,
                pdf_vals=pdf_vals,
                fill_color=point_color,
                alpha=0.22,
                label=density_label,
            )
            _add_label(density_label)

        # History trajectory (mapped to ring)
        if plot_history and history_traj is not None:
            traj = history_traj[:, i, :]  # (T, D)
            traj_xy = _map_state_to_ring_xy(traj).numpy()
            ax.plot(traj_xy[:, 0], traj_xy[:, 1], color='black', linewidth=1.5, label='History trajectory')
            _add_label('History trajectory')

        # Observation as vertical line on the ring-x axis.
        if obs_x is not None:
            obs_x_clip = float(np.clip(obs_x, -1.2, 1.2))
            ax.axvline(obs_x_clip, color='orange', linestyle='--', linewidth=2.0, alpha=0.7, label='Observation')
            _add_label('Observation')

        # True state mapped to ring (black X).
        if true_xy is not None:
            ax.scatter(
                true_xy[0], true_xy[1],
                marker='x', s=120, c='black', linewidths=2.0, zorder=11, label='True state'
            )
            _add_label('True state')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        if legend_in_figure:
            ax.set_xlabel("ring-x")
            ax.set_ylabel("ring-y")
            mode_str = "scatter" if xy_plot.shape[0] < density_threshold else "hexbin density"
            ax.set_title(f"{args.dataset} (ring map, {mode_str}), cloud {i}")
            ax.legend(loc='upper right', fontsize=9, frameon=True)

        fig.savefig(f"{prefix}_{i}_ring.png", bbox_inches='tight', dpi=150)
        plt.close(fig)

        if plot_cdf:
            # Empirical CDF on normalized phase in [0, 1), using the full cloud.
            if phase01.size > 0:
                phase_sorted = np.sort(phase01)
                cdf = np.arange(1, phase_sorted.size + 1, dtype=np.float64) / float(phase_sorted.size)
                n_phase = int(phase_sorted.size)

                fig_cdf, ax_cdf = plt.subplots(figsize=(7, 4.2))
                ax_cdf.step(
                    phase_sorted,
                    cdf,
                    where='post',
                    color=point_color,
                    linewidth=1.6,
                    label='Empirical CDF',
                )
                _add_label('Empirical CDF')
                if n_phase < 500:
                    ax_cdf.scatter(
                        phase_sorted,
                        np.zeros_like(phase_sorted),
                        s=50,
                        alpha=0.55,
                        c=point_color,
                        edgecolors='none',
                        label='Samples',
                    )
                    _add_label('Samples')
                    if legend_in_figure:
                        ax_cdf.legend(loc='lower right', fontsize=8, frameon=True)
                ax_cdf.set_xlim(0.0, 1.0)
                ax_cdf.set_ylim(0.0, 1.0)
                if legend_in_figure:
                    ax_cdf.set_xlabel("phase = angle / (2*pi)")
                    ax_cdf.set_ylabel("empirical CDF")
                    ax_cdf.set_title(f"{args.dataset} empirical CDF, cloud {i}")
                ax_cdf.grid(True, linestyle='--', alpha=0.35)
                fig_cdf.tight_layout()
                fig_cdf.savefig(f"{prefix}_{i}_ring_cdf.png", bbox_inches='tight', dpi=150)
                plt.close(fig_cdf)

                # PDF on [0, 1): KDE for small N, histogram PDF for large N.
                phase_grid, pdf_vals = _estimate_phase_pdf_curve_01(phase01)
                fig_pdf, ax_pdf = plt.subplots(figsize=(7, 4.2))
                if n_phase < 500:
                    ax_pdf.plot(phase_grid, pdf_vals, color=point_color, linewidth=1.8, label='Phase PDF')
                    _add_label('Phase PDF')
                    ax_pdf.scatter(
                        np.mod(phase01, 1.0),
                        np.zeros_like(phase01),
                        s=50,
                        alpha=0.55,
                        c=point_color,
                        edgecolors='none',
                        label='Samples',
                    )
                    _add_label('Samples')
                    if legend_in_figure:
                        ax_pdf.legend(loc='upper right', fontsize=8, frameon=True)
                        ax_pdf.set_title(f"{args.dataset} phase PDF, cloud {i}")
                else:
                    ax_pdf.plot(phase_grid, pdf_vals, color=point_color, linewidth=1.6, label='Phase PDF')
                    _add_label('Phase PDF')
                    if legend_in_figure:
                        ax_pdf.set_title(f"{args.dataset} phase PDF, cloud {i}")

                ax_pdf.set_xlim(0.0, 1.0)
                if legend_in_figure:
                    ax_pdf.set_xlabel("phase = angle / (2*pi)")
                    ax_pdf.set_ylabel("PDF")
                ax_pdf.grid(True, linestyle='--', alpha=0.35)
                fig_pdf.tight_layout()
                fig_pdf.savefig(f"{prefix}_{i}_ring_pdf.png", bbox_inches='tight', dpi=150)
                plt.close(fig_pdf)

                ring_pdf_labels = _save_ring_pdf_on_ring(
                    args=args,
                    prefix=prefix,
                    cloud_idx=i,
                    point_color=point_color,
                    phase01=phase01,
                    phase_grid=phase_grid,
                    pdf_vals=pdf_vals,
                    plot_history=plot_history,
                    history_traj=history_traj,
                    obs_x=obs_x,
                    true_xy=true_xy,
                    legend_in_figure=legend_in_figure,
                )
                for lbl in ring_pdf_labels:
                    _add_label(lbl)

    if not legend_in_figure:
        density_color = 'red' if point_color == 'red' else 'blue'
        legend_handles: List = []
        legend_labels: List[str] = []

        show_ensemble = (
            ensemble_label in legend_labels_used
            or 'Ensemble density' in legend_labels_used
            or density_label in legend_labels_used
        )
        if show_ensemble:
            legend_handles.append(
                Line2D([0], [0], marker='o', linestyle='None',
                       markerfacecolor=point_color, markeredgecolor='none', markersize=8)
            )
            legend_labels.append(ensemble_label)
        if density_label in legend_labels_used:
            legend_handles.append(
                mpatches.Patch(facecolor=density_color, alpha=0.22, edgecolor='none')
            )
            legend_labels.append(density_label)
        if 'History trajectory' in legend_labels_used:
            legend_handles.append(
                Line2D([0], [0], color='black', linewidth=1.5)
            )
            legend_labels.append('History trajectory')
        if 'Observation' in legend_labels_used:
            legend_handles.append(
                Line2D([0], [0], color='orange', linestyle='--', linewidth=2.0)
            )
            legend_labels.append('Observation')
        if 'True state' in legend_labels_used:
            legend_handles.append(
                Line2D([0], [0], marker='x', linestyle='None',
                       color='black', markeredgecolor='black', markersize=10)
            )
            legend_labels.append('True state')

        legend_dir = os.path.dirname(prefix) or "."
        legend_name = standalone_legend_name
        if not legend_name:
            legend_name = "post_density" if point_color == "red" else "prior_density"
        legend_prefix = os.path.join(legend_dir, legend_name)

        if len(legend_handles) > 0:
            _save_horizontal_legend_image(
                prefix=legend_prefix,
                handles=legend_handles,
                labels=legend_labels,
                save_pdf=False,
                dpi=150,
                fontsize=11,
                frameon=True,
            )
        if save_combined_prior_post_legend:
            combined_legend_prefix = os.path.join(legend_dir, "prior_post_density_combined")
            combined_top_handles: List = [
                Line2D([0], [0], marker='o', linestyle='None',
                       markerfacecolor='blue', markeredgecolor='none', markersize=8, label='predictive ensemble'),
                Line2D([0], [0], marker='o', linestyle='None',
                       markerfacecolor='red', markeredgecolor='white', markersize=8, label='filtering ensemble'),
                Line2D([0], [0], marker='x', linestyle='None',
                       color='black', markeredgecolor='black', markersize=10, label='True state'),
                Line2D([0], [0], color='orange', linestyle='--', linewidth=2.0, label='Observation'),
            ]
            combined_bottom_handles: List = [
                mpatches.Patch(facecolor='blue', alpha=0.22, edgecolor='none', label='predictive density'),
                mpatches.Patch(facecolor='red', alpha=0.22, edgecolor='none', label='filtering density'),
            ]
            _save_two_row_legend_image(
                prefix=combined_legend_prefix,
                top_handles=combined_top_handles,
                top_labels=[h.get_label() for h in combined_top_handles],
                bottom_handles=combined_bottom_handles,
                bottom_labels=[h.get_label() for h in combined_bottom_handles],
                save_pdf=False,
                dpi=150,
                fontsize=11,
                frameon=True,
            )

    print(f"Processed {len(indices_to_process)} ring-mapped point clouds with prefix '{prefix}'.")
