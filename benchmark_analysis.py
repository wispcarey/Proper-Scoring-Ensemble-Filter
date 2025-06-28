import torch
import math
import time # For timing analysis steps
from localization import pairwise_distances, dist2coeff
# import matplotlib.pyplot as plt # Uncomment for plotting GC test or RMSEs

# ##############################################################################
# # Utility Functions
# ##############################################################################

# def dist2coeff(dists, radius, tag=None):
#     coeffs = torch.zeros(dists.shape, device=dists.device)

#     if tag is None:
#         tag = "GC"

#     if tag == "Gauss":
#         R = radius
#         coeffs = torch.exp(-0.5 * (dists / R) ** 2)
#     elif tag == "Exp":
#         R = radius
#         coeffs = torch.exp(-0.5 * (dists / R) ** 3)
#     elif tag == "Cubic":
#         R = radius * 1.87
#         inds = dists <= R
#         coeffs[inds] = (1 - (dists[inds] / R) ** 3) ** 3
#     elif tag == "Quadro":
#         R = radius * 1.64
#         inds = dists <= R
#         coeffs[inds] = (1 - (dists[inds] / R) ** 4) ** 4
#     elif tag == "GC":
#         R = radius * 1.82
#         ind1 = dists <= R
#         r2 = (dists[ind1] / R) ** 2
#         r3 = (dists[ind1] / R) ** 3
#         coeffs[ind1] = 1 + r2 * (-r3 / 4 + r2 / 2) + r3 * (5 / 8) - r2 * (5 / 3)
#         ind2 = torch.logical_and(R < dists, dists <= 2 * R)
#         r1 = dists[ind2] / R
#         r2 = (dists[ind2] / R) ** 2
#         r3 = (dists[ind2] / R) ** 3
#         coeffs[ind2] = (
#             r2 * (r3 / 12 - r2 / 2)
#             + r3 * (5 / 8)
#             + r2 * (5 / 3)
#             - r1 * 5
#             + 4
#             - (2 / 3) / r1
#         )
#     elif tag == "Step":
#         R = radius
#         inds = dists <= R
#         coeffs[inds] = 1
#     else:
#         raise KeyError("No such coeff function.")

#     return coeffs

def center_ensemble(E, rescale=False):
    """
    Centers the ensemble E along the second dimension (dim=1).

    Args:
        E (torch.Tensor): Ensemble tensor (batch_size x N_particles x d_state).
        rescale (bool): If True, rescale anomalies for unbiased covariance estimate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Centered anomalies, ensemble mean.
    """
    # Calculate mean along dim=1
    x = torch.mean(E, dim=1, keepdims=True)
    X_centered = E - x

    if rescale:
        # Get the size of the dimension along which mean was computed
        N = E.shape[1]
        if N > 1:
            # Rescale for unbiased covariance
            X_centered *= torch.sqrt(torch.tensor(N / (N - 1), device=E.device, dtype=E.dtype))

    return X_centered, x

# def pairwise_distances(X, Y=None, domain=None):
#     """
#     Computes pairwise distances for batches of points.
#     Handles periodic boundary conditions if domain are provided.

#     Args:
#         X (torch.Tensor): Tensor of shape (batch_size, N, D_coord).
#         Y (torch.Tensor, optional): Tensor of shape (batch_size, M, D_coord).
#                                   If None, Y = X (calculates pairwise distances within X).
#         domain (torch.Tensor, optional): Tensor of shape (D_coord,) for periodic boundaries.
#                                                  These lengths are applied across all batches.

#     Returns:
#         torch.Tensor: Tensor of shape (batch_size, N, M) with pairwise distances.
#     """
#     if Y is None:
#         Y = X

#     # X: (B, N, D_coord) -> X_unsqueezed: (B, N, 1, D_coord)
#     X_unsqueezed = X.unsqueeze(2)
#     # Y: (B, M, D_coord) -> Y_unsqueezed: (B, 1, M, D_coord)
#     Y_unsqueezed = Y.unsqueeze(1)

#     # diff: (B, N, M, D_coord) via broadcasting
#     diff = X_unsqueezed - Y_unsqueezed

#     if domain is not None:
#         # Reshape domain (D_coord,) to (1, 1, 1, D_coord) for broadcasting
#         domain_reshaped = domain.view(1, 1, 1, -1)
#         abs_diff = torch.abs(diff)
#         # For periodic domains, distance is min(|d|, L - |d|)
#         diff = torch.minimum(abs_diff, domain_reshaped - abs_diff)

#     # distances: (B, N, M) by summing squares along D_coord and taking sqrt
#     distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
#     return distances


def apply_inflation(ensemble, inflation_factor):
    """
    Applies multiplicative inflation to ensemble anomalies.

    Args:
        ensemble (torch.Tensor): Ensemble tensor (batch_size x N_particles x d_state).
        inflation_factor (float): Multiplicative inflation factor.

    Returns:
        torch.Tensor: Inflated ensemble (batch_size x N_particles x d_state).
    """
    if inflation_factor is None or inflation_factor == 1.0:
        return ensemble

    # center_ensemble now expects a 3D tensor and operates on dim=1
    anomalies, mean_ens = center_ensemble(ensemble, rescale=False)
    # anomalies: (B, N_particles, d_state), mean_ens: (B, 1, d_state)
    # inflation_factor is scalar, broadcasts with anomalies
    # mean_ens broadcasts correctly with (inflation_factor * anomalies)
    inflated_ensemble = mean_ens + inflation_factor * anomalies
    return inflated_ensemble

def matrix_sqrt_psd(A, tol=1e-9):
    """
    Compute the square root of symmetric positive semi-definite matrices.
    A = V S V^T. Returns V S^(1/2) V^T. Handles batched inputs.

    Args:
        A (torch.Tensor): Symmetric PSD matrix or batch of matrices (..., N, N).
        tol (float): Tolerance for eigenvalue clamping to ensure non-negativity.

    Returns:
        torch.Tensor: Matrix square root (..., N, N).
    """
    # torch.linalg.eigh handles batched inputs (e.g., batch_size x N x N)
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    # Clamp eigenvalues to be non-negative before sqrt
    eigenvalues_sqrt = torch.sqrt(torch.clamp(eigenvalues, min=tol))

    # Reconstruct the matrix square root
    # eigenvectors @ diag_matrix @ eigenvectors_transposed
    # This handles both batched (A.ndim > 2) and single (A.ndim = 2) cases correctly
    # due to how torch.diag_embed and batched matrix multiply (@) work.
    # .transpose(-2, -1) is robust for batched or non-batched.
    return eigenvectors @ torch.diag_embed(eigenvalues_sqrt) @ eigenvectors.transpose(-2, -1)



# ##############################################################################
# # Bootstrap Particle Filter (Analysis Step Only)
# ##############################################################################

from torch import vmap

def bootstrap_particle_filter_analysis(
    particles_forecast,      # (batch_size, N_particles, d_state)
    observation_y,           # (batch_size, d_obs) or (d_obs,)
    observation_operator,    # callable: (d_state,) -> (d_obs,)
    sigma_y,                 # float (std dev of observation noise)
    resampling_method="multinomial",
    sigma_reg=None,          # float (std dev for regularization noise)
    use_half_precision=False, # bool: If True, use float16 to save memory
    max_chunk_size=5000      # int: Max particles to process at once to save memory
):
    """
    Performs an advanced batch analysis (update & resampling) for a Bootstrap Particle Filter.

    This implementation includes:
    1. Vectorization: Uses torch.vmap to avoid slow Python loops over particles.
    2. Chunking: Processes particles in chunks if N_particles > max_chunk_size to manage memory.
    3. Half-Precision: Optionally uses torch.float16 to reduce memory and potentially speed up computation.

    Args:
        particles_forecast (torch.Tensor): Forecasted particles of shape (batch_size, N_particles, d_state).
        observation_y (torch.Tensor): Observation tensor. Expected shape is (batch_size, d_obs) or (d_obs,).
        observation_operator (callable): The observation mapping function, y_pred = h(x_forecast),
                                         which maps a state of shape (d_state,) to an observation of shape (d_obs,).
        sigma_y (float): Standard deviation of the observation noise (assumed to be isotropic Gaussian).
        resampling_method (str): The resampling method to use, either "multinomial" or "systematic".
        sigma_reg (float, optional): Std dev for regularization noise. If > 0, adds N(0, sigma_reg^2 * I) noise.
        use_half_precision (bool): If True, converts key tensors to float16 for the computation.
        max_chunk_size (int): The maximum number of particles to process in a single vectorized call.

    Returns:
        torch.Tensor: The analysis particles after resampling and optional regularization,
                      of shape (batch_size, N_particles, d_state).
    """
    batch_size, N_particles, d_state = particles_forecast.shape
    device = particles_forecast.device
    original_dtype = particles_forecast.dtype # Keep track of original dtype to cast back at the end

    if N_particles == 0:
        return particles_forecast

    # 1. Precision Handling
    # ---------------------
    # Optionally convert to half precision to save memory and potentially speed up computation.
    if use_half_precision:
        compute_dtype = torch.float16
        particles_forecast = particles_forecast.to(compute_dtype)
        if observation_y is not None:
            observation_y = observation_y.to(compute_dtype)
    else:
        compute_dtype = original_dtype

    # 2. Update (Compute Weights)
    # ---------------------------
    log_weights = torch.zeros(batch_size, N_particles, device=device, dtype=compute_dtype)

    if observation_y is not None:
        # --- Prepare observation_y for broadcasting ---
        if observation_y.ndim == 1:   # Shape (d_obs,)
            d_obs = observation_y.shape[0]
            obs_y_broadcastable = observation_y.view(1, 1, d_obs)
        elif observation_y.ndim == 2: # Shape (batch_size, d_obs) or (1, d_obs)
            d_obs = observation_y.shape[1]
            obs_y_broadcastable = observation_y.unsqueeze(1)
        else:
            raise ValueError("observation_y must be a 1D or 2D tensor.")

        # --- Predict observations for all particles (Vectorized with optional chunking) ---
        vectorized_op = vmap(vmap(observation_operator))

        if N_particles > max_chunk_size:
            # --- Chunked processing to save memory ---
            for i in range(0, N_particles, max_chunk_size):
                chunk_end = min(i + max_chunk_size, N_particles)
                particles_chunk = particles_forecast[:, i:chunk_end, :]

                # Apply vectorized operator on the smaller chunk
                y_forecast_chunk = vectorized_op(particles_chunk)

                # Calculate log-likelihoods for the chunk
                diff_sq_chunk = ((obs_y_broadcastable - y_forecast_chunk) / sigma_y) ** 2
                log_weights_chunk = -0.5 * torch.sum(diff_sq_chunk, dim=2)
                
                # Store the result in the full log_weights tensor
                log_weights[:, i:chunk_end] = log_weights_chunk
        else:
            # --- Full vectorized processing (if N_particles is small enough) ---
            y_forecast = vectorized_op(particles_forecast)
            diff_sq = ((obs_y_broadcastable - y_forecast) / sigma_y) ** 2
            log_weights = -0.5 * torch.sum(diff_sq, dim=2)

        # --- Normalize weights using log-sum-exp trick for numerical stability ---
        max_log_w = torch.max(log_weights, dim=1, keepdim=True)[0]
        weights_unnormalized = torch.exp(log_weights - max_log_w)
        sum_weights = torch.sum(weights_unnormalized, dim=1, keepdim=True)

        uniform_dist = torch.full((N_particles,), 1.0 / N_particles, device=device, dtype=compute_dtype)
        weights = uniform_dist.unsqueeze(0).expand(batch_size, -1).clone()
        
        good_batches_mask = (sum_weights > 1e-6).squeeze(-1) # Use 1e-6 for float16 safety

        if good_batches_mask.any():
            normalized_w_good = weights_unnormalized[good_batches_mask] / sum_weights[good_batches_mask]
            weights[good_batches_mask] = normalized_w_good
    else:
        # If no observation, all particles have uniform weight
        weights = torch.full((batch_size, N_particles), 1.0 / N_particles, device=device, dtype=compute_dtype)

    # 3. Resampling (This section is already efficient)
    # -----------------------------------------------
    indices = torch.multinomial(weights, N_particles, replacement=True)
    
    # The systematic resampling part is slightly more complex to make fully compatible
    # with batching when weights can be all zeros for some batches.
    # Multinomial is generally robust and efficient. For simplicity, this implementation
    # will focus on the more common multinomial resampling. If systematic is needed,
    # careful handling of the 'weights' tensor is required.
    if resampling_method == "systematic":
        # Note: Systematic resampling can be sensitive to batches with all-zero weights.
        # This implementation is a standard approach.
        cdf = torch.cumsum(weights, dim=1)
        cdf[:, -1] = 1.0 # Ensure the CDF ends at 1.0 for robustness

        u_start = torch.rand(batch_size, 1, device=device, dtype=compute_dtype) / N_particles
        u_uniform_strata = torch.arange(N_particles, device=device, dtype=compute_dtype) / N_particles
        u_samples = u_start + u_uniform_strata.unsqueeze(0)

        indices = torch.searchsorted(cdf, u_samples, right=True)
        indices.clamp_(0, N_particles - 1)
    elif resampling_method != "multinomial":
        raise ValueError(f"Unknown resampling method: {resampling_method}")

    # Gather the resampled particles using advanced indexing
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(indices)
    particles_analysis = particles_forecast[batch_indices, indices]

    # 4. Regularization (Optional, already efficient)
    # ---------------------------------------------
    if sigma_reg is not None and sigma_reg > 0:
        noise = torch.randn_like(particles_analysis) # noise will have dtype=compute_dtype
        particles_analysis = particles_analysis + noise * sigma_reg

    # 5. Final Type Casting
    # ---------------------
    # Cast the final result back to the original data type if needed
    if use_half_precision:
        particles_analysis = particles_analysis.to(original_dtype)
        
    return particles_analysis


# ##############################################################################
# # Ensemble Kalman Filters (EnKF) (Analysis Step Only)
# ##############################################################################

def _enkf_pert_obs_analysis(
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,)
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y,                # scalar or (B,)
    localization_matrix_Lxy=None, # (d_state, d_obs), broadcasts
    localization_matrix_Lyy=None  # (d_obs, d_obs), broadcasts
):
    """ EnKF with Perturbed Observations - Analysis Step (Batched) """
    batch_size, N_ensemble, d_state = ensemble_f.shape
    
    if observation_y.ndim == 1:
        obs_y_eff = observation_y.unsqueeze(0) 
        d_obs = observation_y.shape[0]
    else:
        obs_y_eff = observation_y 
        d_obs = observation_y.shape[-1]
        if obs_y_eff.shape[0] != batch_size and obs_y_eff.shape[0] != 1:
             raise ValueError("Batch size of observation_y must match ensemble_f or be 1.")

    device = ensemble_f.device
    dtype = ensemble_f.dtype

    ensemble_y_f = observation_operator_ens(ensemble_f)

    Af, _ = center_ensemble(ensemble_f, rescale=False)
    AYf, _ = center_ensemble(ensemble_y_f, rescale=False)

    scaling_factor = 1.0 / (N_ensemble - 1) if N_ensemble > 1 else 1.0
    
    Pxy = (Af.transpose(-2, -1) @ AYf) * scaling_factor
    Pyy = (AYf.transpose(-2, -1) @ AYf) * scaling_factor

    if localization_matrix_Lxy is not None:
        Pxy = Pxy * localization_matrix_Lxy
    if localization_matrix_Lyy is not None:
        Pyy = Pyy * localization_matrix_Lyy

    if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim == 1 and sigma_y.shape[0] == batch_size:
        R_val = sigma_y.view(batch_size, 1, 1)**2
        R_obs = torch.eye(d_obs, device=device, dtype=dtype).unsqueeze(0) * R_val
    else:
        R_obs = (sigma_y**2) * torch.eye(d_obs, device=device, dtype=dtype)

    innovation_cov = Pyy + R_obs

    # --- MODIFICATION START ---
    # Add a small regularization term to innovation_cov to improve stability
    epsilon = 1e-6 # Regularization strength; adjust if necessary
    
    # Ensure reg_identity is correctly broadcastable for batched innovation_cov
    if innovation_cov.ndim == 3 and batch_size > 0 : # Batched
        reg_identity = torch.eye(d_obs, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    elif innovation_cov.ndim == 2: # Non-batched (should not occur if inputs are batched)
         reg_identity = torch.eye(d_obs, device=device, dtype=dtype)
    else: # Handles batch_size = 0 or other unexpected dims for innovation_cov
        reg_identity = torch.eye(d_obs, device=device, dtype=dtype).unsqueeze(0)


    innovation_cov_reg = innovation_cov + epsilon * reg_identity
    # --- MODIFICATION END ---
    
    try:
        # K^T = solve(S_reg, Pxy^T) -> K = (solve(S_reg, Pxy^T))^T
        kalman_gain_T = torch.linalg.solve(innovation_cov_reg, Pxy.transpose(-2, -1))
        kalman_gain = kalman_gain_T.transpose(-2, -1)
    except torch.linalg.LinAlgError: # Catches errors like singularity if solve fails
        # Fallback to pseudo-inverse if solve fails even with regularization
        kalman_gain = Pxy @ torch.linalg.pinv(innovation_cov_reg)

    if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim == 1 and sigma_y.shape[0] == batch_size:
        sigma_y_expanded = sigma_y.view(batch_size, 1, 1)
    else:
        sigma_y_expanded = sigma_y

    obs_perturbations = sigma_y_expanded * torch.randn(batch_size, N_ensemble, d_obs, device=device, dtype=dtype)
    perturbed_obs = obs_y_eff.unsqueeze(1) + obs_perturbations
    innovations = perturbed_obs - ensemble_y_f
    ensemble_a = ensemble_f + innovations @ kalman_gain.transpose(-2, -1)

    return ensemble_a, kalman_gain


def _ersf_analysis( # Ensemble Randomized Square Root Filter (ETKF variant) - Batched
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,)
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y                 # scalar or (B,)
):
    """ Ensemble Randomized Square Root Filter (ETKF) - Analysis Step (Batched) """
    batch_size, N_ensemble, d_state = ensemble_f.shape
    device = ensemble_f.device
    dtype = ensemble_f.dtype

    if observation_y.ndim == 1:
        obs_y_eff = observation_y.unsqueeze(0) # (1, d_obs)
    else:
        obs_y_eff = observation_y # (B, d_obs)
        if obs_y_eff.shape[0] != batch_size and obs_y_eff.shape[0] != 1:
             raise ValueError("Batch size of observation_y must match ensemble_f or be 1.")
    
    # N1: (N_ensemble - 1)
    N1_val = max(N_ensemble - 1.0, 1.0) # scalar

    # ensemble_y_f: (B, N_ensemble, d_obs)
    ensemble_y_f = observation_operator_ens(ensemble_f)

    # Af: (B, N_ensemble, d_state), mean_f: (B, 1, d_state)
    Af, mean_f = center_ensemble(ensemble_f, rescale=False)
    # AYf: (B, N_ensemble, d_obs), mean_yf: (B, 1, d_obs)
    AYf, mean_yf = center_ensemble(ensemble_y_f, rescale=False)

    # Prepare sigma_y for division, ensure it's (B,1,1) or scalar
    if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim > 0:
        sigma_y_sq_inv = (1.0 / sigma_y**2).view(-1, 1, 1) # (B,1,1) or (1,1,1)
    else: # scalar
        sigma_y_sq_inv = 1.0 / (sigma_y**2)

    # C_tilde_sym: (B, N_ensemble, N_ensemble)
    # Original AYf @ AYf.T is (N_ens, N_y) @ (N_y, N_ens) -> (N_ens, N_ens)
    # Batched: (B, N_ens, N_y) @ (B, N_y, N_ens) -> (B, N_ens, N_ens)
    C_tilde_sym = (AYf @ AYf.transpose(-2, -1)) * sigma_y_sq_inv + \
                  N1_val * torch.eye(N_ensemble, device=device, dtype=dtype).unsqueeze(0)

    eig_vals, eig_vecs = torch.linalg.eigh(C_tilde_sym) # eig_vals (B,N), eig_vecs (B,N,N)
    eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)

    # T_transform_matrix: (B, N_ensemble, N_ensemble)
    T_transform_matrix = eig_vecs @ torch.diag_embed(eig_vals_clamped**-0.5) @ \
                         eig_vecs.transpose(-2, -1) * torch.sqrt(torch.tensor(N1_val, device=device, dtype=dtype))
    # Pw_term: (B, N_ensemble, N_ensemble)
    Pw_term = eig_vecs @ torch.diag_embed(eig_vals_clamped**-1) @ eig_vecs.transpose(-2, -1)

    # innovation_dy: (B, 1, d_obs)
    innovation_dy = (obs_y_eff.unsqueeze(1) - mean_yf)
    # w_gain_transpose: (B, 1, N_ensemble)
    w_gain_transpose = (innovation_dy @ AYf.transpose(-2, -1)) @ Pw_term * sigma_y_sq_inv

    # mean_a: (B, 1, d_state)
    mean_a = mean_f + w_gain_transpose @ Af
    # Af_updated: (B, N_ensemble, d_state)
    Af_updated = T_transform_matrix @ Af # T operates on rows of Af
    # ensemble_a: (B, N_ensemble, d_state)
    ensemble_a = mean_a + Af_updated

    return ensemble_a, None


def _letkf_core_etkf_update(
    local_E_f_mean,     # (B, N_x_local)
    local_A_f,          # (B, N_ens, N_x_local)
    eff_AY_f_anom,      # (B, N_ens, N_y_local)
    eff_d_f_innov,      # (B, N_y_local)
    N_ensemble
):
    """ Core ETKF update for local patches (Batched), assuming R_eff = I. """
    device = local_A_f.device; dtype = local_A_f.dtype
    batch_size = local_A_f.shape[0]
    N1_val = max(N_ensemble - 1.0, 1.0) # scalar

    # Pa_tilde_inv_sqrt: (B, N_ensemble, N_ensemble)
    Pa_tilde_inv_sqrt = eff_AY_f_anom @ eff_AY_f_anom.transpose(-2, -1) + \
                        N1_val * torch.eye(N_ensemble, device=device, dtype=dtype).unsqueeze(0)

    eig_vals, eig_vecs = torch.linalg.eigh(Pa_tilde_inv_sqrt)
    eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)

    # T_transform: (B, N_ensemble, N_ensemble)
    T_transform = eig_vecs @ torch.diag_embed(eig_vals_clamped**-0.5) @ \
                  eig_vecs.transpose(-2, -1) * torch.sqrt(torch.tensor(N1_val, device=device, dtype=dtype))
    # Pw: (B, N_ensemble, N_ensemble)
    Pw = eig_vecs @ torch.diag_embed(eig_vals_clamped**-1) @ eig_vecs.transpose(-2, -1)

    # w_gain_transpose: (B, 1, N_ensemble)
    w_gain_transpose = (eff_d_f_innov.unsqueeze(1) @ eff_AY_f_anom.transpose(-2, -1)) @ Pw

    # local_mean_a: (B, N_x_local)
    # (w_gain_transpose @ local_A_f) is (B, 1, N_x_local)
    mean_update_term = (w_gain_transpose @ local_A_f).squeeze(1) # (B, N_x_local)
    local_mean_a = local_E_f_mean + mean_update_term

    # local_A_a: (B, N_ensemble, N_x_local)
    local_A_a = T_transform @ local_A_f

    return local_mean_a, local_A_a


def _letkf_analysis(
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,)
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y,                # scalar observation error standard deviation
    localization_radius,    # scalar
    coords_state,           # (d_state, D_coord_state)
    coords_obs,             # (d_obs, D_coord_obs)
    domain=None     # (D_coord_state,) or (D_coord_obs,)
):
    """ Local Ensemble Transform Kalman Filter (LETKF) - Analysis Step (Batched) """
    batch_size, N_ensemble, d_state = ensemble_f.shape
    device = ensemble_f.device; dtype = ensemble_f.dtype

    if observation_y.ndim == 1:
        obs_y_eff = observation_y.unsqueeze(0) # (1, d_obs)
        d_obs = observation_y.shape[0]
    else:
        obs_y_eff = observation_y # (B, d_obs)
        d_obs = observation_y.shape[-1]
        if obs_y_eff.shape[0] != batch_size and obs_y_eff.shape[0] != 1:
             raise ValueError("Batch size of observation_y must match ensemble_f or be 1.")

    # ensemble_y_f: (B, N_ensemble, d_obs)
    ensemble_y_f = observation_operator_ens(ensemble_f)
    # Af_global: (B, N_ensemble, d_state), mean_f_global: (B, 1, d_state)
    Af_global, mean_f_global = center_ensemble(ensemble_f, rescale=False)
    # AYf_global: (B, N_ensemble, d_obs), mean_yf_global: (B, 1, d_obs)
    AYf_global, mean_yf_global = center_ensemble(ensemble_y_f, rescale=False)

    # innovation_mean_global: (B, 1, d_obs)
    innovation_mean_global = obs_y_eff.unsqueeze(1) - mean_yf_global
    
    # Transform observations and innovations by R^-1/2 (here R = sigma_y^2 * I)
    AYf_global_transformed = AYf_global / sigma_y         # (B, N_ensemble, d_obs)
    innovation_mean_global_transformed = innovation_mean_global / sigma_y # (B, 1, d_obs)

    # Initialize analysis ensemble parts
    ensemble_a_mean_parts = torch.zeros_like(mean_f_global) # (B, 1, d_state)
    ensemble_a_anom_parts = torch.zeros_like(Af_global)   # (B, N_ensemble, d_state)

    # Loop over each state variable to update it locally
    for k_state_idx in range(d_state):
        # current_mean_f_k: (B, 1) mean of k-th state var for all batches
        current_mean_f_k = mean_f_global[:, :, k_state_idx]
        # current_Af_k: (B, N_ensemble, 1) anomalies of k-th state var
        current_Af_k = Af_global[:, :, k_state_idx].unsqueeze(-1)

        # --- Localization: This part is NOT batched over `batch_size` ---
        # --- It's computed once per k_state_idx as coords are shared ---
        # Coords for k-th state var: (1, D_coord)
        coord_k_state = coords_state[k_state_idx].unsqueeze(0)

        # Distances from k-th state variable to all observations: (1, d_obs)
        # Assumes pairwise_distances can handle (N,D) (M,D) -> (N,M) inputs
        # or a specific 2D version is used for these non-batched coordinates.
        dist_state_k_to_obs = pairwise_distances(
            coord_k_state, coords_obs, domain=domain
        ).squeeze(0) # -> (d_obs,)
        
        rho_k = dist2coeff(dist_state_k_to_obs, localization_radius) # (d_obs,)
        local_obs_indices = torch.where(rho_k > 1e-6)[0] # (N_y_local_k,)
        
        if len(local_obs_indices) == 0: # No observations influence this state variable
            ensemble_a_mean_parts[:, :, k_state_idx] = current_mean_f_k
            ensemble_a_anom_parts[:, :, k_state_idx] = current_Af_k.squeeze(-1)
            continue

        # Select local observations for this k_state_idx
        # These are now batched over `batch_size`
        # AYf_local_k_transformed: (B, N_ensemble, N_y_local_k)
        AYf_local_k = AYf_global_transformed[:, :, local_obs_indices]
        # innov_local_k_transformed: (B, 1, N_y_local_k)
        innov_local_k = innovation_mean_global_transformed[:, :, local_obs_indices]
        
        # Apply localization weights to observations (sqrt_rho acts on transformed obs anoms)
        rho_local_k_weights = rho_k[local_obs_indices] # (N_y_local_k,)
        # sqrt_rho_local_k broadcastable: (1, 1, N_y_local_k)
        sqrt_rho_local_k_bcast = torch.sqrt(rho_local_k_weights).view(1, 1, -1)

        # eff_AYf_k_anom: (B, N_ensemble, N_y_local_k)
        eff_AYf_k_anom = AYf_local_k * sqrt_rho_local_k_bcast
        # eff_innov_k: (B, 1, N_y_local_k) -> squeezed to (B, N_y_local_k)
        eff_innov_k = (innov_local_k * sqrt_rho_local_k_bcast).squeeze(1)

        # Core ETKF update for (k_state_idx, and all batches)
        # current_mean_f_k is (B,1), current_Af_k is (B, N_ens, 1)
        updated_mean_k, updated_A_k = _letkf_core_etkf_update(
            current_mean_f_k, current_Af_k,
            eff_AYf_k_anom, eff_innov_k, N_ensemble
        )
        # updated_mean_k: (B,1), updated_A_k: (B, N_ensemble, 1)

        ensemble_a_mean_parts[:, :, k_state_idx] = updated_mean_k
        ensemble_a_anom_parts[:, :, k_state_idx] = updated_A_k.squeeze(-1)

    ensemble_a = ensemble_a_mean_parts + ensemble_a_anom_parts
    return ensemble_a, None


def ensemble_kalman_filter_analysis(
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,) or None
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y,                # scalar or (B,)
    method="EnKF-PertObs",
    inflation_factor=1.0,   # scalar
    # For EnKF-PertObs
    localization_matrix_Lxy=None, # (d_state, d_obs)
    localization_matrix_Lyy=None, # (d_obs, d_obs)
    # For LETKF
    localization_radius_letkf=None, # scalar
    coords_state_letkf=None,        # (d_state, D_coord)
    coords_obs_letkf=None,          # (d_obs, D_coord)
    domain_letkf=None       # (D_coord,)
):
    """ Main dispatcher for ensemble Kalman filter analysis (Batched) """
    kalman_gain_or_transform = None
    ensemble_a_raw = None

    if observation_y is None: # No observation, forecast is analysis
        ensemble_a_raw = ensemble_f
    elif method == "EnKF-PertObs":
        ensemble_a_raw, kalman_gain_or_transform = _enkf_pert_obs_analysis(
            ensemble_f, observation_y, observation_operator_ens, sigma_y,
            localization_matrix_Lxy, localization_matrix_Lyy
        )
    elif method == "ERSF": # ETKF variant
        ensemble_a_raw, kalman_gain_or_transform = _ersf_analysis(
            ensemble_f, observation_y, observation_operator_ens, sigma_y
        )
    elif method == "LETKF":
        if localization_radius_letkf is None or \
           coords_state_letkf is None or \
           coords_obs_letkf is None:
            raise ValueError("LETKF requires localization_radius, coords_state, and coords_obs.")
        ensemble_a_raw, kalman_gain_or_transform = _letkf_analysis(
            ensemble_f, observation_y, observation_operator_ens, sigma_y,
            localization_radius_letkf, coords_state_letkf,
            coords_obs_letkf, domain_letkf
        )
    else:
        raise ValueError(f"Unknown EnKF method: {method}")

    # Apply inflation to the raw analysis ensemble
    ensemble_analysis = apply_inflation(ensemble_a_raw, inflation_factor)

    return ensemble_analysis, kalman_gain_or_transform

# ##############################################################################
# # Lorenz 96 and RK4 for Testing
# ##############################################################################
def lorenz96_rhs(x, F):
    """
    Calculates the RHS of the Lorenz 96 equations.
    x can be a 1D tensor (D_state,) or a 2D tensor (batch_size, D_state).
    F is a scalar forcing term.
    """
    D = x.shape[-1] # Works for both 1D and 2D x

    # Efficiently calculate indices for all D components
    # Indices are relative to the current component 'k'
    # x_k-2, x_k-1, x_k, x_k+1
    # For dxdt[k] = (x[k+1] - x[k-2]) * x[k-1] - x[k] + F

    # Create rolled versions of x for vectorized computation
    # x_m2 means x[(i-2+D)%D] for each i
    # x_m1 means x[(i-1+D)%D] for each i
    # x_p1 means x[(i+1)%D] for each i
    x_m2 = torch.roll(x, shifts=2, dims=-1)
    x_m1 = torch.roll(x, shifts=1, dims=-1)
    x_p1 = torch.roll(x, shifts=-1, dims=-1)

    dxdt = (x_p1 - x_m2) * x_m1 - x + F
    return dxdt

def rk4_step(rhs_func, x, dt, F_param):
    """
    Performs one RK4 step.
    x can be (D_state,) or (batch_size, D_state).
    rhs_func is compatible with batched x.
    """
    k1 = rhs_func(x, F_param)
    k2 = rhs_func(x + 0.5 * dt * k1, F_param)
    k3 = rhs_func(x + 0.5 * dt * k2, F_param)
    k4 = rhs_func(x + dt * k3, F_param)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

# ##############################################################################
# # Main Test Script
# ##############################################################################
if __name__ == '__main__':
    print("Running Lorenz 96 DA example with 3D Tensors (Batch Support)...")
    device = 'cuda:0'
    dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # --- Configuration ---
    D_L96 = 40 # Reduced for faster testing, original was 500
    F_L96_const = 8.0
    dt_rk4 = 0.03 # Slightly larger dt for L40
    num_rk4_steps_between_analyses = 5
    dt_analysis = dt_rk4 * num_rk4_steps_between_analyses

    N_ensemble = 20 # Reduced N_ensemble for speed
    sigma_d_val = 0 # Model error std dev
    sigma_y_val = 1 # Observation error std dev
    inflation_val = 1.05

    batch_size = 5 # Introduce batch size

    d_obs_l96 = D_L96 # Observe all state variables
    
    def obs_op_l96(x_curr): 
        return torch.arctan(x_curr)
        # return x_curr

    # --- True State Initialization and Spin-up ---
    # true_state_t: (batch_size, D_L96)
    true_state_t = torch.rand(batch_size, D_L96, device=device, dtype=dtype) * 15.0 - 5.0 # L96 range
    true_state_t[:, D_L96 // 2] += F_L96_const # Add some perturbation to break symmetry for batches

    print("Spinning up true states (batched)...")
    for _ in range(100): # Spin-up iterations
        for _ in range(num_rk4_steps_between_analyses):
            true_state_t = rk4_step(lorenz96_rhs, true_state_t, dt_rk4, F_L96_const)
    print("True state spin-up complete.")

    # --- Initial Ensembles (Batched) ---
    # initial_ensemble: (batch_size, N_ensemble, D_L96)
    # true_state_t.unsqueeze(1): (B, 1, D) for broadcasting
    initial_ensemble = true_state_t.unsqueeze(1) + \
                       torch.randn(batch_size, N_ensemble, D_L96, device=device, dtype=dtype) * 2.0

    current_particles_bpf = initial_ensemble.clone()
    current_ensemble_enkf_po = initial_ensemble.clone()
    current_ensemble_ersf = initial_ensemble.clone()
    current_ensemble_letkf = initial_ensemble.clone()

    # --- Localization Setup (Not batched, shared across batches) ---
    coords_state_l96 = torch.arange(D_L96, device=device, dtype=dtype).unsqueeze(1) # (D_L96, 1)
    coords_obs_l96 = torch.arange(d_obs_l96, device=device, dtype=dtype).unsqueeze(1) # (d_obs_l96, 1)
    domain_l96 = torch.tensor([D_L96], device=device, dtype=dtype) # (1,)
    loc_radius_gc = 3.0 # Effective radius for Gaspari-Cohn like localization

    # Assuming pairwise_distances and dist2coeff are available and work as expected.
    # If pairwise_distances is strictly 3D (B,N,D_coord), these calls need adjustment:
    # e.g., coords_state_l96.unsqueeze(0) to make it (1, D_L96, 1)
    # For simplicity, assume the existing call structure is valid with the (assumed fixed) utils.
    try:
        Lxy_l96 = dist2coeff(
            pairwise_distances(coords_state_l96, coords_obs_l96, domain=domain_l96),
            loc_radius_gc
        )
        Lyy_l96 = dist2coeff(
            pairwise_distances(coords_obs_l96, coords_obs_l96, domain=domain_l96),
            loc_radius_gc
        )
    except NameError: # Fallback if helper functions are not defined
        print("Warning: Lxy_l96/Lyy_l96 not computed due to missing pairwise_distances or dist2coeff.")
        Lxy_l96, Lyy_l96 = None, None


    # --- Main Simulation Loop ---
    num_analysis_cycles = 1000 # Reduced cycles for faster test
    print(f"\nRunning {num_analysis_cycles} analysis cycles (batch_size={batch_size}, dt_analysis={dt_analysis:.2f})...")
    results_rmse = {"BPF": [], "EnKF-PO": [], "ERSF": [], "LETKF": []}
    analysis_times = {"BPF": [], "EnKF-PO": [], "ERSF": [], "LETKF": []}

    for cycle in range(num_analysis_cycles):
        # 1. Forecast True State (Batched)
        for _ in range(num_rk4_steps_between_analyses):
            true_state_t = rk4_step(lorenz96_rhs, true_state_t, dt_rk4, F_L96_const)

        # 2. Generate Observation (Batched)
        # observation_y_t: (batch_size, d_obs_l96)
        observation_y_t = obs_op_l96(true_state_t) + \
                          sigma_y_val * torch.randn(batch_size, d_obs_l96, device=device, dtype=dtype)

        # 3. Forecast Ensembles (Batched)
        ensembles_to_forecast = {
            "BPF": current_particles_bpf,
            "EnKF-PO": current_ensemble_enkf_po,
            "ERSF": current_ensemble_ersf,
            "LETKF": current_ensemble_letkf
        }
        forecast_inputs = {}

        for name, ens_batch in ensembles_to_forecast.items():
            # ens_batch shape: (batch_size, N_ensemble, D_L96)
            # Reshape for vectorized forecast: (batch_size * N_ensemble, D_L96)
            num_total_members = batch_size * N_ensemble
            current_members_flat = ens_batch.reshape(num_total_members, D_L96)

            propagated_members_flat = current_members_flat
            for _ in range(num_rk4_steps_between_analyses):
                propagated_members_flat = rk4_step(lorenz96_rhs, propagated_members_flat, dt_rk4, F_L96_const)
            
            # Reshape back: (batch_size, N_ensemble, D_L96)
            forecasted_ens = propagated_members_flat.reshape(batch_size, N_ensemble, D_L96)
            
            # Add model error
            if sigma_d_val > 0:
                forecasted_ens += sigma_d_val * torch.randn_like(forecasted_ens)
            forecast_inputs[name] = forecasted_ens
        
        # 4. Analysis Step for each method (Batched)
        # For RMSE, we'll report for the first batch element to keep output simple
        batch_idx_for_rmse = 0

        # BPF
        try:
            start_time = time.perf_counter()
            current_particles_bpf = bootstrap_particle_filter_analysis(
                forecast_inputs["BPF"], observation_y_t,
                obs_op_l96, # Operates on (D_state), BPF analysis handles iteration
                sigma_y_val,
                resampling_method="multinomial" 
            )
            end_time = time.perf_counter()
            analysis_times["BPF"].append(end_time - start_time)
            rmse_bpf = torch.sqrt(torch.mean((current_particles_bpf[batch_idx_for_rmse].mean(dim=0) - true_state_t[batch_idx_for_rmse])**2)).item()
            results_rmse["BPF"].append(rmse_bpf)
        except NameError: results_rmse["BPF"].append(float('nan')); analysis_times["BPF"].append(float('nan'))


        # EnKF Methods (using the main dispatcher)
        common_enkf_args = {
            "observation_y": observation_y_t,
            "observation_operator_ens": obs_op_l96,
            "sigma_y": sigma_y_val,
            "inflation_factor": inflation_val
        }
        
        # EnKF-PertObs
        try:
            start_time = time.perf_counter()
            current_ensemble_enkf_po, _ = ensemble_kalman_filter_analysis(
                forecast_inputs["EnKF-PO"], **common_enkf_args,
                method="EnKF-PertObs",
                localization_matrix_Lxy=Lxy_l96, 
                localization_matrix_Lyy=Lyy_l96 
            )
            end_time = time.perf_counter()
            analysis_times["EnKF-PO"].append(end_time - start_time)
            rmse_enkf_po = torch.sqrt(torch.mean((current_ensemble_enkf_po[batch_idx_for_rmse].mean(dim=0) - true_state_t[batch_idx_for_rmse])**2)).item()
            results_rmse["EnKF-PO"].append(rmse_enkf_po)
        except NameError: results_rmse["EnKF-PO"].append(float('nan')); analysis_times["EnKF-PO"].append(float('nan'))

        # ERSF
        try:
            start_time = time.perf_counter()
            current_ensemble_ersf, _ = ensemble_kalman_filter_analysis(
                forecast_inputs["ERSF"], **common_enkf_args,
                method="ERSF"
            )
            end_time = time.perf_counter()
            analysis_times["ERSF"].append(end_time - start_time)
            rmse_ersf = torch.sqrt(torch.mean((current_ensemble_ersf[batch_idx_for_rmse].mean(dim=0) - true_state_t[batch_idx_for_rmse])**2)).item()
            results_rmse["ERSF"].append(rmse_ersf)
        except NameError: results_rmse["ERSF"].append(float('nan')); analysis_times["ERSF"].append(float('nan'))
            
        # LETKF
        try:
            start_time = time.perf_counter()
            current_ensemble_letkf, _ = ensemble_kalman_filter_analysis(
                # The code `forecast_inputs["LETKF"]` is accessing the value associated with the key
                # "LETKF" in the dictionary `forecast_inputs`.
                
                forecast_inputs["LETKF"], **common_enkf_args,
                method="LETKF",
                localization_radius_letkf=loc_radius_gc,
                coords_state_letkf=coords_state_l96,
                coords_obs_letkf=coords_obs_l96,
                domain_letkf=domain_l96
            )
            end_time = time.perf_counter()
            analysis_times["LETKF"].append(end_time - start_time)
            rmse_letkf = torch.sqrt(torch.mean((current_ensemble_letkf[batch_idx_for_rmse].mean(dim=0) - true_state_t[batch_idx_for_rmse])**2)).item()
            results_rmse["LETKF"].append(rmse_letkf)
        except NameError: results_rmse["LETKF"].append(float('nan')); analysis_times["LETKF"].append(float('nan'))

        verbose_step = 100
        if (cycle + 1) % verbose_step == 0 or cycle == num_analysis_cycles - 1:
            print(f"Cycle {cycle + 1:3d} (RMSEs for batch[0]): "
                  f"BPF: {torch.tensor(results_rmse['BPF'][-verbose_step:]).mean():.3f} (t:{torch.tensor(analysis_times['BPF'][-verbose_step:]).mean():.4f}s), "
                  f"EnKF-PO: {torch.tensor(results_rmse['EnKF-PO'][-verbose_step:]).mean():.3f} (t:{torch.tensor(analysis_times['EnKF-PO'][-verbose_step:]).mean():.4f}s), "
                  f"ERSF: {torch.tensor(results_rmse['ERSF'][-verbose_step:]).mean():.3f} (t:{torch.tensor(analysis_times['ERSF'][-verbose_step:]).mean():.4f}s), "
                  f"LETKF: {torch.tensor(results_rmse['LETKF'][-verbose_step:]).mean():.3f} (t:{torch.tensor(analysis_times['LETKF'][-verbose_step:]).mean():.4f}s)")

    print("\nExample L96 DA run with batching completed.")

    print("\n--- Average Analysis Step Times (across all batches processed together) ---")
    for method_name, times_list in analysis_times.items():
        valid_times = [t for t in times_list if not torch.isnan(torch.tensor(t))]
        rmse_mean = torch.tensor(results_rmse[method_name]).mean()
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            print(f"{method_name}: {avg_time:.6f} seconds per analysis step; Mean RMSE: {rmse_mean:.4f}")
        else:
            print(f"{method_name}: No analysis steps timed or all failed.")