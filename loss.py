import torch
import torch.nn as nn
import warnings

def compute_es(ens_states, true_states, norm_p=1):
    """
    Computes the Energy Score (ES).
    ES(F, y) = E_F[||X - y||^norm_p] - (1/2) * E_F[||X - X'||^norm_p]
    where ||.|| is the L2 norm. For classical ES, norm_p is 1.

    Args:
        ens_states (torch.Tensor): Ensemble predictions. Shape: [T, B, N, D].
        true_states (torch.Tensor): Ground truth. Shape: [T, B, D].
        norm_p (int, float): The power to which the L2 norm of distances is raised. Defaults to 1.

    Returns:
        torch.Tensor: Energy Score. Shape: [T, B].
    """
    T, B, N, D = ens_states.shape

    if N <= 1: # ES is not well-defined or is trivially zero for N <= 1.
        return torch.zeros(T, B, device=ens_states.device, dtype=ens_states.dtype)

    # First term: E_F[||X - y||^norm_p]
    # Approximate E_F by averaging over ensemble members.
    true_expanded = true_states.unsqueeze(2)  # Shape: [T, B, 1, D]
    # L2 norm for distances, then raise to norm_p.
    dist_to_true = torch.norm(ens_states - true_expanded, p=2, dim=-1) # Shape: [T, B, N]
    if norm_p != 1:
        dist_to_true = torch.pow(dist_to_true, norm_p)
    
    term_obs = torch.mean(dist_to_true, dim=2)  # Shape: [T, B]
    
    sum_pairwise_dist = torch.zeros(T, B, device=ens_states.device, dtype=ens_states.dtype)
    for i in range(N):
        for j in range(i + 1, N): # Iterate over distinct pairs (i < j)
            dist_pair = torch.norm(ens_states[:, :, i, :] - ens_states[:, :, j, :], p=2, dim=-1) # Shape: [T, B]
            if norm_p != 1:
                dist_pair = torch.pow(dist_pair, norm_p)
            sum_pairwise_dist += dist_pair
            
    term_pair_expectation = (2 * sum_pairwise_dist) / (N ** 2) # Shape: [T, B]
    
    es = term_obs - 0.5 * term_pair_expectation  # Shape: [T, B]
    return es

def compute_kernel_es(ens_states, true_states, sigma=None):
    """
    Computes the Kernel Energy Score (kES) using a Gaussian kernel.
    kES(F, y) = -E_F[k(X, y)] + (1/2) * E_F[k(X, X')]
    where k(x,y) = exp(-||x-y||_L2^2 / (2*sigma^2)). Lower is better.

    Args:
        ens_states (torch.Tensor): Ensemble predictions. Shape: [T, B, N, D].
        true_states (torch.Tensor): Ground truth. Shape: [T, B, D].
        sigma (float, optional): Kernel bandwidth. If None, it's estimated as the median
                                 of L2 distances between ensemble members and true_states per (T,B).

    Returns:
        torch.Tensor: Kernel Energy Score. Shape: [T, B].
    """
    T, B, N, D = ens_states.shape
    device = ens_states.device
    dtype = ens_states.dtype

    if N == 0 : # Cannot compute if ensemble is empty
        return torch.zeros(T, B, device=device, dtype=dtype)
    if N == 1 and sigma is None: # Median distance for sigma estimation needs at least 1 point,
                                 # but pairwise term is ill-defined for N=1
        # Handle N=1 case: pairwise term is zero or undefined.
        # If sigma is not provided, need a fallback or error for N=1.
        # For MMD(F, delta_y) with N=1, E_F[k(X,X')] = k(x1,x1)=1. So kES = -k(x1,y) + 0.5
        # If sigma is None and N=1, true_expanded - ens_states will be used for median, which works.
        pass


    current_sigma_val = None
    if sigma is None:
        true_expanded_for_sigma = true_states.unsqueeze(2)  # Shape: [T, B, 1, D]
        distances_for_sigma = torch.norm(ens_states - true_expanded_for_sigma, p=2, dim=-1)  # Shape: [T, B, N]
        # Median over N members for each (T,B)
        current_sigma_val = torch.median(distances_for_sigma, dim=2, keepdim=False)[0] + 1e-8 # Shape: [T, B]
    else:
        current_sigma_val = torch.tensor(sigma, device=device, dtype=dtype).expand(T, B) + 1e-8 # Shape: [T, B]
    
    # Reshape sigma for broadcasting: [T, B, 1, 1]
    current_sigma_val_sq = (current_sigma_val.unsqueeze(-1).unsqueeze(-1)) ** 2


    # First term: -E_F[k(X, y)]
    true_expanded = true_states.unsqueeze(2)  # Shape: [T, B, 1, D]
    diff_obs = ens_states - true_expanded      # Shape: [T, B, N, D]
    dist_sq_obs = torch.sum(diff_obs ** 2, dim=-1, keepdim=True)  # Shape: [T, B, N, 1]
    k_obs = torch.exp(-dist_sq_obs / (2 * current_sigma_val_sq))  # Shape: [T, B, N, 1]
    # Average over ensemble members N
    term1_ef_k_xy = torch.mean(k_obs, dim=2).squeeze(-1)  # Shape: [T, B]


    # Second term: (1/2) * E_F[k(X, X')]
    if N <= 1: # Pairwise term is zero or ill-defined
        term2_ef_k_xx_prime_avg = torch.zeros_like(term1_ef_k_xy)
        if N == 1: # E_F[k(X,X')] for N=1 could be k(x1,x1) = 1
             term2_ef_k_xx_prime_avg = torch.ones_like(term1_ef_k_xy)


    else: # N > 1
        sum_pairwise_kernel = torch.zeros(T, B, device=device, dtype=dtype)
        for i in range(N):
            xi = ens_states[:, :, i:i+1, :]  # Shape: [T, B, 1, D]
            for j in range(i + 1, N): # Iterate over distinct pairs (i < j)
                xj = ens_states[:, :, j:j+1, :]  # Shape: [T, B, 1, D]
                diff_pair = xi - xj              # Shape: [T, B, 1, D]
                dist_sq_pair = torch.sum(diff_pair ** 2, dim=-1, keepdim=True)  # Shape: [T, B, 1, 1]
                k_pair = torch.exp(-dist_sq_pair / (2 * current_sigma_val_sq))  # Shape: [T, B, 1, 1]
                sum_pairwise_kernel += k_pair.squeeze(-1).squeeze(-1)  # Accumulate [T, B]

        # E_F[k(X, X')] approximated by average over N*(N-1)/2 distinct pairs
        term2_ef_k_xx_prime_avg = (2 * sum_pairwise_kernel) / (N ** 2) # Shape: [T, B]
    
    kernel_es_val = -term1_ef_k_xy + 0.5 * term2_ef_k_xx_prime_avg
    return kernel_es_val

#######################################################
# related to the weighted particle filter loss functions
import torch

import torch

def _get_batched_weights(w, shape_constraints):
    """
    Internal helper: Normalize weights for batched computation.
    
    Inputs:
        w: (Tensor or None) Weights of shape [K, N]
        shape_constraints: (tuple) (K, N) to generate uniform weights if w is None
        
    Output:
        w_norm: (Tensor) Normalized weights of shape [K, N], sum=1 per batch
    """
    K, N = shape_constraints
    if w is None:
        return torch.full((K, N), 1.0 / N, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Ensure shape is [K, N]
    w = w.view(K, N)
    # Normalize: [K, N] / [K, 1]
    return w / (w.sum(dim=1, keepdim=True) + 1e-8)

def _compute_batched_loss(x, y, w_x, w_y, loss_type, sigma_val=None):
    """
    Internal helper: Computes weighted energy distance or MMD using fully vectorized operations.
    
    Inputs:
        x: (Tensor) Shape [K, N, D]
        y: (Tensor) Shape [K, M, D]
        w_x: (Tensor) Shape [K, N]
        w_y: (Tensor) Shape [K, M]
        loss_type: (str) 'wpf_ed', 'wpf_fmmd', 'wpf_ammd'
        sigma_val: (float, optional) Fixed sigma for fmmd
        
    Output:
        loss: (Tensor) Shape [K], loss per batch element
    """
    # 1. Compute Pairwise Distances [K, N, M] / [K, N, N] / [K, M, M]
    # torch.cdist handles batches automatically
    d_xy = torch.cdist(x, y, p=2) 
    d_xx = torch.cdist(x, x, p=2)
    d_yy = torch.cdist(y, y, p=2)

    # 2. Define Kernel Function
    if loss_type == 'wpf_ed':
        # Identity kernel for Energy Distance
        k_xy, k_xx, k_yy = d_xy, d_xx, d_yy
        
    elif loss_type in ['wpf_fmmd', 'wpf_ammd']:
        if loss_type == 'wpf_ammd':
            # Adaptive: Estimate sigma per batch element using Median Heuristic
            # We use d_xy for estimation, detached from graph
            with torch.no_grad():
                # Flatten last two dims: [K, N*M]
                flat_d = d_xy.view(d_xy.shape[0], -1)
                # Compute median per batch: [K, 1, 1] for broadcasting
                median_d = torch.median(flat_d, dim=1).values.view(-1, 1, 1)
                sigma = median_d.detach()
                sigma[sigma < 1e-5] = 1.0 # Avoid division by zero
        else:
            # Fixed Sigma
            sigma = float(sigma_val)

        gamma = 1.0 / (2 * sigma**2)
        k_xy = torch.exp(- (d_xy**2) * gamma)
        k_xx = torch.exp(- (d_xx**2) * gamma)
        k_yy = torch.exp(- (d_yy**2) * gamma)

    # 3. Compute Expectations via Batch Matrix Multiplication
    # w_x: [K, N] -> [K, 1, N]
    # K_matrix: [K, N, M]
    # w_y: [K, M] -> [K, M, 1]
    # Result: [K, 1, 1] -> squeeze to [K]
    
    wx_unsq = w_x.unsqueeze(1)
    wy_unsq = w_y.unsqueeze(2)
    wx_unsq_T = w_x.unsqueeze(2) # [K, N, 1] used for xx
    wy_unsq_T = w_y.unsqueeze(2) # [K, M, 1] used for yy (same as above)

    # E[X, Y] = w_x^T * K_xy * w_y
    e_xy = torch.bmm(torch.bmm(wx_unsq, k_xy), wy_unsq).view(-1)
    
    # E[X, X] = w_x^T * K_xx * w_x
    e_xx = torch.bmm(torch.bmm(wx_unsq, k_xx), wx_unsq_T).view(-1)
    
    # E[Y, Y] = w_y^T * K_yy * w_y
    e_yy = torch.bmm(torch.bmm(w_y.unsqueeze(1), k_yy), wy_unsq_T).view(-1)

    if loss_type == 'wpf_ed':
        return 2 * e_xy - e_xx - e_yy
    else:
        # MMD Squared
        return e_xx - 2 * e_xy + e_yy

def compute_loss(ens_tensor, batch_v, loss_type, ignore_first=0, end_ind=None, 
                 valid_B_mask=None, norm_p=1, kes_sigma=1, return_sum=False, 
                 normalize_val=None, wpf_input=None):
    """
    Computes loss. Supports various types including L2, ES, kernel ES, and WPF-based distances.

    Inputs:
        ens_tensor: (torch.Tensor) Ensemble predictions. Shape: [T, B, N, D].
        batch_v: (torch.Tensor) Ground truth. Shape: [T, B, D].
        loss_type: (str) Type of loss: 
                          Standard: 'l2', 'nl2', 'rmse', 'es', 'nes', 'tnes', 'kes', 'nkes', 'tnkes'.
                          WPF (require wpf_input): 'wpf_ed', 'wpf_fmmd', 'wpf_ammd'.
        ignore_first: (int) Number of initial time steps to ignore.
        end_ind: (int, optional) Last time step index to consider. Defaults to end of trajectory.
        valid_B_mask: (torch.Tensor, optional) Boolean mask for valid batch items.
                                               Shape: [B] or [T, B]. Defaults to all valid.
        norm_p: (int, float) Exponent for distances in ES/normalization.
        kes_sigma: (float) Bandwidth sigma for kernel ES and wpf_fmmd (default).
        return_sum: (bool) If True, returns sum of losses over valid elements. Else, returns mean.
        normalize_val: (torch.Tensor, optional) Normalization stats [B, D].
        wpf_input: (dict, optional) Extra input for WPF losses. Contains:
                                    - 'target_ens': [T, B, M, D] (Target ensemble)
                                    - 'ens_weights': [T, B, N] (Optional weights for ens_tensor)
                                    - 'target_weights': [T, B, M] (Optional weights for target_ens)
                                    - 'sigma': float (Optional override for kes_sigma in wpf_fmmd)

    Output:
        torch.Tensor: Computed loss (scalar).
    """
    _, B, D = batch_v.shape
    
    # 1. Normalization
    if normalize_val is not None:
        assert normalize_val.shape == (B, D), f"normalize_val shape mismatch: {normalize_val.shape} vs ({B}, {D})"
        batch_v = batch_v / normalize_val.unsqueeze(0)
        ens_tensor = ens_tensor / normalize_val.unsqueeze(1).unsqueeze(0)
    
    full_time_steps = batch_v.size(0)
    
    if end_ind is None:
        end_ind = full_time_steps - 1 
    end_ind = min(end_ind, full_time_steps - 1)

    if ignore_first > end_ind:
        return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

    # 2. Construct Masks
    if valid_B_mask is None:
        valid_B_mask = torch.ones(full_time_steps, batch_v.size(1), dtype=torch.bool, device=batch_v.device)
    elif valid_B_mask.ndim == 1: # Batch dimension only
        valid_B_mask = valid_B_mask.unsqueeze(0).expand(full_time_steps, -1)

    valid_B_mask_sliced = valid_B_mask[ignore_first:end_ind + 1, :]

    # 3. Time Slicing
    ens_states_timed = ens_tensor[ignore_first:end_ind + 1, :, :, :] 
    true_states_timed = batch_v[ignore_first:end_ind + 1, :, :]   
    
    if not valid_B_mask_sliced.any():
        return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

    ens_mean_timed = torch.mean(ens_states_timed, dim=2)  # Shape: [selected_time, B, D]
    
    loss_values_per_element = None # Will store [selected_time, B] shaped losses

    # =========================================================================
    # Existing Loss Types
    # =========================================================================
    if loss_type == "l2":
        loss_values_per_element = torch.sum((ens_mean_timed - true_states_timed) ** 2, dim=2)
    
    elif loss_type == 'nl2':
        error_norm_2 = torch.sum((ens_mean_timed - true_states_timed) ** 2, dim=2)
        true_norm_2 = torch.sum(true_states_timed ** 2, dim=2)
        loss_values_per_element = error_norm_2 / (true_norm_2 + 1e-8)
    
    elif loss_type == 'rmse':
        mse_features = (ens_mean_timed - true_states_timed) ** 2
        loss_values_per_element = torch.sqrt(torch.sum(mse_features, dim=2) + 1e-8)
    
    elif loss_type == 'es':
        # Assuming compute_es is available in scope
        loss_values_per_element = compute_es(ens_states_timed, true_states_timed, norm_p=norm_p)
    
    elif loss_type == 'nes' or loss_type == 'tnes':
        es_vals = compute_es(ens_states_timed, true_states_timed, norm_p=norm_p)
        true_norm_vals = torch.norm(true_states_timed, p=2, dim=2) ** norm_p
        
        if loss_type == 'nes':
            loss_values_per_element = es_vals / (true_norm_vals + 1e-8)
        else: # tnes
            sum_es_per_batch = torch.zeros(batch_v.size(1), device=batch_v.device, dtype=ens_tensor.dtype)
            sum_norm_per_batch = torch.zeros(batch_v.size(1), device=batch_v.device, dtype=ens_tensor.dtype)
            for b_idx in range(batch_v.size(1)):
                valid_time_for_b = valid_B_mask_sliced[:, b_idx]
                if valid_time_for_b.any():
                    sum_es_per_batch[b_idx] = torch.sum(es_vals[valid_time_for_b, b_idx])
                    sum_norm_per_batch[b_idx] = torch.sum(true_norm_vals[valid_time_for_b, b_idx])
            
            final_batch_mask = valid_B_mask_sliced.any(dim=0)
            if not final_batch_mask.any(): return torch.tensor(0.0, device=batch_v.device, requires_grad=True)
            
            ratios = sum_es_per_batch[final_batch_mask] / (sum_norm_per_batch[final_batch_mask] + 1e-8)
            if return_sum: return torch.sum(ratios)
            return torch.mean(ratios)
            
    elif loss_type == 'kes':
        # Assuming compute_kernel_es is available in scope
        loss_values_per_element = compute_kernel_es(ens_states_timed, true_states_timed, sigma=kes_sigma)
    
    elif loss_type == 'nkes' or loss_type == 'tnkes':
        kes_vals = compute_kernel_es(ens_states_timed, true_states_timed, sigma=kes_sigma)
        true_norm_vals = torch.norm(true_states_timed, p=norm_p, dim=2)
        
        if loss_type == 'nkes':
            loss_values_per_element = kes_vals / (true_norm_vals + 1e-8)
        else: # tnkes
            sum_kes_per_batch = torch.zeros(batch_v.size(1), device=batch_v.device, dtype=ens_tensor.dtype)
            sum_norm_per_batch = torch.zeros(batch_v.size(1), device=batch_v.device, dtype=ens_tensor.dtype)
            for b_idx in range(batch_v.size(1)):
                valid_time_for_b = valid_B_mask_sliced[:, b_idx]
                if valid_time_for_b.any():
                    sum_kes_per_batch[b_idx] = torch.sum(kes_vals[valid_time_for_b, b_idx])
                    sum_norm_per_batch[b_idx] = torch.sum(true_norm_vals[valid_time_for_b, b_idx])

            final_batch_mask = valid_B_mask_sliced.any(dim=0)
            if not final_batch_mask.any(): return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

            ratios = sum_kes_per_batch[final_batch_mask] / (sum_norm_per_batch[final_batch_mask] + 1e-8)
            if return_sum: return torch.sum(ratios)
            return torch.mean(ratios)

    # =========================================================================
    # OPTIMIZED: WPF Loss Types (Ensemble vs Ensemble)
    # =========================================================================
    elif loss_type in ['wpf_ed', 'wpf_fmmd', 'wpf_ammd']:
        if wpf_input is None:
            raise ValueError(f"Loss type {loss_type} requires 'wpf_input' dictionary.")
        
        target_ens = wpf_input['target_ens'] # [T, B, M, D]
        w_ens = wpf_input.get('ens_weights', None) # [T, B, N]
        w_target = wpf_input.get('target_weights', None) # [T, B, M]
        
        # 1. Normalize target ensemble if needed
        if normalize_val is not None:
            target_ens = target_ens / normalize_val.unsqueeze(0).unsqueeze(2)
        
        # 2. Slice time for WPF inputs
        target_ens_timed = target_ens[ignore_first:end_ind + 1, :, :, :]
        
        w_ens_timed = None
        if w_ens is not None:
            w_ens_timed = w_ens[ignore_first:end_ind + 1, :, :]
            
        w_target_timed = None
        if w_target is not None:
            w_target_timed = w_target[ignore_first:end_ind + 1, :, :]
            
        # 3. Vectorization Preparation
        T_slice, B_slice, N, _ = ens_states_timed.shape
        M = target_ens_timed.shape[2]
        
        # Flatten T and B into K = T * B
        x_flat = ens_states_timed.reshape(-1, N, D)       # [K, N, D]
        y_flat = target_ens_timed.reshape(-1, M, D)       # [K, M, D]
        
        w_x_flat = w_ens_timed.reshape(-1, N) if w_ens_timed is not None else None
        w_y_flat = w_target_timed.reshape(-1, M) if w_target_timed is not None else None
        
        # Normalize weights using helper
        K = x_flat.shape[0]
        w_x_norm = _get_batched_weights(w_x_flat, (K, N))
        w_y_norm = _get_batched_weights(w_y_flat, (K, M))
        
        # Get Sigma for Fixed MMD
        sigma_val = wpf_input.get('sigma', kes_sigma)
        
        # Compute Loss (Vectorized)
        # Returns [K] tensor
        flat_loss_values = _compute_batched_loss(
            x_flat, y_flat, w_x_norm, w_y_norm, loss_type, sigma_val
        )
        
        # Reshape back to [T, B]
        loss_values_per_element = flat_loss_values.view(T_slice, B_slice)

    else:
        raise NotImplementedError(f"Loss type '{loss_type}' is not implemented")
    
    # Final Aggregation
    masked_loss_values = loss_values_per_element[valid_B_mask_sliced]
    if masked_loss_values.numel() == 0:
        return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

    if return_sum:
        return torch.sum(masked_loss_values)
    else:
        return torch.mean(masked_loss_values)

class MultiLossUncertaintyWeight(nn.Module):
    def __init__(self, num_losses):
        super(MultiLossUncertaintyWeight, self).__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_losses)) # Learnable log(variance) for each loss

    def forward(self, losses): # losses: list or tensor of individual losses
        total_loss = 0
        for i, loss_val in enumerate(losses):
            precision = torch.exp(-self.log_sigma[i]) # Corresponds to 1/sigma^2
            total_loss += precision * loss_val + 0.5 * self.log_sigma[i] # Maximize likelihood formulation
        return total_loss

def _sqrt_newton_schulz(A: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
    """
    Computes the matrix square root of a batch of positive definite matrices.
    Uses the Denman-Beavers iteration (also known as Newton-Schulz iteration)
    for numerical computation.

    Args:
        A (torch.Tensor): The input batch of positive definite matrices of shape (..., d, d).
        num_iters (int): The number of iterations.

    Returns:
        torch.Tensor: The matrix square root of A, with the same shape as A.
    """
    X = A.clone()
    for _ in range(num_iters):
        X_inv = torch.inverse(X)
        X = 0.5 * (X + A @ X_inv)
    return X


def wasserstein2_multivariate_gaussian(
    mean_true: torch.Tensor,
    cov_true: torch.Tensor,
    mean_sample: torch.Tensor,
    cov_sample: torch.Tensor
) -> torch.Tensor:
    """
    Computes the 2-Wasserstein distance between two batches of multivariate Gaussian distributions.

    This function supports two input shapes for batching:
    1. 3D mean tensor: (T, B, d), where T and B are batch dimensions.
    2. 2D mean tensor: (B, d), where B is the batch dimension.

    The output shape will match the batch dimensions of the input.

    Formula: W_2^2(N_1, N_2) = ||μ_1 - μ_2||_2^2 + Tr(Σ_1 + Σ_2 - 2 * (Σ_1^{1/2} Σ_2 Σ_1^{1/2})^{1/2})

    Args:
        mean_true (torch.Tensor): Means of the true distributions, shape (T, B, d) or (B, d).
        cov_true (torch.Tensor): Covariances of the true distributions, shape (T, B, d, d) or (B, d, d).
        mean_sample (torch.Tensor): Means of the sample distributions, shape (T, B, d) or (B, d).
        cov_sample (torch.Tensor): Covariances of the sample distributions, shape (T, B, d, d) or (B, d, d).

    Returns:
        torch.Tensor: The W_2 distance (not squared) with shape (T, B) or (B,).
    """
    # --- Check input shapes and prepare for batch processing ---
    if mean_true.dim() == 3:  # Shape is (T, B, d)
        batch_shape = mean_true.shape[:2]  # (T, B)
        T, B, d = mean_true.shape
        
        # Flatten batch dimensions for processing
        proc_mean_true = mean_true.view(T * B, d)
        proc_cov_true = cov_true.view(T * B, d, d)
        proc_mean_sample = mean_sample.view(T * B, d)
        proc_cov_sample = cov_sample.view(T * B, d, d)

    elif mean_true.dim() == 2:  # Shape is (B, d)
        batch_shape = mean_true.shape[:1] # (B,)
        B, d = mean_true.shape
        
        # Inputs are already in the correct batch format
        proc_mean_true = mean_true
        proc_cov_true = cov_true
        proc_mean_sample = mean_sample
        proc_cov_sample = cov_sample
    else:
        raise ValueError(
            f"Unsupported input shape. Expected a mean tensor of shape (T, B, d) or (B, d), "
            f"but got {mean_true.shape}."
        )
        
    # --- Core W2 distance calculation ---

    # Mean term: ||μ_1 - μ_2||_2^2
    term_mean = torch.sum((proc_mean_true - proc_mean_sample)**2, dim=1)

    # Covariance term
    # Compute Σ_1^{1/2} using a numerically stable iterative method
    sqrt_cov_true = _sqrt_newton_schulz(proc_cov_true)
    
    # Compute the product M = (Σ_1^{1/2} Σ_2 Σ_1^{1/2})
    cov_prod = sqrt_cov_true @ proc_cov_sample @ sqrt_cov_true
    
    # Compute the square root of the product: M^{1/2}
    sqrt_cov_prod = _sqrt_newton_schulz(cov_prod)

    # Compute the trace of the covariance term
    trace_term = torch.diagonal(proc_cov_true + proc_cov_sample - 2 * sqrt_cov_prod, dim1=-2, dim2=-1).sum(-1)
    
    # The trace term should be non-negative, but can be slightly negative due
    # to numerical errors. Clamp it to zero.
    w2_squared = term_mean + torch.relu(trace_term)
    
    # The final W_2 distance is the square root. Clamp to zero for safety.
    w2_dist = torch.sqrt(torch.relu(w2_squared))

    # Reshape the result to match the original batch dimensions
    return w2_dist.view(*batch_shape)

if __name__ == "__main__":
    pass