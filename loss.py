import torch
import torch.nn as nn
import math
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

def compute_root_mean_variance(ens_states, unbiased=False, eps=1e-8):
    """
    Computes root mean variance (RMV) of an ensemble.

    RMV_t,b = sqrt( mean_d Var_n[ X_t,b,n,d ] )

    Args:
        ens_states (torch.Tensor): Ensemble predictions. Shape: [T, B, N, D].
        unbiased (bool): Use unbiased variance estimate across ensemble members.
        eps (float): Numerical floor before sqrt.

    Returns:
        torch.Tensor: RMV per (T, B). Shape: [T, B].
    """
    if ens_states.ndim != 4:
        raise ValueError(f"ens_states must have shape [T, B, N, D], got {tuple(ens_states.shape)}")

    T, B, N, _ = ens_states.shape
    if N <= 1:
        return torch.zeros(T, B, device=ens_states.device, dtype=ens_states.dtype)

    ens_var = torch.var(ens_states, dim=2, unbiased=unbiased)  # [T, B, D]
    mean_var = torch.mean(ens_var, dim=-1)                     # [T, B]
    return torch.sqrt(torch.clamp(mean_var, min=eps))


def compute_spread_error_ratio(ens_states, true_states, unbiased=False, eps=1e-8):
    """
    Computes spread-error ratio (SER) per (T, B).

    SER_t,b = RMV_t,b / RMSE_t,b,
    where RMSE_t,b is based on ensemble mean error against truth.

    Args:
        ens_states (torch.Tensor): Ensemble predictions. Shape: [T, B, N, D].
        true_states (torch.Tensor): Ground truth. Shape: [T, B, D].
        unbiased (bool): Use unbiased variance estimate for RMV.
        eps (float): Numerical stabilizer.

    Returns:
        torch.Tensor: SER per (T, B). Shape: [T, B].
    """
    if ens_states.ndim != 4 or true_states.ndim != 3:
        raise ValueError(
            f"Expected ens_states [T,B,N,D] and true_states [T,B,D], got "
            f"{tuple(ens_states.shape)} and {tuple(true_states.shape)}"
        )

    spread = compute_root_mean_variance(ens_states, unbiased=unbiased, eps=eps)  # [T, B]
    ens_mean = torch.mean(ens_states, dim=2)                                      # [T, B, D]
    err = torch.sqrt(torch.mean((ens_mean - true_states) ** 2, dim=-1) + eps)     # [T, B]
    return spread / (err + eps)


def compute_spread_error_ratio_minus_1(ens_states, true_states, unbiased=False, eps=1e-8):
    """
    Computes |SER - 1| per (T, B), where SER is the spread-error ratio.

    This preserves the per-step difference from ideal calibration before any
    temporal averaging is applied.

    Args:
        ens_states (torch.Tensor): Ensemble predictions. Shape: [T, B, N, D].
        true_states (torch.Tensor): Ground truth. Shape: [T, B, D].
        unbiased (bool): Use unbiased variance estimate for RMV.
        eps (float): Numerical stabilizer.

    Returns:
        torch.Tensor: |SER - 1| per (T, B). Shape: [T, B].
    """
    ser = compute_spread_error_ratio(
        ens_states=ens_states,
        true_states=true_states,
        unbiased=unbiased,
        eps=eps,
    )
    return torch.abs(ser - 1.0)


def sample_projection_directions(state_dim, num_projections, device=None, dtype=None, seed=None):
    """
    Sample fixed projection directions on the unit sphere for rank-hist evaluation.

    Args:
        state_dim (int): State dimension D.
        num_projections (int): Number of projection directions P.
        device: Torch device for returned tensor.
        dtype: Torch dtype for returned tensor.
        seed (int or None): Optional deterministic seed.

    Returns:
        torch.Tensor: Projection directions with shape [P, D].
    """
    if state_dim <= 0:
        raise ValueError(f"state_dim must be positive, got {state_dim}")
    if num_projections <= 0:
        raise ValueError(f"num_projections must be positive, got {num_projections}")

    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")

    if state_dim == 1:
        return torch.ones((num_projections, 1), device=device, dtype=dtype)

    cpu_gen = None
    if seed is not None:
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(int(seed))

    dirs_cpu = torch.randn((num_projections, state_dim), generator=cpu_gen, device="cpu", dtype=torch.float32)
    dirs_cpu = dirs_cpu / torch.clamp(torch.norm(dirs_cpu, dim=1, keepdim=True), min=1e-12)
    return dirs_cpu.to(device=device, dtype=dtype)


def compute_quantile_crps_1d(ens_samples, truth_quantiles, quantile_probs):
    """
    Compute 1D distribution-vs-distribution CRPS divergence from quantile-encoded truth.

    This computes:
        E|X - Y| - 0.5 E|X - X'| - 0.5 E|Y - Y'|
    where X, X' are empirical ensemble samples and Y, Y' are from the truth distribution
    represented by quantiles Q(p). The truth expectations are approximated with quadrature
    weights derived from `quantile_probs`.

    Args:
        ens_samples (torch.Tensor): Shape [..., N], ensemble samples.
        truth_quantiles (torch.Tensor): Shape [..., K], truth quantile values.
        quantile_probs (torch.Tensor): Shape [K], quantile levels in [0, 1].

    Returns:
        torch.Tensor: Shape [...], non-negative divergence values (NaN where inputs are invalid).
    """
    if ens_samples.ndim < 1 or truth_quantiles.ndim < 1:
        raise ValueError("ens_samples and truth_quantiles must have at least 1 dimension.")
    if ens_samples.shape[:-1] != truth_quantiles.shape[:-1]:
        raise ValueError(
            f"Leading shape mismatch: ens_samples {tuple(ens_samples.shape)} vs "
            f"truth_quantiles {tuple(truth_quantiles.shape)}"
        )
    if quantile_probs.ndim != 1:
        raise ValueError(f"quantile_probs must be 1D, got {tuple(quantile_probs.shape)}")
    if truth_quantiles.shape[-1] != quantile_probs.shape[0]:
        raise ValueError(
            f"Quantile length mismatch: truth_quantiles K={truth_quantiles.shape[-1]} vs "
            f"quantile_probs K={quantile_probs.shape[0]}"
        )

    ens = ens_samples.to(torch.float32)
    q_true = truth_quantiles.to(torch.float32)
    probs = quantile_probs.to(device=ens.device, dtype=torch.float32)

    valid = torch.isfinite(ens).all(dim=-1) & torch.isfinite(q_true).all(dim=-1)
    if not valid.any():
        return torch.full(ens.shape[:-1], float("nan"), device=ens.device, dtype=torch.float32)

    order = torch.argsort(probs)
    probs_sorted = probs.index_select(0, order)
    q_sorted = q_true.index_select(-1, order)

    ens_safe = torch.where(torch.isfinite(ens), ens, torch.zeros_like(ens))
    q_safe = torch.where(torch.isfinite(q_sorted), q_sorted, torch.zeros_like(q_sorted))

    # Quadrature weights over p from trapz-style rule.
    k = int(probs_sorted.numel())
    if k <= 1:
        quad_w = torch.ones_like(probs_sorted)
    else:
        quad_w = torch.zeros_like(probs_sorted)
        quad_w[0] = 0.5 * (probs_sorted[1] - probs_sorted[0])
        quad_w[-1] = 0.5 * (probs_sorted[-1] - probs_sorted[-2])
        if k > 2:
            quad_w[1:-1] = 0.5 * (probs_sorted[2:] - probs_sorted[:-2])
        quad_sum = torch.sum(quad_w)
        if torch.isfinite(quad_sum) and quad_sum > 1e-12:
            quad_w = quad_w / quad_sum
        else:
            quad_w = torch.full_like(probs_sorted, 1.0 / float(max(k, 1)))

    # E|X-Y| using weighted truth quantile support.
    term_xy_by_q = torch.abs(ens_safe.unsqueeze(-1) - q_safe.unsqueeze(-2)).mean(dim=-2)  # [..., K]
    w_shape = [1] * (term_xy_by_q.ndim - 1) + [k]
    quad_w_b = quad_w.view(*w_shape)
    term_xy = torch.sum(term_xy_by_q * quad_w_b, dim=-1)  # [...]

    # 0.5 * E|X-X'|
    term_xx = 0.5 * torch.abs(ens_safe.unsqueeze(-1) - ens_safe.unsqueeze(-2)).mean(dim=(-2, -1))  # [...]

    # 0.5 * E|Y-Y'| with quadrature weights.
    q_pair = torch.abs(q_safe.unsqueeze(-1) - q_safe.unsqueeze(-2))  # [..., K, K]
    w_i = quad_w.view(*([1] * (q_pair.ndim - 2)), k, 1)
    w_j = quad_w.view(*([1] * (q_pair.ndim - 2)), 1, k)
    term_yy = 0.5 * torch.sum(q_pair * (w_i * w_j), dim=(-2, -1))  # [...]

    crps_div = term_xy - term_xx - term_yy
    crps_div = torch.clamp(crps_div, min=0.0)
    crps_div = crps_div.masked_fill(~valid, float("nan"))
    return crps_div


def compute_projected_quantile_crps_components(
    ens_states,
    state_truth_quantiles,
    quantile_probs,
    pca_truth_quantiles=None,
    pca_directions=None,
    pca_projection_offsets=None,
    max_state_dims=3,
    max_pca_dims=3,
):
    """
    Compute projected CRPS divergence components over:
    - first `max_state_dims` state coordinates
    - up to `max_pca_dims` PCA projections (if quantiles + directions are provided)

    Args:
        ens_states (torch.Tensor): [T, B, N, D]
        state_truth_quantiles (torch.Tensor): [T, B, Ds, K]
        quantile_probs (torch.Tensor): [K]
        pca_truth_quantiles (torch.Tensor or None): [T, B, P, K]
        pca_directions (torch.Tensor or None): [T, B, D, P] or [T, B, P, D]
        pca_projection_offsets (torch.Tensor or None): [T, B, D], offsets subtracted
            from ens_states before PCA projection to match centered PCA-score quantiles.

    Returns:
        dict with keys:
            - state_scores: [S, T, B]
            - pca_scores: [P, T, B]
            - state_mean: [T, B]
            - pca_mean: [T, B]
            - overall_mean: [T, B]
    """
    if ens_states.ndim != 4:
        raise ValueError(f"ens_states must be [T,B,N,D], got {tuple(ens_states.shape)}")
    if state_truth_quantiles.ndim != 4:
        raise ValueError(
            f"state_truth_quantiles must be [T,B,Ds,K], got {tuple(state_truth_quantiles.shape)}"
        )
    if ens_states.shape[0] != state_truth_quantiles.shape[0] or ens_states.shape[1] != state_truth_quantiles.shape[1]:
        raise ValueError(
            f"Time/batch mismatch: ens_states {tuple(ens_states.shape[:2])} vs "
            f"state_truth_quantiles {tuple(state_truth_quantiles.shape[:2])}"
        )
    if state_truth_quantiles.shape[-1] != quantile_probs.shape[0]:
        raise ValueError(
            f"Quantile length mismatch: state_truth_quantiles K={state_truth_quantiles.shape[-1]} vs "
            f"quantile_probs K={quantile_probs.shape[0]}"
        )

    T, B, _, D = ens_states.shape
    state_scores = []
    pca_scores = []

    n_state_dims = min(int(max_state_dims), D, int(state_truth_quantiles.shape[2]))
    for d_idx in range(n_state_dims):
        score_d = compute_quantile_crps_1d(
            ens_samples=ens_states[:, :, :, d_idx],
            truth_quantiles=state_truth_quantiles[:, :, d_idx, :],
            quantile_probs=quantile_probs,
        )
        state_scores.append(score_d)

    if pca_truth_quantiles is not None and pca_directions is not None:
        if pca_truth_quantiles.ndim != 4:
            raise ValueError(
                f"pca_truth_quantiles must be [T,B,P,K], got {tuple(pca_truth_quantiles.shape)}"
            )
        if pca_truth_quantiles.shape[:2] != (T, B):
            raise ValueError(
                f"pca_truth_quantiles time/batch mismatch: expected {(T, B)}, got {tuple(pca_truth_quantiles.shape[:2])}"
            )
        if pca_truth_quantiles.shape[-1] != quantile_probs.shape[0]:
            raise ValueError(
                f"Quantile length mismatch for PCA: K={pca_truth_quantiles.shape[-1]} vs "
                f"{quantile_probs.shape[0]}"
            )

        dirs = pca_directions
        if dirs.ndim != 4:
            raise ValueError(f"pca_directions must be 4D, got {tuple(dirs.shape)}")
        if dirs.shape[:2] != (T, B):
            raise ValueError(
                f"pca_directions time/batch mismatch: expected {(T, B)}, got {tuple(dirs.shape[:2])}"
            )
        if dirs.shape[2] == D:
            dirs_tb_dp = dirs
        elif dirs.shape[3] == D:
            dirs_tb_dp = dirs.transpose(-1, -2)
        else:
            raise ValueError(
                f"pca_directions must have state dim D={D} in axis 2 or 3, got {tuple(dirs.shape)}"
            )

        dirs_tb_dp = dirs_tb_dp.to(device=ens_states.device, dtype=ens_states.dtype)
        dirs_tb_dp = dirs_tb_dp / torch.clamp(torch.norm(dirs_tb_dp, dim=2, keepdim=True), min=1e-12)

        ens_for_proj = ens_states
        if pca_projection_offsets is not None:
            offsets = torch.as_tensor(
                pca_projection_offsets,
                device=ens_states.device,
                dtype=ens_states.dtype,
            )
            if offsets.shape != (T, B, D):
                raise ValueError(
                    f"pca_projection_offsets must be [T,B,D]={ (T, B, D) }, got {tuple(offsets.shape)}"
                )
            ens_for_proj = ens_states - offsets.unsqueeze(2)

        n_pca_dims = min(int(max_pca_dims), int(pca_truth_quantiles.shape[2]), int(dirs_tb_dp.shape[3]))
        if n_pca_dims > 0:
            proj = torch.einsum("tbnd,tbdp->tbnp", ens_for_proj, dirs_tb_dp[:, :, :, :n_pca_dims])
            for p_idx in range(n_pca_dims):
                score_p = compute_quantile_crps_1d(
                    ens_samples=proj[:, :, :, p_idx],
                    truth_quantiles=pca_truth_quantiles[:, :, p_idx, :],
                    quantile_probs=quantile_probs,
                )
                pca_scores.append(score_p)

    empty_scores = torch.empty((0, T, B), device=ens_states.device, dtype=torch.float32)
    state_scores_stacked = (
        torch.stack(state_scores, dim=0).to(torch.float32) if len(state_scores) > 0 else empty_scores
    )
    pca_scores_stacked = (
        torch.stack(pca_scores, dim=0).to(torch.float32) if len(pca_scores) > 0 else empty_scores
    )

    if state_scores_stacked.shape[0] > 0:
        state_mean = torch.nanmean(state_scores_stacked, dim=0)
    else:
        state_mean = torch.full((T, B), float("nan"), device=ens_states.device, dtype=torch.float32)

    if pca_scores_stacked.shape[0] > 0:
        pca_mean = torch.nanmean(pca_scores_stacked, dim=0)
    else:
        pca_mean = torch.full((T, B), float("nan"), device=ens_states.device, dtype=torch.float32)

    score_parts = []
    if state_scores_stacked.shape[0] > 0:
        score_parts.append(state_scores_stacked)
    if pca_scores_stacked.shape[0] > 0:
        score_parts.append(pca_scores_stacked)

    if len(score_parts) == 0:
        overall_mean = torch.full((T, B), float("nan"), device=ens_states.device, dtype=torch.float32)
    else:
        overall_mean = torch.nanmean(torch.cat(score_parts, dim=0), dim=0)

    return {
        "state_scores": state_scores_stacked,
        "pca_scores": pca_scores_stacked,
        "state_mean": state_mean,
        "pca_mean": pca_mean,
        "overall_mean": overall_mean,
    }


def compute_projected_quantile_crps(
    ens_states,
    state_truth_quantiles,
    quantile_probs,
    pca_truth_quantiles=None,
    pca_directions=None,
    pca_projection_offsets=None,
    max_state_dims=3,
    max_pca_dims=3,
):
    """
    Compute projected CRPS divergence averaged over available state/PCA projections.

    Returns:
        torch.Tensor: [T, B], mean projected divergence over available projections.
    """
    components = compute_projected_quantile_crps_components(
        ens_states=ens_states,
        state_truth_quantiles=state_truth_quantiles,
        quantile_probs=quantile_probs,
        pca_truth_quantiles=pca_truth_quantiles,
        pca_directions=pca_directions,
        pca_projection_offsets=pca_projection_offsets,
        max_state_dims=max_state_dims,
        max_pca_dims=max_pca_dims,
    )
    return components["overall_mean"]


def compute_ensemble_rank_histogram(
    ens_states,
    true_states,
    projection_directions=None,
    num_projections=8,
    tie_break="random",
    seed=0,
):
    """
    Compute rank histogram statistics for ensemble calibration checks.

    Procedure:
    1. Project high-dimensional ensemble and truth to 1D using fixed directions.
    2. Compute rank r = # {x_n < y} in each projected sample (with tie handling).
    3. Aggregate counts over all (time, batch, projection). Rank bins are 0..N.

    Args:
        ens_states (torch.Tensor): Ensemble predictions [T, B, N, D].
        true_states (torch.Tensor): Truth [T, B, D].
        projection_directions (torch.Tensor or None): [P, D]. If None, sampled internally.
        num_projections (int): Used only when projection_directions is None.
        tie_break (str): "random" or "midpoint" for ties.
        seed (int): RNG seed for random tie-breaking and direction sampling.

    Returns:
        dict with keys:
            - counts: Tensor [N+1] (CPU, int64)
            - probabilities: Tensor [N+1] (CPU, float32)
            - total_samples: int
            - num_bins: int
            - num_projections: int
            - freq_var: float
    """
    if ens_states.ndim != 4 or true_states.ndim != 3:
        raise ValueError(
            f"Expected ens_states [T,B,N,D] and true_states [T,B,D], got "
            f"{tuple(ens_states.shape)} and {tuple(true_states.shape)}"
        )
    if tie_break not in {"random", "midpoint"}:
        raise ValueError(f"tie_break must be 'random' or 'midpoint', got '{tie_break}'")

    T, B, N, D = ens_states.shape
    if true_states.shape != (T, B, D):
        raise ValueError(
            f"Shape mismatch: ens_states is {(T, B, N, D)} but true_states is {tuple(true_states.shape)}"
        )

    if projection_directions is None:
        projection_directions = sample_projection_directions(
            state_dim=D,
            num_projections=num_projections,
            device=ens_states.device,
            dtype=ens_states.dtype,
            seed=seed,
        )
    else:
        if projection_directions.ndim != 2 or projection_directions.shape[1] != D:
            raise ValueError(
                f"projection_directions must have shape [P, {D}], got {tuple(projection_directions.shape)}"
            )
        projection_directions = projection_directions.to(device=ens_states.device, dtype=ens_states.dtype)
        projection_directions = projection_directions / torch.clamp(
            torch.norm(projection_directions, dim=1, keepdim=True),
            min=1e-12,
        )

    P = int(projection_directions.shape[0])

    # Keep only finite samples before projection.
    finite_mask = torch.isfinite(true_states).all(dim=-1) & torch.isfinite(ens_states).all(dim=(2, 3))  # [T, B]
    if not finite_mask.any():
        empty_counts = torch.zeros(N + 1, dtype=torch.int64)
        empty_probs = torch.full((N + 1,), float("nan"), dtype=torch.float32)
        return {
            "counts": empty_counts,
            "probabilities": empty_probs,
            "total_samples": 0,
            "num_bins": int(N + 1),
            "num_projections": P,
            "freq_var": float("nan"),
        }

    ens_valid = ens_states[finite_mask]    # [K, N, D]
    true_valid = true_states[finite_mask]  # [K, D]
    K = ens_valid.shape[0]

    tie_gen = None
    if tie_break == "random":
        tie_gen = torch.Generator(device="cpu")
        tie_gen.manual_seed(int(seed))

    counts = torch.zeros(N + 1, device=ens_states.device, dtype=torch.int64)

    for p in range(P):
        direction = projection_directions[p]                  # [D]
        ens_1d = torch.matmul(ens_valid, direction)          # [K, N]
        true_1d = torch.matmul(true_valid, direction)        # [K]
        true_col = true_1d.unsqueeze(1)                      # [K, 1]

        rank_low = torch.sum(ens_1d < true_col, dim=1)       # [K]
        rank_high = torch.sum(ens_1d <= true_col, dim=1)     # [K]

        if tie_break == "random":
            span = (rank_high - rank_low).to(torch.float32)  # [K], includes 0 when no tie
            rand_cpu = torch.rand((K,), generator=tie_gen, device="cpu", dtype=torch.float32)
            offset = torch.floor(rand_cpu * (span.cpu() + 1.0)).to(torch.long).to(rank_low.device)
            ranks = rank_low + offset
        else:
            ranks = torch.div(rank_low + rank_high, 2, rounding_mode="floor")

        counts += torch.bincount(ranks, minlength=N + 1)

    total_samples = int(counts.sum().item())
    probs = counts.to(torch.float32) / max(total_samples, 1)
    if total_samples <= 0:
        freq_var = float("nan")
    else:
        freq_var = float(torch.var(probs, unbiased=False).item())

    return {
        "counts": counts.detach().cpu(),
        "probabilities": probs.detach().cpu(),
        "total_samples": total_samples,
        "num_bins": int(N + 1),
        "num_projections": P,
        "freq_var": freq_var,
    }

#######################################################
# related to the weighted particle filter loss functions

import torch

def _get_batched_weights(w, shape_constraints, do_detach=True):
    """
    Internal helper: Normalize weights, optionally detaching them.
    
    Inputs:
        w: (Tensor or None) Weights of shape [K, N]
        shape_constraints: (tuple) (K, N)
        do_detach: (bool) Whether to detach weights from the graph.
        
    Output:
        w_norm: (Tensor) Normalized weights.
    """
    K, N = shape_constraints
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if w is None:
        return torch.full((K, N), 1.0 / N, device=device)
    
    # Ensure shape is [K, N]
    w_view = w.view(K, N)
    
    if do_detach:
        w_view = w_view.detach()
    
    # Normalize: [K, N] / [K, 1]
    w_norm = w_view / (w_view.sum(dim=1, keepdim=True) + 1e-8)
    return w_norm

def _compute_batched_wpf_loss(x, y, w_x, w_y, loss_type, sigma_val=None):
    """
    Internal helper: Computes weighted energy distance or MMD.
    
    Inputs:
        x, y: (Tensor) Data samples.
        w_x, w_y: (Tensor) Normalized weights.
        loss_type: (str) 'wpf_ed', 'wpf_fmmd', 'wpf_ammd'
        sigma_val: (float)
        
    Output:
        loss: (Tensor) Shape [K]
    """
    # 1. Compute Pairwise Distances
    d_xy = torch.cdist(x, y, p=2) 
    d_xx = torch.cdist(x, x, p=2)
    d_yy = torch.cdist(y, y, p=2)

    # 2. Define Kernel Function
    if loss_type == 'wpf_ed':
        k_xy, k_xx, k_yy = d_xy, d_xx, d_yy
    elif loss_type in ['wpf_fmmd', 'wpf_ammd']:
        if loss_type == 'wpf_ammd':
            with torch.no_grad():
                flat_d = d_xy.view(d_xy.shape[0], -1)
                median_d = torch.median(flat_d, dim=1).values.view(-1, 1, 1)
                sigma = median_d.clamp(min=1e-5)
        else:
            sigma = float(sigma_val)

        gamma = 1.0 / (2 * sigma**2)
        k_xy = torch.exp(- (d_xy**2) * gamma)
        k_xx = torch.exp(- (d_xx**2) * gamma)
        k_yy = torch.exp(- (d_yy**2) * gamma)

    # 3. Compute Expectations via BMM
    wx_unsq = w_x.unsqueeze(1)    # [K, 1, N]
    wy_unsq = w_y.unsqueeze(2)    # [K, M, 1]
    wx_unsq_T = w_x.unsqueeze(2)  # [K, N, 1]
    wy_unsq_T = w_y.unsqueeze(2)  # [K, M, 1]

    e_xy = torch.bmm(torch.bmm(wx_unsq, k_xy), wy_unsq).view(-1)
    e_xx = torch.bmm(torch.bmm(wx_unsq, k_xx), wx_unsq_T).view(-1)
    e_yy = torch.bmm(torch.bmm(w_y.unsqueeze(1), k_yy), wy_unsq_T).view(-1)

    if loss_type == 'wpf_ed':
        return 2 * e_xy - e_xx - e_yy
    return e_xx - 2 * e_xy + e_yy

def _compute_batched_nll(h_ens, y_obs, w_ens, sigma_y):
    """
    Internal helper: Computes weighted Negative Log Likelihood.
    
    Inputs:
        h_ens: (Tensor) Projected ensemble [T, B, N, d_obs]
        y_obs: (Tensor) Ground truth observations [T, B, d_obs]
        w_ens: (Tensor or None) Ensemble weights [T, B, N]
        sigma_y: (float or Tensor) Observation noise standard deviation
        
    Output:
        nll: (Tensor) Loss per element [T, B]
    """
    T, B, N, d_obs = h_ens.shape
    y_obs_unsqueezed = y_obs.unsqueeze(2) # [T, B, 1, d_obs]
    
    # 1. Compute squared Mahalanobis distance
    # Supports broadcasting if sigma_y is scalar or [d_obs]
    sq_error = torch.sum(((y_obs_unsqueezed - h_ens) / sigma_y) ** 2, dim=-1) # [T, B, N]
    
    # 2. Handle Weights (Normalized and Detached)
    if w_ens is not None:
        w_flat = w_ens.reshape(-1, N)
        w_norm = _get_batched_weights(w_flat, (T * B, N)).view(T, B, N)
        inner_term = torch.log(w_norm + 1e-12) - 0.5 * sq_error
    else:
        inner_term = -0.5 * sq_error - math.log(N)
    
    return -torch.logsumexp(inner_term, dim=2) 

def compute_loss(ens_tensor, batch_v, loss_type, ignore_first=0, end_ind=None, 
                 valid_B_mask=None, norm_p=1, kes_sigma=1, return_sum=False, 
                 normalize_val=None, additional_inputs=None):
    """
    Computes loss. Supports L2, ES, Kernel ES, WPF-based distances, and NLL.

    Inputs:
        ens_tensor: [T, B, N, D]
        batch_v: [T, B, D]
        loss_type: 'l2', 'rmse', 'rmv', 'ser', 'spread_error_ratio',
                   'ser_minus_1', 'spread_error_ratio_minus_1',
                   'rank_uniform_l1', 'rank_uniform_l2', 'rank_chi2',
                   'es', 'kes', 'wpf_ed', 'wpf_fmmd', 'wpf_ammd', 'nll'
        additional_inputs: (dict)
            - For WPF: 'target_ens', 'ens_weights', 'target_weights', 'sigma'
            - For NLL: 'obs_map', 'sigma_y', 'true_obs', 'ens_weights' (optional)
    """
    _, B, D = batch_v.shape
    
    if normalize_val is not None:
        batch_v = batch_v / normalize_val.unsqueeze(0)
        ens_tensor = ens_tensor / normalize_val.unsqueeze(1).unsqueeze(0)
    
    full_time_steps = batch_v.size(0)
    if end_ind is None:
        end_ind = full_time_steps - 1 
    end_ind = min(end_ind, full_time_steps - 1)

    if ignore_first > end_ind:
        return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

    if valid_B_mask is None:
        valid_B_mask = torch.ones(full_time_steps, B, dtype=torch.bool, device=batch_v.device)
    elif valid_B_mask.ndim == 1:
        valid_B_mask = valid_B_mask.unsqueeze(0).expand(full_time_steps, -1)

    valid_B_mask_sliced = valid_B_mask[ignore_first:end_ind + 1, :]
    ens_states_timed = ens_tensor[ignore_first:end_ind + 1, :, :, :] 
    true_states_timed = batch_v[ignore_first:end_ind + 1, :, :]   
    
    if not valid_B_mask_sliced.any():
        return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

    loss_values_per_element = None 

    if loss_type == "l2":
        ens_mean_timed = torch.mean(ens_states_timed, dim=2)
        loss_values_per_element = torch.sum((ens_mean_timed - true_states_timed) ** 2, dim=2)
    
    elif loss_type == 'nl2':
        ens_mean_timed = torch.mean(ens_states_timed, dim=2)
        error_norm_2 = torch.sum((ens_mean_timed - true_states_timed) ** 2, dim=2)
        true_norm_2 = torch.sum(true_states_timed ** 2, dim=2)
        loss_values_per_element = error_norm_2 / (true_norm_2 + 1e-8)

    elif loss_type == 'nes' or loss_type == 'tnes':
        es_vals = compute_es(ens_states_timed, true_states_timed, norm_p=norm_p)
        true_norm_vals = torch.norm(true_states_timed, p=2, dim=2) ** norm_p
        if loss_type == 'nes':
            loss_values_per_element = es_vals / (true_norm_vals + 1e-8)
        else: # tnes
            sum_es_per_batch = torch.zeros(B, device=batch_v.device, dtype=ens_tensor.dtype)
            sum_norm_per_batch = torch.zeros(B, device=batch_v.device, dtype=ens_tensor.dtype)
            for b_idx in range(B):
                valid_time_for_b = valid_B_mask_sliced[:, b_idx]
                if valid_time_for_b.any():
                    sum_es_per_batch[b_idx] = torch.sum(es_vals[valid_time_for_b, b_idx])
                    sum_norm_per_batch[b_idx] = torch.sum(true_norm_vals[valid_time_for_b, b_idx])
            
            final_batch_mask = valid_B_mask_sliced.any(dim=0)
            if not final_batch_mask.any(): return torch.tensor(0.0, device=batch_v.device, requires_grad=True)
            
            ratios = sum_es_per_batch[final_batch_mask] / (sum_norm_per_batch[final_batch_mask] + 1e-8)
            if return_sum: return torch.sum(ratios)
            return torch.mean(ratios)

    elif loss_type == 'nkes' or loss_type == 'tnkes':
        kes_vals = compute_kernel_es(ens_states_timed, true_states_timed, sigma=kes_sigma)
        true_norm_vals = torch.norm(true_states_timed, p=norm_p, dim=2)
        if loss_type == 'nkes':
            loss_values_per_element = kes_vals / (true_norm_vals + 1e-8)
        else: # tnkes
            sum_kes_per_batch = torch.zeros(B, device=batch_v.device, dtype=ens_tensor.dtype)
            sum_norm_per_batch = torch.zeros(B, device=batch_v.device, dtype=ens_tensor.dtype)
            for b_idx in range(B):
                valid_time_for_b = valid_B_mask_sliced[:, b_idx]
                if valid_time_for_b.any():
                    sum_kes_per_batch[b_idx] = torch.sum(kes_vals[valid_time_for_b, b_idx])
                    sum_norm_per_batch[b_idx] = torch.sum(true_norm_vals[valid_time_for_b, b_idx])

            final_batch_mask = valid_B_mask_sliced.any(dim=0)
            if not final_batch_mask.any(): return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

            ratios = sum_kes_per_batch[final_batch_mask] / (sum_norm_per_batch[final_batch_mask] + 1e-8)
            if return_sum: return torch.sum(ratios)
            return torch.mean(ratios)
            
    elif loss_type == 'rmse':
        ens_mean_timed = torch.mean(ens_states_timed, dim=2)
        loss_values_per_element = torch.sqrt(torch.sum((ens_mean_timed - true_states_timed) ** 2, dim=2) + 1e-8)

    elif loss_type == 'rmv':
        loss_values_per_element = compute_root_mean_variance(ens_states_timed)

    elif loss_type in ['ser', 'spread_error_ratio']:
        loss_values_per_element = compute_spread_error_ratio(ens_states_timed, true_states_timed)

    elif loss_type in ['ser_minus_1', 'spread_error_ratio_minus_1']:
        loss_values_per_element = compute_spread_error_ratio_minus_1(ens_states_timed, true_states_timed)

    elif loss_type in ['rank_uniform_l1', 'rank_uniform_l2', 'rank_chi2']:
        rank_cfg = additional_inputs if isinstance(additional_inputs, dict) else {}
        rank_stats = compute_ensemble_rank_histogram(
            ens_states=ens_states_timed,
            true_states=true_states_timed,
            projection_directions=rank_cfg.get('projection_directions', None),
            num_projections=int(rank_cfg.get('num_projections', 8)),
            tie_break=str(rank_cfg.get('tie_break', 'random')).lower(),
            seed=int(rank_cfg.get('seed', 0)),
        )
        if loss_type == 'rank_uniform_l1':
            rank_metric = rank_stats['uniform_l1']
        elif loss_type == 'rank_uniform_l2':
            rank_metric = rank_stats['uniform_l2']
        else:
            rank_metric = rank_stats['chi2']
        return torch.as_tensor(rank_metric, device=batch_v.device, dtype=ens_tensor.dtype)
    
    elif loss_type == 'es':
        loss_values_per_element = compute_es(ens_states_timed, true_states_timed, norm_p=norm_p)
            
    elif loss_type == 'kes':
        loss_values_per_element = compute_kernel_es(ens_states_timed, true_states_timed, sigma=kes_sigma)

    elif loss_type.startswith('wpf_'):
        # Parse loss type for ST (Stop Gradient) variants
        is_st = loss_type.startswith('wpf_st_')
        real_loss_type = loss_type.replace('wpf_st_', 'wpf_')
        
        if real_loss_type not in ['wpf_ed', 'wpf_fmmd', 'wpf_ammd']:
             raise NotImplementedError(f"WPF Loss type '{loss_type}' is not implemented")

        if additional_inputs is None or 'target_ens' not in additional_inputs:
            raise ValueError(f"Loss {loss_type} requires 'target_ens' in additional_inputs.")
        
        target_ens = additional_inputs['target_ens']
        w_ens = additional_inputs.get('ens_weights', None)
        w_target = additional_inputs.get('target_weights', None)
        
        if normalize_val is not None:
            target_ens = target_ens / normalize_val.unsqueeze(0).unsqueeze(2)
        
        target_ens_timed = target_ens[ignore_first:end_ind + 1, :, :, :]
        w_ens_timed = w_ens[ignore_first:end_ind + 1, :, :] if w_ens is not None else None
        w_target_timed = w_target[ignore_first:end_ind + 1, :, :] if w_target is not None else None
            
        # Handle gradient stop logic based on 'st' flag
        if is_st:
            target_ens_timed = target_ens_timed.detach()
            detach_target_weights = True # Detach BOTH ensemble and target weights
        else:
            detach_target_weights = False # Gradients flow through everything

        T_slice, B_slice, N, _ = ens_states_timed.shape
        M = target_ens_timed.shape[2]
        
        x_flat = ens_states_timed.reshape(-1, N, D)
        y_flat = target_ens_timed.reshape(-1, M, D)
        w_x_flat = w_ens_timed.reshape(-1, N) if w_ens_timed is not None else None
        w_y_flat = w_target_timed.reshape(-1, M) if w_target_timed is not None else None
        
        K_total = x_flat.shape[0]
        
        # Normalize and optionally detach weights
        w_x_norm = _get_batched_weights(w_x_flat, (K_total, N), do_detach=False)
        w_y_norm = _get_batched_weights(w_y_flat, (K_total, M), do_detach=detach_target_weights)
        
        sigma_val = additional_inputs.get('sigma', kes_sigma)
        flat_loss_values = _compute_batched_wpf_loss(x_flat, y_flat, w_x_norm, w_y_norm, real_loss_type, sigma_val)
        loss_values_per_element = flat_loss_values.view(T_slice, B_slice)

    elif loss_type == 'nll':
        if additional_inputs is None:
            raise ValueError("NLL requires additional_inputs.")
        
        # Validation
        for k in ['obs_map', 'sigma_y', 'true_obs']:
            if k not in additional_inputs:
                raise ValueError(f"NLL requires '{k}' in additional_inputs.")
        
        h_ens = additional_inputs['obs_map'](ens_states_timed)
        y_obs = additional_inputs['true_obs'][ignore_first:end_ind + 1, :, :]
        w_ens = additional_inputs.get('ens_weights', None)
        if w_ens is not None:
            w_ens = w_ens[ignore_first:end_ind + 1, :, :]
        
        loss_values_per_element = _compute_batched_nll(
            h_ens, y_obs, w_ens, additional_inputs['sigma_y']
        )
    
    elif loss_type == 'nnll':
        if additional_inputs is None:
            raise ValueError("NNLL requires additional_inputs.")
        
        h_ens = additional_inputs['obs_map'](ens_states_timed)
        y_obs = additional_inputs['true_obs'][ignore_first:end_ind + 1, :, :]
        w_ens = additional_inputs.get('ens_weights', None)
        if w_ens is not None:
            w_ens = w_ens[ignore_first:end_ind + 1, :, :]
        
        nll_vals = _compute_batched_nll(
            h_ens, y_obs, w_ens, additional_inputs['sigma_y']
        )

        # y_obs shape: [T_slice, B_slice, d_obs]
        y_norm = torch.norm(y_obs, p=2, dim=-1) # Shape: [T_slice, B_slice]
        loss_values_per_element = nll_vals / (y_norm + 1e-8)

    else:
        raise NotImplementedError(f"Loss type '{loss_type}' is not implemented")
    
    masked_loss_values = loss_values_per_element[valid_B_mask_sliced]
    if masked_loss_values.numel() == 0:
        return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

    if return_sum:
        return torch.sum(masked_loss_values)
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
