import torch

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

def StochasticENKF_analysis(ens_v_f, H, sigma_y, obs_y):
    B, N, d = ens_v_f.shape
    
    if H.dim() == 2:
        H = H.unsqueeze(0).expand(B, -1, -1)

    ens_mean_f = ens_v_f.mean(dim=1, keepdim=True)  
    v_a = ens_v_f - ens_mean_f
    y_a = torch.bmm(v_a, H.transpose(1, 2))

    Cyy = torch.bmm(y_a.transpose(1,2), y_a) + sigma_y ** 2 * (N-1) * torch.eye(H.shape[1], device=ens_v_f.device).unsqueeze(0).expand(B, -1, -1)
    H_invCyy = torch.linalg.solve(Cyy, H).transpose(1, 2)
    K = torch.bmm(v_a.transpose(1,2), torch.bmm(v_a, H_invCyy))

    y_samps = torch.bmm(ens_v_f, H.transpose(1, 2))
    y_samps += sigma_y * torch.randn_like(y_samps, device=ens_v_f.device)

    ens_v_a = ens_v_f - torch.bmm(y_samps - obs_y, K.transpose(1, 2))

    return ens_v_a, K

def EnKF_analysis(E, Eo, y_true, sigma_y, a_method="PertObs"):
    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(1)

    B, N, D = E.shape
    N1 = N - 1
    
    mu = torch.mean(E, dim=1, keepdim=True)
    A = E - mu  # Ens anomalies
    xo = torch.mean(Eo, dim=1, keepdim=True)
    Y = Eo - xo  # Obs ens anomalies
    dy = y_true - xo  # Mean "innovation"

    if "PertObs" in a_method:
        C = torch.bmm(Y.transpose(1, 2), Y) + sigma_y ** 2 * N1 * torch.eye(Eo.shape[2], device=Eo.device).unsqueeze(0).expand(B, -1, -1)
        D = mean0(sigma_y * torch.randn_like(Eo, device=Eo.device))
        YinvC = mrdiv(Y, C)
        KG = torch.bmm(A.transpose(1,2), YinvC)
        dE = torch.bmm(y_true - D - Eo, KG.transpose(1, 2))
        E = E + dE

        return E, KG
    
    elif "Sqrt" in a_method:
        # if a_method == "Sqrt" and N > D:
        #     a_method = "Sqrt svd"
        
        # if "svd" in a_method:
        #     # SVD method
        #     V, s, _ = torch.linalg.svd(Y / sigma_y, full_matrices=False)
        #     d = pad0(s**2, N).unsqueeze(2) + N1
            
        #     Pw = torch.bmm(V * d ** (-1), V.transpose(1, 2))
        #     T = torch.bmm(V * d ** (-0.5), V.transpose(1, 2)) * torch.sqrt(torch.tensor(N1))
        # else:
        #     # Eigenvalue decomposition method
        #     d, V = torch.linalg.eigh(torch.bmm(Y, Y.transpose(1, 2)) / sigma_y ** 2 + N1 * torch.eye(N, device=Eo.device).unsqueeze(0).expand(B, -1, -1))
        #     T = torch.bmm(V, torch.diag_embed(d ** -0.5)).bmm(V.transpose(1, 2)) * torch.sqrt(torch.tensor(N1))
        #     Pw = torch.bmm(V, torch.diag_embed(d ** -1)).bmm(V.transpose(1, 2))
        
        # Eigenvalue decomposition method
        d, V = torch.linalg.eigh(torch.bmm(Y, Y.transpose(1, 2)) / sigma_y ** 2 + N1 * torch.eye(N, device=Eo.device).unsqueeze(0).expand(B, -1, -1))
        T = torch.bmm(V, torch.diag_embed(d ** -0.5)).bmm(V.transpose(1, 2)) * torch.sqrt(torch.tensor(N1))
        Pw = torch.bmm(V, torch.diag_embed(d ** -1)).bmm(V.transpose(1, 2))

        w = torch.bmm(torch.bmm(dy, Y.transpose(1,2)), Pw) / sigma_y ** 2
        E = mu + torch.bmm(w, A) + torch.bmm(T, A)

        return E, w + T
    
def loc_EnKF_analysis(E, Eo, y_true, sigma_y, Lvy, Lyy, a_method="PertObs"):
    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(1)

    B, N, D = E.shape
    N1 = N - 1
    d = Eo.shape[2]

    mu = torch.mean(E, dim=1, keepdim=True)
    A = E - mu  # Ensemble anomalies
    xo = torch.mean(Eo, dim=1, keepdim=True)
    Y = Eo - xo  # Observation ensemble anomalies
    dy = y_true - xo  # Mean innovation

    if "PertObs" in a_method:
        # Compute sample covariances
        C_vy = torch.bmm(A.transpose(1, 2), Y)  # (B, D, d)
        # Apply localization to C_vy
        C_vy = C_vy * Lvy.unsqueeze(0)  # (B, D, d) * (1, D, d)

        # Compute localized C_yy
        C_yy = torch.bmm(Y.transpose(1, 2), Y) * Lyy.unsqueeze(0) + sigma_y ** 2 * N1 * torch.eye(d, device=Eo.device).unsqueeze(0)

        # Add observation perturbations
        D = mean0(sigma_y * torch.randn_like(Eo))

        # Solve for Kalman gain: K = C_vy @ C_yy^{-1}
        K = torch.linalg.solve(C_yy, C_vy.transpose(1, 2)).transpose(1, 2)  # (B, D, d)

        # Update ensemble
        E = E + torch.bmm(y_true - D - Eo, K.transpose(1, 2))

        return E, K

    else:
        raise NotImplemented

            
        
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

if __name__ == "__main__":
    def compare_svd_eig(Y, sigma_y, N1, iterations=10):
        B, N, D = Y.shape

        svd_diff_T = []
        svd_diff_Pw = []

        # SVD method
        V_svd, s_svd, _ = torch.linalg.svd(Y / sigma_y, full_matrices=False)
        d_svd = pad0(s_svd**2, N).unsqueeze(2) + N1
        
        Pw_svd = torch.bmm(V_svd * d_svd ** (-1), V_svd.transpose(1, 2))
        T_svd = torch.bmm(V_svd * d_svd ** (-0.5), V_svd.transpose(1, 2)) * torch.sqrt(torch.tensor(N1))

        # Eigenvalue decomposition method
        d_eig, V_eig = torch.linalg.eigh(torch.bmm(Y, Y.transpose(1, 2)) / sigma_y ** 2 + N1 * torch.eye(N, device=Y.device).unsqueeze(0).expand(B, -1, -1))
        T_eig = torch.bmm(V_eig, torch.diag_embed(d_eig ** -0.5)).bmm(V_eig.transpose(1, 2)) * torch.sqrt(torch.tensor(N1))
        Pw_eig = torch.bmm(V_eig, torch.diag_embed(d_eig ** -1)).bmm(V_eig.transpose(1, 2))

        # Record the difference for each iteration
        diff_T = torch.norm(T_svd - T_eig)
        diff_Pw = torch.norm(Pw_svd - Pw_eig)
        
        svd_diff_T.append(diff_T.item())
        svd_diff_Pw.append(diff_Pw.item())

        # Compute average differences
        avg_diff_T = sum(svd_diff_T) / iterations
        avg_diff_Pw = sum(svd_diff_Pw) / iterations

        print(f"\nAverage T difference over {iterations} iterations: {avg_diff_T}")
        print(f"Average Pw difference over {iterations} iterations: {avg_diff_Pw}")

    # Test parameters
    B, N, D = 3, 6, 2  # Batch size, Ensemble size, Data dimension (N > D)
    sigma_y = 1.0
    N1 = N - 1

    # Number of iterations
    iterations = 10

    # Compare SVD and Eig results multiple times
    for i in range(iterations):
        Y = torch.randn(B, N, D)
        compare_svd_eig(Y, sigma_y, N1, iterations)