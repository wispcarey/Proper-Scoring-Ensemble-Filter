import torch
import matplotlib.pyplot as plt
import numpy as np

def pairwise_distances(A, B=None, domain=None):
    if B is None:
        B = A

    A = torch.atleast_2d(torch.as_tensor(A))
    B = torch.atleast_2d(torch.as_tensor(B))
    mA, nA = A.shape
    mB, nB = B.shape
    assert nA == nB, "The last axis of A and B must have equal length."

    d = A[:, None] - B

    if domain is not None:
        domain = torch.reshape(
            torch.as_tensor(domain, device=d.device, dtype=d.dtype),
            (1, 1, -1),
        )
        d = torch.abs(d)
        d = torch.minimum(d, domain - d)

    distances = torch.sqrt((d * d).sum(dim=-1))

    return distances.reshape(mA, mB)



def dist2coeff(dists, radius, tag=None):
    if radius is None:
        return torch.ones_like(dists)
    
    coeffs = torch.zeros(dists.shape, device=dists.device)

    if tag is None:
        tag = "GC"

    if tag == "Gauss":
        R = radius
        coeffs = torch.exp(-0.5 * (dists / R) ** 2)
    elif tag == "Exp":
        R = radius
        coeffs = torch.exp(-0.5 * (dists / R) ** 3)
    elif tag == "Cubic":
        R = radius * 1.87
        inds = dists <= R
        coeffs[inds] = (1 - (dists[inds] / R) ** 3) ** 3
    elif tag == "Quadro":
        R = radius * 1.64
        inds = dists <= R
        coeffs[inds] = (1 - (dists[inds] / R) ** 4) ** 4
    elif tag == "GC":
        R = radius * 1.82
        ind1 = dists <= R
        r2 = (dists[ind1] / R) ** 2
        r3 = (dists[ind1] / R) ** 3
        coeffs[ind1] = 1 + r2 * (-r3 / 4 + r2 / 2) + r3 * (5 / 8) - r2 * (5 / 3)
        ind2 = torch.logical_and(R < dists, dists <= 2 * R)
        r1 = dists[ind2] / R
        r2 = (dists[ind2] / R) ** 2
        r3 = (dists[ind2] / R) ** 3
        coeffs[ind2] = (
            r2 * (r3 / 12 - r2 / 2)
            + r3 * (5 / 8)
            + r2 * (5 / 3)
            - r1 * 5
            + 4
            - (2 / 3) / r1
        )
    elif tag == "Step":
        R = radius
        inds = dists <= R
        coeffs[inds] = 1
    else:
        raise KeyError("No such coeff function.")

    return coeffs

def create_loc_mat(dist_map, diff_dist, input_tensor):
    B, D = dist_map.shape
    M, N = input_tensor.shape

    # Expand input_tensor to match the shape B*M*N
    input_tensor_expanded = input_tensor.unsqueeze(0).expand(B, M, N)

    # Reshape diff_dist for broadcasting and comparison
    diff_dist_expanded = diff_dist.view(1, D, 1, 1)

    # Create a mask by comparing input_tensor_expanded with diff_dist
    mask = input_tensor_expanded.unsqueeze(1) == diff_dist_expanded

    # Expand dist_map to match the dimensions of the mask
    dist_map_expanded = dist_map.view(B, D, 1, 1)

    # Use the mask to select and replace values in the output tensor
    loc_mat = torch.sum(mask * dist_map_expanded, dim=1)

    return loc_mat

def plot_GC(radius, start_val, end_val, make_plot=True):
    dists = torch.linspace(start_val, end_val, 500)
    
    coeffs = torch.zeros_like(dists)
    
    R = radius * 1.82
    
    ind1 = dists <= R
    r2 = (dists[ind1] / R) ** 2
    r3 = (dists[ind1] / R) ** 3
    coeffs[ind1] = 1 + r2 * (-r3 / 4 + r2 / 2) + r3 * (5 / 8) - r2 * (5 / 3)
    
    ind2 = torch.logical_and(R < dists, dists <= 2 * R)
    r1 = dists[ind2] / R
    r2 = (dists[ind2] / R) ** 2
    r3 = (dists[ind2] / R) ** 3
    coeffs[ind2] = (
        r2 * (r3 / 12 - r2 / 2)
        + r3 * (5 / 8)
        + r2 * (5 / 3)
        - r1 * 5
        + 4
        - (2 / 3) / r1
    )
    
    dists_np = dists.numpy()
    coeffs_np = coeffs.numpy()
    
    if make_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(dists_np, coeffs_np, label=f'Radius = {radius}')
        plt.title('Function Plot of Coefficients for GC')
        plt.xlabel('Distance')
        plt.ylabel('Coefficients')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return dists_np, coeffs_np
