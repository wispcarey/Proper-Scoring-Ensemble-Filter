import numpy as np
from scipy.spatial.distance import cdist
import warnings

# Ignore warnings that may arise from calculating cosine distance with zero vectors
warnings.filterwarnings("ignore", category=RuntimeWarning)


def energy_score(X, y, metric='euclidean', p=3):
    """Calculates the energy score for a single point y with respect to a sample X."""
    if y.ndim == 1: y = y.reshape(1, -1)
    if X.ndim == 1: X = X.reshape(-1, 1)
    metric_kwargs = {'p': p} if metric == 'minkowski' else {}
    dist_xy = cdist(X, y, metric=metric, **metric_kwargs)
    dist_xx = cdist(X, X, metric=metric, **metric_kwargs)
    return 2 * np.mean(dist_xy) - np.mean(dist_xx)

def calculate_mean_energy_score(X, Y, metric='euclidean', p=3):
    """Calculates the mean energy score of each point in sample Y with respect to sample X."""
    
    metric_kwargs = {'p': p} if metric == 'minkowski' else {}
    ess = np.zeros(len(Y))
    for i in range(Y.shape[0]):
        y = Y[i]
        div_norm = cdist(y[None,:], np.zeros_like(y)[None,:], metric=metric, **metric_kwargs)
        # div_norm = cdist(np.mean(X,axis=0)[None,:], np.zeros_like(y)[None,:], metric=metric, **metric_kwargs)
        ess[i] = energy_score(X, y, metric, p) / (div_norm[0] + 1e-8)
        # ess[i] = energy_score(X, y, metric, p) 
    return np.mean(ess)


# Main script: Modified to verify the inequality ES(X, Y) >= ES(Y, Y)
if __name__ == "__main__":
    metrics_to_test = {
        'Euclidean (L2)':    {'metric': 'euclidean'},
        'Manhattan (L1)':    {'metric': 'cityblock'},
        'Chebyshev (L-inf)': {'metric': 'chebyshev'},
        # 'Cosine':            {'metric': 'cosine'}
    }

    num_simulations_per_metric = 2000
    
    print(f"--- Verifying the inequality ES(X, Y) >= ES(Y, Y) ---")
    print(f"--- Running {num_simulations_per_metric} simulations for each metric ---\n")

    for metric_name, metric_params in metrics_to_test.items():
        print(f"--- Testing Metric: {metric_name} ---")
        
        num_violations = 0
        
        for i in range(num_simulations_per_metric):
            # Generate random data
            N, M, d = np.random.randint(2, 11, size=3)
            d = max(2, d) # Ensure d is at least 2
            
            X = np.random.randn(N, d)
            Y = X * (np.random.rand() + 0.5) + np.random.rand(d)
            
            if metric_params['metric'] == 'cosine':
                X += 5
                Y += 5

            # Calculate both sides of the inequality
            lhs = calculate_mean_energy_score(X, Y, **metric_params)
            rhs = calculate_mean_energy_score(Y, Y, **metric_params)
            
            # Print a few sample results
            if i < 4:
                print(f"  Sim {i+1:3d}: ES(X, Y) = {lhs:9.4f}, ES(Y, Y) = {rhs:9.4f}.  Holds: {lhs >= rhs - 1e-9}")

            # Check for violations, allowing for small floating point errors
            if lhs < rhs - 1e-9:
                num_violations += 1
        
        # Report the summary for this metric
        print(f"\nSummary for {metric_name}:")
        if num_violations == 0:
            print(f"✅ Inequality ES(X, Y) >= ES(Y, Y) held true for all {num_simulations_per_metric} simulations.")
        else:
            print(f"❌ Found {num_violations} violations in {num_simulations_per_metric} simulations.")
        print("-" * 60 + "\n")