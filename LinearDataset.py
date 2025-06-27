import torch
import numpy as np
import os
import pickle
from torch.utils.data import Dataset
from scipy.linalg import eigvals


class LinearSystemDataset(Dataset):
    """
    PyTorch Dataset for generating random linear system parameters.
    
    Each sample contains:
    - m: vector of length d
    - C: d x d symmetric matrix
    - A: d x d matrix with eigenvalues having absolute values < 1.1
    - H: d_obs x d matrix
    - sigma_v: positive scalar (observation noise)
    - sigma_y: positive scalar (process noise)
    """
    
    def __init__(self, d, d_obs, num_samples, sigma_v=None, sigma_y=None, 
                 data_name="default", load_existing=True, seed=None):
        """
        Initialize the dataset.
        
        Args:
            d (int): Dimension of state vector
            d_obs (int): Dimension of observation vector
            num_samples (int): Number of samples to generate
            sigma_v (float or None): Fixed value for sigma_v, or None for random generation
            sigma_y (float or None): Fixed value for sigma_y, or None for random generation
            data_name (str): Name for saving/loading data files
            load_existing (bool): Whether to load existing data if available
            seed (int or None): Random seed for reproducibility
        """
        self.d = d
        self.d_obs = d_obs
        self.num_samples = num_samples
        self.sigma_v_fixed = sigma_v
        self.sigma_y_fixed = sigma_y
        self.data_name = data_name
        self.seed = seed
        
        # Create data directory if it doesn't exist
        os.makedirs("data/linear", exist_ok=True)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # File path for saving/loading data
        self.data_file = f"data/linear/{data_name}_d{d}_dobs{d_obs}_n{num_samples}.pkl"
        
        # Try to load existing data if requested
        if load_existing and os.path.exists(self.data_file):
            print(f"Loading existing data from {self.data_file}")
            self._load_data()
        else:
            print(f"Generating new data and saving to {self.data_file}")
            self._generate_data()
            self._save_data()
    
    def _generate_stable_matrix_A(self):
        """
        Generate a random matrix A with eigenvalues having absolute values < 1.1.
        
        Strategy: Generate a random matrix, compute its eigenvalues, and scale them
        if necessary to ensure stability.
        """
        # Generate random matrix
        A = np.random.randn(self.d, self.d)
        
        # Compute eigenvalues
        eigs = eigvals(A)
        max_abs_eig = np.max(np.abs(eigs))
        
        # Scale matrix if largest eigenvalue is too large
        if max_abs_eig >= 1.1:
            scaling_factor = 1.05 / max_abs_eig  # Scale to be slightly less than 1.1
            A = A * scaling_factor
        
        return A
    
    def _generate_symmetric_matrix_C(self):
        """
        Generate a random symmetric positive definite matrix C.
        """
        # Generate random matrix
        temp = np.random.randn(self.d, self.d)
        # Make it symmetric and positive definite
        C = temp @ temp.T
        # Add small diagonal term to ensure positive definiteness
        C += 0.1 * np.eye(self.d)
        C = torch.linalg.cholesky(C)
        return C
    
    def _generate_data(self):
        """
        Generate all random parameters for the dataset.
        """
        self.data = []
        
        for i in range(self.num_samples):
            # Generate m: random vector of length d
            m = np.random.randn(self.d)
            
            # Generate C: symmetric positive definite matrix
            C = self._generate_symmetric_matrix_C()
            
            # Generate A: matrix with controlled eigenvalues
            A = self._generate_stable_matrix_A()
            
            # Generate H: random observation matrix
            H = np.random.randn(self.d_obs, self.d)
            
            # Generate sigma_v
            if self.sigma_v_fixed is not None:
                sigma_v = self.sigma_v_fixed
            else:
                sigma_v = np.random.uniform(0, 0.3)
            
            # Generate sigma_y
            if self.sigma_y_fixed is not None:
                sigma_y = self.sigma_y_fixed
            else:
                sigma_y = np.random.uniform(0.1, 1.0)
            
            # Store as dictionary
            sample = {
                'm': m,
                'C': C,
                'A': A,
                'H': H,
                'sigma_v': sigma_v,
                'sigma_y': sigma_y
            }
            
            self.data.append(sample)
    
    def _save_data(self):
        """
        Save generated data to file.
        """
        save_dict = {
            'data': self.data,
            'metadata': {
                'd': self.d,
                'd_obs': self.d_obs,
                'num_samples': self.num_samples,
                'sigma_v_fixed': self.sigma_v_fixed,
                'sigma_y_fixed': self.sigma_y_fixed,
                'seed': self.seed
            }
        }
        
        with open(self.data_file, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Data saved to {self.data_file}")
    
    def _load_data(self):
        """
        Load data from file.
        """
        try:
            with open(self.data_file, 'rb') as f:
                loaded_dict = pickle.load(f)
            
            self.data = loaded_dict['data']
            metadata = loaded_dict['metadata']
            
            # Verify metadata matches current parameters
            if (metadata['d'] != self.d or 
                metadata['d_obs'] != self.d_obs or 
                metadata['num_samples'] != self.num_samples):
                print("Warning: Loaded data parameters don't match current parameters")
                print(f"Loaded: d={metadata['d']}, d_obs={metadata['d_obs']}, n={metadata['num_samples']}")
                print(f"Current: d={self.d}, d_obs={self.d_obs}, n={self.num_samples}")
            
            print(f"Successfully loaded {len(self.data)} samples")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Generating new data instead...")
            self._generate_data()
            self._save_data()
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing all parameters as PyTorch tensors
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        sample = self.data[idx]
        
        # Convert numpy arrays to PyTorch tensors
        # Note: sigma values are returned as scalars, which will be properly batched by DataLoader
        return {
            'm': torch.tensor(sample['m'], dtype=torch.float32),
            'C': torch.tensor(sample['C'], dtype=torch.float32),
            'A': torch.tensor(sample['A'], dtype=torch.float32),
            'H': torch.tensor(sample['H'], dtype=torch.float32),
            'sigma_v': torch.tensor([sample['sigma_v']], dtype=torch.float32),  # Wrap scalar in tensor
            'sigma_y': torch.tensor([sample['sigma_y']], dtype=torch.float32)   # Wrap scalar in tensor
        }
    
    def get_sample_info(self, idx=0):
        """
        Print information about a specific sample for debugging.
        
        Args:
            idx (int): Index of the sample to inspect
        """
        if idx >= len(self.data):
            print(f"Index {idx} out of range")
            return
        
        sample = self.data[idx]
        A = sample['A']
        eigs = eigvals(A)
        
        print(f"Sample {idx} information:")
        print(f"  m shape: {sample['m'].shape}")
        print(f"  C shape: {sample['C'].shape}, symmetric: {np.allclose(sample['C'], sample['C'].T)}")
        print(f"  A shape: {sample['A'].shape}")
        print(f"  A eigenvalues (abs): {np.abs(eigs)}")
        print(f"  Max |eigenvalue|: {np.max(np.abs(eigs)):.4f}")
        print(f"  H shape: {sample['H'].shape}")
        print(f"  sigma_v: {sample['sigma_v']:.4f}")
        print(f"  sigma_y: {sample['sigma_y']:.4f}")


# Example usage
if __name__ == "__main__":
    # Example 1: Generate dataset with random sigma values
    dataset1 = LinearSystemDataset(
        d=5, 
        d_obs=3, 
        num_samples=100, 
        data_name="example1",
        load_existing=False,
        seed=42
    )
    
    # Example 2: Generate dataset with fixed sigma values
    dataset2 = LinearSystemDataset(
        d=4, 
        d_obs=2, 
        num_samples=50, 
        sigma_v=0.1, 
        sigma_y=0.5,
        data_name="example2_fixed_sigma",
        load_existing=False
    )
    
    # Print sample information
    dataset1.get_sample_info(0)
    
    # Get a single sample
    sample = dataset1[0]
    print(f"\nSingle sample tensor shapes:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape}")
    
    # Test batch functionality
    from torch.utils.data import DataLoader
    
    print(f"\nTesting batch functionality:")
    dataloader = DataLoader(dataset1, batch_size=4, shuffle=True)
    
    # Get first batch
    batch = next(iter(dataloader))
    print(f"Batch tensor shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    # Verify that sigma values are properly batched
    print(f"\nBatch sigma_v values: {batch['sigma_v'].squeeze()}")
    print(f"Batch sigma_y values: {batch['sigma_y'].squeeze()}")
    
    # Example of how to use in training loop
    print(f"\nExample training loop usage:")
    for i, batch in enumerate(dataloader):
        if i >= 2:  # Just show first 2 batches
            break
        
        m = batch['m']        # Shape: [batch_size, d]
        C = batch['C']        # Shape: [batch_size, d, d]
        A = batch['A']        # Shape: [batch_size, d, d] 
        H = batch['H']        # Shape: [batch_size, d_obs, d]
        sigma_v = batch['sigma_v'].squeeze()  # Shape: [batch_size]
        sigma_y = batch['sigma_y'].squeeze()  # Shape: [batch_size]
        
        print(f"  Batch {i+1}: m={m.shape}, C={C.shape}, A={A.shape}, H={H.shape}")
        print(f"           sigma_v={sigma_v.shape}, sigma_y={sigma_y.shape}")