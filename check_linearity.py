import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from config.cli import get_parameters
from utils import setup_optimizer_and_scheduler, load_checkpoint, redirect_output
from train_test_utils import set_models


# [REVISED FUNCTION]
def check_partial_linearity(model, d_input, start_ind, end_ind, n_samples=2000, return_data=False):
    """
    Checks if a model is affine/linear with respect to a specific slice of its input
    by fitting a linear model and calculating the R-squared score.

    Args:
        model (nn.Module): The neural network to test.
        d_input (int): The total dimension of the input vector.
        start_ind (int): The starting index of the input slice to test.
        end_ind (int): The ending index of the input slice to test.
        n_samples (int): The number of samples to generate for the linear regression.
        return_data (bool): If True, returns data needed for visualization.

    Returns:
        float: The R-squared score.
        (optional) tuple: If return_data is True, also returns (slices, y_true, solution).
    """
    model.eval()
    device = next(model.parameters()).device
    slice_dim = end_ind - start_ind

    with torch.no_grad():
        # 1. Generate a dataset
        x_background = torch.randn(1, d_input, device=device) * 5
        x_background[:, start_ind:end_ind] = 0
        x_background = x_background.repeat(n_samples, 1)
        slices = torch.abs(torch.randn(n_samples, slice_dim, device=device) * 5)
        # slices = torch.randn(n_samples, slice_dim, device=device) * 5
        x_full = x_background.clone()
        x_full[:, start_ind:end_ind] = slices
        y_true = model(x_full)

        # 2. Fit a linear model
        S_aug = torch.cat([slices, torch.ones(n_samples, 1, device=device)], dim=1)
        try:
            solution = torch.linalg.lstsq(S_aug, y_true).solution
        except torch.linalg.LinAlgError:
            print("Warning: Least squares solving failed. The data might be ill-conditioned.")
            if return_data:
                return 0.0, None, None, None
            return 0.0
        y_pred_linear = S_aug @ solution

        # 3. Calculate the R-squared score
        ss_total = torch.sum((y_true - y_true.mean(dim=0))**2)
        ss_residual = torch.sum((y_true - y_pred_linear)**2)
        if ss_total < 1e-6:
            r2_score = 1.0
        else:
            r2_score = 1 - ss_residual / ss_total
        
        if return_data:
            # Return the solution (weights + bias) instead of pre-calculated predictions
            return r2_score.item(), slices, y_true, solution
        else:
            return r2_score.item()


# --- [REVISED FUNCTION] Visualization ---
def visualize_linearity(slices, y_true, solution, input_vis_ind, output_vis_ind, start_ind, save_folder="."):
    """
    Generates and saves plots to visualize the partial linearity fit by extracting
    the precise 1D linear relationship from the fitted model.

    Args:
        slices (torch.Tensor): The input slices used for the test (N, slice_dim).
        y_true (torch.Tensor): The true model outputs (N, output_dim).
        solution (torch.Tensor): The solution from lstsq, containing weights and biases.
        input_vis_ind (list): List of indices within the slice to plot.
        output_vis_ind (list): List of indices of the model output to plot.
        start_ind (int): The starting index of the input slice (for naming files).
        save_folder (str): The folder where plots will be saved.
    """
    print(f"Generating linearity visualization plots in folder: {save_folder}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Extract weights and biases from the solution tensor
    # W shape: (slice_dim, output_dim), b shape: (1, output_dim)
    W = solution[:-1, :]
    b = solution[-1, :]

    # Move data to CPU and convert to numpy for plotting
    slices_np = slices.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    W_np = W.cpu().numpy()
    b_np = b.cpu().numpy()

    for i in input_vis_ind:
        for j in output_vis_ind:
            # Check if indices are valid
            if i >= slices_np.shape[1] or j >= y_true_np.shape[1]:
                print(f"Warning: Skipping plot for (input_idx={i}, output_idx={j}) due to out-of-bounds index.")
                continue

            x_data = slices_np[:, i]
            y_data_true = y_true_np[:, j]
            
            # Get the specific slope and intercept for this 1D relationship
            slope = W_np[i, j]
            intercept = b_np[j]

            # Create points for the line plot
            x_line = np.array([x_data.min(), x_data.max()])
            y_line = slope * x_line + intercept

            plt.figure(figsize=(10, 6))
            plt.scatter(x_data, y_data_true, alpha=0.3, label="Actual Model Output")
            plt.plot(x_line, y_line, color='red', linewidth=2, label="Fitted 1D Linear Relationship")
            
            # Set titles and labels
            global_input_dim = start_ind + i
            plt.title(f"Linearity Check: Input Dimension {global_input_dim} vs. Output Dimension {j}")
            plt.xlabel(f"Input Value at Dimension {global_input_dim}")
            plt.ylabel(f"Output Value at Dimension {j}")
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            filename = f"linearity_vis_input_{global_input_dim}_vs_output_{j}_abs.png"
            filepath = os.path.join(save_folder, filename)
            plt.savefig(filepath)
            plt.close() # Close the figure to free memory
    print("Visualization plots saved.")


if __name__ == "__main__":
    args = get_parameters()
    
    if args.cp_load_path == "no":
        raise ValueError("The parameter cp_load_path is invalid.")
    folder_name = os.path.dirname(args.cp_load_path)
    
    with redirect_output(save_output=not args.normal_output, save_folder=folder_name, filename="check_linearity_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))
        
        print(f"Checking partial linearity for the model loaded from checkpoint: {args.cp_load_path}")

        model_list = set_models(args)
        model, infl_model, local_model, st_model1, st_model2 = model_list

        if args.cp_load_path != "no":
            load_checkpoint(model_list, None, None, filename=args.cp_load_path, use_data_parallel=args.use_data_parallel)

        if args.v == 'EtE' or args.v == 'EtE-LRes':
            d_input = args.input_dim
            start_ind = args.ori_dim + args.obs_dim 
            end_ind = args.ori_dim + args.obs_dim * 2

            print("\n" + "="*50)
            print("Performing partial linearity check by fitting a linear model...")
            print(f"Checking linearity w.r.t. input slice [{start_ind}:{end_ind}]")
            
            # --- [MODIFIED] Get R2 score and the solution for plotting ---
            r2_score, slices, y_true, solution = check_partial_linearity(
                model=model,
                d_input=d_input,
                start_ind=start_ind,
                end_ind=end_ind,
                n_samples=50000,
                return_data=True
            )
            print(solution)
            
            print(f"Linear Fit R-squared (R²) Score: {r2_score:.6f}")
            print("="*50 + "\n")
            
            # --- [MODIFIED] Perform visualization using the solution tensor ---
            if r2_score > 0.0 and solution is not None:
                input_vis_ind = [0]
                output_vis_ind = [0, 1, 2]
                
                visualize_linearity(
                    slices=slices,
                    y_true=y_true,
                    solution=solution, # Pass the solution tensor
                    input_vis_ind=input_vis_ind,
                    output_vis_ind=output_vis_ind,
                    start_ind=start_ind,
                    save_folder=folder_name
                )
            else:
                print("Skipping visualization due to failed linear fit or zero R2 score.")

        else:
            raise NotImplementedError