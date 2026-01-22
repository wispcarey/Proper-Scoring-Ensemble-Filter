import datetime
import os
import math

import torch
import torch.nn as nn

from config.cli import get_parameters

from utils import setup_optimizer_and_scheduler, save_checkpoint, load_checkpoint
from utils import partial_obs_operator, get_dataloader, redirect_output
from train_test_utils import train_model, test_model, set_models, print_test_results


if __name__ == "__main__":
    args = get_parameters()
    
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # redirect output
    with redirect_output(save_output=not args.normal_output, save_folder=args.save_folder, filename="ft_output.txt"):
        # folder name
        folder_name = args.save_folder
        
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

        # set models
        model_list = set_models(args)
        model, infl_model, local_model, st_model1, st_model2 = model_list
        ft_params = sum(sum(p.numel() for p in model.parameters()) for model in model_list[:3])
        print(f'Fine-tuning parameters: {ft_params}')


        ##################### fine-tuning on different N
        N_list = [5,10,15,20, 40, 60, 100]
        # N_list = [40, 60, 100]
        ori_batch_size = args.batch_size

        for N in N_list:
            if N == 40 or N == 60:
                args.batch_size = ori_batch_size // 4
            elif N == 100:
                args.batch_size = ori_batch_size // 8
            else:
                args.batch_size = ori_batch_size 
            args.print_batch = math.ceil(args.train_traj_num / args.batch_size)
            args.N = N
            # optimizer
            optimizer, scheduler = setup_optimizer_and_scheduler(model_list, args)
            train_loader, test_loader = get_dataloader(args)

            if args.cp_load_path != "no":
                load_checkpoint(model_list, None, None, filename=args.cp_load_path, use_data_parallel=args.use_data_parallel)
                
                # Freeze parameters for static models
                for param in st_model1.parameters():
                    param.requires_grad = False
                for param in st_model2.parameters():
                    param.requires_grad = False

                # Calculate and print parameter counts
                total_params = sum(p.numel() for model in model_list for p in model.parameters())
                trainable_params = sum(p.numel() for model in model_list for p in model.parameters() if p.requires_grad)
                
                print(f"Total parameters: {total_params}")
                print(f"Trainable parameters: {trainable_params}")

            # Fine-tuning
            train_loss_list = []
            test_epochs = []
            train_records = {"train_loss": [], "test_epochs": []}
            
            print("Training Start")
            
            # Initial test before training
            initial_test_results = test_model(test_loader, model_list, args, H_info=H_info, plot_figures=True, fig_name=f'{folder_name}/ft_test_{args.N}_0', save_pdf=False)
            print_test_results(initial_test_results)
            
            for key, value in initial_test_results.items():
                if key != "loc_tensor":
                    train_records[key] = [value]
            train_records['test_epochs'].append(0)
            
            for epoch in range(1, 1 + args.epochs):
                train_loss = train_model(epoch, train_loader, model_list, optimizer, scheduler, args, H_info=H_info)
                train_records['train_loss'].append(train_loss)
                
                if epoch % args.save_epoch == 0:
                    # Test at each save epoch
                    epoch_test_results = test_model(test_loader, model_list, args, H_info=H_info, plot_figures=True, fig_name=f'{folder_name}/ft_test_{args.N}_{epoch}', save_pdf=False)
                    print_test_results(epoch_test_results)
                    
                    for key, value in epoch_test_results.items():
                        if key != "loc_tensor":
                            train_records[key] = [value]
                    train_records['test_epochs'].append(epoch)
                    
                    # Save training records and model checkpoint
                    torch.save(train_records, os.path.join(folder_name, f"ft_records_{N}.pt"))
                    save_checkpoint(model_list, optimizer, scheduler, filename=os.path.join(folder_name, f"ft_cp_{N}_{epoch}.pth"))