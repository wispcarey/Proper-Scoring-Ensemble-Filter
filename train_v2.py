import datetime
import os

import torch
import torch.nn as nn

from config.cli import get_parameters

from utils import setup_optimizer_and_scheduler, save_checkpoint, load_checkpoint
from utils import partial_obs_operator, get_dataloader, redirect_output

from train_test_utils import train_model_v2, test_model_v2, set_models, print_test_results_v2


if __name__ == "__main__":
    args = get_parameters()
    
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    
    # redirect output
    with redirect_output(save_output=not args.normal_output, save_folder=args.save_folder, filename="output.txt"):
        # save_folder
        folder_name = args.save_folder
        
        # torch.cuda.set_device(args.device)
        for key, value in vars(args).items():
            print(f"{key}: {value}")

        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        train_loader, test_loader = get_dataloader(args)

        # set models
        model_list = set_models(args)
        model, infl_model, local_model, st_model1, st_model2 = model_list

        # optimizer
        optimizer, scheduler = setup_optimizer_and_scheduler(model_list, args)

        # load checkpoint
        if args.cp_load_path != "no":
            load_checkpoint(model_list, None, None, filename=args.cp_load_path, use_data_parallel=args.use_data_parallel)

        # Training
        train_loss_list = []
        test_epochs = []
        train_records = {"train_loss": [], "test_epochs": []}
        
        print("Training Start")
        
        # Initial test before training
        initial_test_results = test_model_v2(test_loader, model_list, args, plot_figures=True, fig_name=f'{folder_name}/test_{args.N}_0', save_pdf=False)
        print_test_results_v2(initial_test_results)
        
        for key, value in initial_test_results.items():
            if key != "loc_tensor":
                train_records[key] = [value]
        train_records['test_epochs'].append(0)
        
        for epoch in range(1, 1 + args.epochs):
            train_loss = train_model_v2(epoch, train_loader, model_list, optimizer, scheduler, args)
            train_records['train_loss'].append(train_loss)
            if torch.isnan(train_loss):
                print("NAN loss. Terminate training.")
                break

            if epoch % args.save_epoch == 0:
                # Test at each save epoch
                epoch_test_results = test_model_v2(test_loader, model_list, args, plot_figures=True, fig_name=f'{folder_name}/test_{args.N}_{epoch}', save_pdf=False)
                print_test_results_v2(epoch_test_results)
                
                for key, value in epoch_test_results.items():
                    if key != "loc_tensor":
                        train_records[key].append(value)
                train_records['test_epochs'].append(epoch)
                
                # Save training records and model checkpoint
                torch.save(train_records, os.path.join(folder_name, f"training_records.pt"))
                save_checkpoint(model_list, optimizer, scheduler, filename=os.path.join(folder_name, f"cp_{epoch}.pth"))
                




