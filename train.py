import os

import torch

from config.cli import get_parameters

from psef.utils.legacy import setup_optimizer_and_scheduler, save_checkpoint, load_checkpoint
from psef.utils.legacy import build_observation_operator, get_dataloader, redirect_output, should_redirect_output

from psef.training.legacy import (
    train_model,
    train_model_v2,
    test_model,
    test_model_v2,
    set_models,
    print_test_results,
    print_test_results_v2,
)


def _prepare_run(args):
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)


def _print_args_and_seed(args):
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.seed is not None and args.seed != "None":
        torch.manual_seed(int(args.seed))


def _build_training_state(args):
    train_loader, test_loader = get_dataloader(args)
    model_list = set_models(args)
    optimizer, scheduler = setup_optimizer_and_scheduler(model_list, args)
    if args.cp_load_path != "no":
        load_checkpoint(
            model_list,
            None,
            None,
            filename=args.cp_load_path,
            use_data_parallel=args.use_data_parallel,
        )
    return train_loader, test_loader, model_list, optimizer, scheduler


def _run_standard_training(args):
    folder_name = args.save_folder
    H_info = build_observation_operator(args)
    train_loader, test_loader, model_list, optimizer, scheduler = _build_training_state(args)
    train_records = {"train_loss": [], "test_epochs": []}

    print("Training Start")

    initial_test_results = test_model(
        test_loader,
        model_list,
        args,
        H_info=H_info,
        plot_figures=args.save_test_figures,
        fig_name=f'{folder_name}/test_{args.N}_0',
        save_pdf=False,
    )
    print_test_results(initial_test_results)

    for key, value in initial_test_results.items():
        if key != "loc_tensor":
            train_records[key] = [value]
    train_records['test_epochs'].append(0)

    for epoch in range(1, 1 + args.epochs):
        train_loss = train_model(epoch, train_loader, model_list, optimizer, scheduler, args, H_info=H_info)
        train_records['train_loss'].append(train_loss)
        if torch.isnan(torch.tensor(train_loss)):
            print("NAN loss. Terminate training.")
            break

        if epoch % args.save_epoch == 0:
            epoch_test_results = test_model(
                test_loader,
                model_list,
                args,
                H_info=H_info,
                plot_figures=args.save_test_figures,
                fig_name=f'{folder_name}/test_{args.N}_{epoch}',
                save_pdf=False,
            )
            print_test_results(epoch_test_results)

            for key, value in epoch_test_results.items():
                if key != "loc_tensor":
                    train_records[key].append(value)
            train_records['test_epochs'].append(epoch)

            torch.save(train_records, os.path.join(folder_name, "training_records.pt"))
            save_checkpoint(model_list, optimizer, scheduler, filename=os.path.join(folder_name, f"cp_{epoch}.pth"))


def _run_linear_training(args):
    folder_name = args.save_folder
    train_loader, test_loader, model_list, optimizer, scheduler = _build_training_state(args)
    train_records = {"train_loss": [], "test_epochs": [], "test_results": []}

    print("Training Start")

    initial_test_results = test_model_v2(
        test_loader,
        model_list,
        args,
        plot_figures=False,
        fig_name=f'{folder_name}/test_{args.N}_0',
        save_pdf=False,
    )
    print_test_results_v2(initial_test_results)

    for key, value in initial_test_results.items():
        if key != "loc_tensor":
            train_records[key] = [value]
    train_records['test_epochs'].append(0)
    train_records['test_results'].append(initial_test_results)

    for epoch in range(1, 1 + args.epochs):
        train_loss = train_model_v2(epoch, train_loader, model_list, optimizer, scheduler, args)
        train_records['train_loss'].append(train_loss)
        if torch.isnan(torch.tensor(train_loss)):
            print("NAN loss. Terminate training.")
            break

        if epoch % args.save_epoch == 0:
            epoch_test_results = test_model_v2(
                test_loader,
                model_list,
                args,
                plot_figures=False,
                fig_name=f'{folder_name}/test_{args.N}_{epoch}',
                save_pdf=False,
            )
            print_test_results_v2(epoch_test_results)

            for key, value in epoch_test_results.items():
                if key != "loc_tensor":
                    train_records[key].append(value)
            train_records['test_epochs'].append(epoch)
            train_records['test_results'].append(epoch_test_results)

            torch.save(train_records, os.path.join(folder_name, "training_records.pt"))
            save_checkpoint(model_list, optimizer, scheduler, filename=os.path.join(folder_name, f"cp_{epoch}.pth"))


def main():
    args = get_parameters()
    _prepare_run(args)

    with redirect_output(save_output=should_redirect_output(args), save_folder=args.save_folder, filename="output.txt"):
        _print_args_and_seed(args)
        if args.dataset == "linear":
            _run_linear_training(args)
        else:
            _run_standard_training(args)


if __name__ == "__main__":
    main()
