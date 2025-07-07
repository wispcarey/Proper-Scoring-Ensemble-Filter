import argparse
import math
import torch
import sys
import os
import datetime

from .dataset_info import DATASET_INFO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from localization import pairwise_distances


def float_or_random(value):
    try:
        return float(value)
    except ValueError:
        if value == "random":
            return value
        raise argparse.ArgumentTypeError(f"Invalid value for dt: {value}")

def float_or_str(value):
    try:
        return float(value)
    except ValueError:
        return value

def float_or_default(value):
    try:
        return float(value)
    except ValueError:
        if value == "default":
            return "default"
        raise argparse.ArgumentTypeError(f"Invalid value: {value}")

def float_or_none_or_default(value):
    """Custom type for argparse that accepts a float, 'none', or 'default'."""
    if isinstance(value, str):
        if value.lower() == 'none':
            return None
        if value.lower() == 'default':
            return 'default'
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a float, 'none', or 'default'.")

def int_or_default(value):
    try:
        return int(value)
    except ValueError:
        if value == "default":
            return "default"
        raise argparse.ArgumentTypeError(f"Invalid value: {value}")

def parse_list_type(s):
    return s.split(',')

def get_parameters():
    parser = argparse.ArgumentParser()
    # dataset setting
    parser.add_argument('--dataset', type=str, default='lorenz96', choices=['lorenz63', 'lorenz96', 'ks', 'linear'],
                        help='Dataset name')
    parser.add_argument('--num_loader_workers', type=int, default=16,
                        help='number of workers for the data loader')
    parser.add_argument('--N', type=int, default=20,
                        help='Number of ensemble sample')
    parser.add_argument('--train_steps', type=int_or_default, default="default",
                        help='Training time steps')
    parser.add_argument('--train_traj_num', type=int_or_default, default="default",
                        help='Number of training trajectories')
    parser.add_argument('--test_steps', type=int_or_default, default="default",
                        help='Testing time steps')
    parser.add_argument('--test_traj_num', type=int_or_default, default="default",
                        help='Number of testing trajectories')
    parser.add_argument('--valid_steps', type=int, default=0,
                        help='Validation time steps')
    parser.add_argument('--burn_steps', type=int, default=1000, help='Test time steps')
    parser.add_argument('--dt', type=float_or_default, default="default", help='Time stepsize')
    parser.add_argument('--dt_iter', type=int_or_default, default="default",
                        help='Separate dt into multiple steps of rk4 forward')
    parser.add_argument('--overlap_rate', type=float, default=0.75,
                        help='Overlap rate when segmenting learning trajectories')
    parser.add_argument('--new_data', action='store_false',
                        help='do not use the check_dist option when generating training data')
    parser.add_argument('--sigma_ens', type=float_or_none_or_default, default="default",
                        help='std of initial ensemble samples (can be float, "none", or "default")')
    parser.add_argument('--sigma_v', type=float_or_none_or_default, default="default",
                        help='noise std in the latent process (can be float or "none")')
    parser.add_argument('--sigma_y', type=float_or_none_or_default, default="default",
                        help='noise std in the observation (can be float or "none")')

    # loss function
    parser.add_argument('--ignore_first', type=int, default=0,
                        help='number of first steps ignored in calculating the loss in each training traj')
    parser.add_argument('--loss_warm_up', action='store_true',
                        help='warm-up the loss according to epochs')
    parser.add_argument('--loss_type', type=parse_list_type, default=["nl2"],
                        help='the type of loss function, split by comma, e.g., "l2,rmse,crps"')
    parser.add_argument('--es_p', type=float, default=1,
                        help='the power of energy score')
    parser.add_argument('--kes_sigma', type=lambda x: float(x) if x.lower() != 'none' else None,
                        default=1e-2,
                        help='the power of energy score (can be float or None)')

    # training setting
    parser.add_argument('--cp_load_path', type=str, default="no",
                        help='checkpoint load path. "no" for training from scratch')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='training epochs')
    parser.add_argument('--hidden_dim', type=int_or_default, default="default",
                        help='Hidden dimension of the NN')
    parser.add_argument('--batch_size', type=int_or_default, default="default",
                        help='training batch size')
    parser.add_argument('--test_batch_size', type=int_or_default, default="default",
                        help='test batch size')
    parser.add_argument('--detach_training_epoch', type=int, default=10000,
                        help='detach training epochs')
    parser.add_argument('--detach_steps', type=int, default=5,
                        help='Number of steps after which to detach gradients in the trajectory (1 means detach every step)')
    parser.add_argument('--no_localization', action='store_true',
                        help='do not apply localization')
    parser.add_argument('--st_output_dim', type=int, default=64,
                        help='dimension of set transformer output')
    parser.add_argument('--st_num_seeds', type=int, default=16,
                        help='number of seeds in PMA layer')
    parser.add_argument('--st_type', type=str, default='joint', choices=['state_only', 'separate', 'joint'],
        help=(
            'Specifies how the Set Transformer is applied to distributions. '
            '"state_only" is used to process only the state distribution, '
            '"separate" processes the state distribution and observation distribution independently, '
            'and "joint" is used to process the joint distribution of state and observation.'
        )
    )
    parser.add_argument('--unfreeze_WQ', action='store_true',
                        help='unfreeze Q weights in the PMA layer')
    parser.add_argument('--loc_max_val', type=float, default=2,
                        help='max value of the localization weights')
    parser.add_argument('--obs_in_loc', action='store_true',
                        help='include observation in the input of localization')
    # parser.set_defaults(obs_in_loc=True)


    # output setting
    parser.add_argument('--print_batch', type=float_or_str, default="auto",
                        help='Number of batchs to print out the results')
    parser.add_argument('--save_epoch', type=int, default=25,
                        help='Number of epoch between save and test')
    parser.add_argument('--trail', type=str, default="",
                        help='trail name')
    parser.add_argument('--zero_infl', action='store_true',
                        help='set inflation to zero')
    
    # optimization setting
    parser.add_argument('--learning_rate', type=float_or_default, default='default',
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--adjust_lr', action='store_true',
                        help='adjusting learning rate')
    parser.set_defaults(adjust_lr=True) ### default adjust lr
    parser.add_argument('--lr_decay_epochs', type=str, default='200,300,400',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='decay rate for learning rate')
    parser.add_argument('--warm_up', action='store_true',
                        help='adjusting learning rate')
    # parser.set_defaults(warm_up=True) ### default warm up
    parser.add_argument('--warm_up_rate', type=float, default=1.02,
                        help='warm-up rate')
    parser.add_argument('--warm_up_epochs', type=int, default=50,
                        help='number of epochs in the beginning to warm up the learning rate')
    parser.add_argument('--SGD', action='store_true',
                        help='use SGD optimizer, otherwise use Adam')
    parser.add_argument('--no_running_loss', action='store_true',
                        help='Do not use an accumulative loss in training. Calculate the loss once finishing the entire trajectory')
    
    # test settings
    parser.add_argument('--pf_verification', action='store_true', help='Use particle filter to approximate true filtering distribution')
    parser.add_argument('--pf_N', type=int, default=1000,
                        help='number particles for PF')
    parser.add_argument('--sigma_reg', type=float_or_none_or_default, default=None,
                        help='the std of noise added to resampling in BPF')


    # others
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'],
                        help='device')
    parser.add_argument('--seed', type=str, default=None, help='Random Seed')
    parser.add_argument('--test_rounds', type=int, default=1, help='Number of test rounds when selecting --test_only')
    parser.add_argument('--GPU_memory', type=int, default=16, help='GPU memory in GB')
    parser.add_argument('--normal_output', action='store_true', help='Just print the output without redirecting to a .txt file')
    parser.add_argument('--suffix', type=str, default="", help='save folder suffix')

    # version setting
    parser.add_argument('--v', type=str, choices=['CorrTerms','EtE','EtE-LRes','EtE2','EnKF','ESRF','LETKF'],
                        default='CorrTerms', help='versions')

    args = parser.parse_args()

    # iterations = args.lr_decay_epochs.split(',')
    # args.lr_decay_epochs = list([])
    # for it in iterations:
    # 	args.lr_decay_epochs.append(int(it))

    if not args.warm_up:
        args.warm_up_rate = 1
        args.warm_up_epochs = 1

    if not args.adjust_lr:
        args.lr_decay_epochs = "0"
        args.lr_decay_rate = 1

    dataset_config = DATASET_INFO.get(args.dataset, {})

    if args.dt == 'default':
        args.dt = dataset_config.get('dt')

    if args.dt_iter == 'default':
        args.dt_iter = dataset_config.get('dt_iter')

    if args.test_steps == 'default':
        args.test_steps = dataset_config.get('test_steps')

    if args.test_traj_num == 'default':
        args.test_traj_num = dataset_config.get('test_traj_num')

    if args.train_steps == 'default':
        args.train_steps = dataset_config.get('train_steps')

    if args.train_traj_num == 'default':
        args.train_traj_num = dataset_config.get('train_traj_num')

    if args.hidden_dim == 'default':
        args.hidden_dim = dataset_config.get('hidden_dim')

    if args.learning_rate == 'default':
        args.learning_rate = dataset_config.get('learning_rate')

    if args.batch_size == 'default':
        ori_batch_size = dataset_config.get('batch_size')
        args.batch_size = ori_batch_size
    else:
        ori_batch_size = args.batch_size

    if args.test_batch_size == 'default':
        args.test_batch_size = args.test_traj_num

    if args.sigma_ens == 'default':
        args.sigma_ens = dataset_config.get('sigma_ens')
        
    if args.sigma_v == 'default':
        args.sigma_v = dataset_config.get('sigma_v')
        
    if args.sigma_y == 'default':
        args.sigma_y = dataset_config.get('sigma_y')
        
    if args.no_running_loss:
        args.running_loss = False
    else:
        args.running_loss = True

    args.ori_dim = dataset_config.get('dim')
    args.obs_dim = dataset_config.get('obs_dim')
    args.obs_inds = dataset_config.get('obs_inds')
    args.clamp = dataset_config.get('clamp')

    if args.device == 'cuda':
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        print(f"Detected {torch.cuda.device_count()} GPUs")
        if num_gpus > 1:
            args.use_data_parallel = True
            args.batch_size = args.batch_size * num_gpus
            print(f"Use DataParallel. Adjusted Batch Size: {args.batch_size}")
        else:
            args.use_data_parallel = False
            print(f"Do not Use DataParallel")
    else:
        print("Use CPU")


    if args.GPU_memory != 16 and args.batch_size is not None:
        args.batch_size = args.batch_size * int(args.GPU_memory / 16)

    if args.print_batch == "auto" and args.train_traj_num is not None and args.batch_size is not None and args.batch_size > 0:
        args.print_batch = math.ceil(args.train_traj_num / args.batch_size)

    if args.adjust_lr:
        print("[INFO] Learning rate adjustment is ENABLED.")

    if args.warm_up:
        print("[INFO] Warm-up is ENABLED.")

    if ori_batch_size != args.batch_size and all(isinstance(v, (int, float)) for v in [args.learning_rate, args.batch_size, ori_batch_size]):
        args.learning_rate = args.learning_rate * (args.batch_size / ori_batch_size) ** 0.5

    if args.st_type == 'state_only':
        print("Only apply an ST on the ensemble state data.")
        args.input_dim = args.ori_dim + 2 * args.obs_dim + args.st_output_dim
        args.local_input_dim = args.st_output_dim
    elif args.st_type == "separate":
        print("Apply STs separately on the ensemble state data and observation data.")
        args.input_dim = args.ori_dim + 2 * args.obs_dim + args.st_output_dim * 2
        args.local_input_dim = args.st_output_dim * 2
    elif args.st_type == 'joint':
        print("Apply an ST on the joint distribution of ensemble state data and observation data.")
        args.input_dim = args.ori_dim + 2 * args.obs_dim + args.st_output_dim * 2
        args.local_input_dim = args.st_output_dim * 2
    else:
        raise ValueError("Please use a valid st_type.")

    if args.obs_in_loc:
        args.local_input_dim += args.obs_dim

    # localization
    if args.ori_dim and args.obs_inds is not None:
        full_inds = torch.arange(0, args.ori_dim)
        Lvy = pairwise_distances(full_inds[:, None], args.obs_inds[:, None], domain=(args.ori_dim,)).to(args.device)
        Lyy = pairwise_distances(args.obs_inds[:, None], args.obs_inds[:, None], domain=(args.ori_dim,)).to(args.device)
        args.diff_dist = torch.unique(torch.cat((Lvy.flatten(), Lyy.flatten())))
        args.num_dist = len(args.diff_dist)
        args.Lvy = Lvy
        args.Lyy = Lyy
    else:
        args.diff_dist = None
        args.num_dist = 0
        args.Lvy = None
        args.Lyy = None

    # for benchmark
    if args.v in ['EnKF','ESRF','LETKF']:
        args.colorbar_range = dataset_config.get('colorbar_range')
    else:
        args.colorbar_range = None

    # Save folder
    if args.cp_load_path != "no":
        args.suffix += "_tuned"
    else:
        args.suffix = ""
    folder_name = os.path.join("save", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    loss_type_name = "_".join([loss_type for loss_type in args.loss_type])
    folder_name += f"{args.dataset}_{args.sigma_y}_{args.N}_{args.train_steps}_{args.train_traj_num}_{loss_type_name}_{args.st_type}_{args.v}{args.suffix}"
    args.save_folder = folder_name

    return args