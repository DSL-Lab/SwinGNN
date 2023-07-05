import argparse
import logging
import os
import pdb
import shutil
import random
import sys
import time
from pprint import pformat

import glob
import numpy as np
import torch
import yaml
from ml_collections import config_dict
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.dist_training import get_ddp_save_flag


def parse_arguments(mode='train'):
    """
    Argument parser and init logging directory.
    """
    parser = argparse.ArgumentParser(description="Running Experiments")

    # logging options
    parser.add_argument('-l', '--log_level', type=str,
                        default='DEBUG', help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument('-m', '--comment', type=str,
                        default="", help="A single line comment for the experiment")

    # distributed training options
    parser.add_argument('--dp', default=False, action='store_true',
                        help='To use DataParallel distributed learning.')
    parser.add_argument('--ddp', default=False, action='store_true',
                        help='To use DistributedDataParallel distributed learning.')
    parser.add_argument('--ddp_gpu_ids', nargs='+', default=None,
                        help="A list of GPU IDs to run DDP distributed learning."
                             "For DP mode, please use CUDA_VISIBLE_DEVICES env. variable to specify GPUs.")
    parser.add_argument('--ddp_init_method', default='env://', type=str,
                        help='torch.distributed.init_process_group options.')

    # model options
    parser.add_argument('--self_cond', type=lambda x: (str(x).lower() == 'true'), default=None,
                        help='To use self-conditioning trick.')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='MCMC sampling steps.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size. Overwrite the loaded config if input is not empty.')
    parser.add_argument('--eval_size', type=int, default=None,
                        help='Total number of samples to generate.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed.')

    # mode specific options
    if mode == 'train':
        parser.add_argument('-c', '--config_file', type=str, required=True,
                            help="Path of config file")
        parser.add_argument('--dataset_name', default=None, type=str,
                            help='To overwrite the dataset name specified in the config.')
        parser.add_argument('--subset', default=None, type=int,
                            help='To overwrite the dataset subset specified in the config.')
        parser.add_argument('--max_node_num', default=None, type=int,
                            help='To overwrite the maximum node number specified in the config.')
        parser.add_argument('--max_epoch', default=None, type=int,
                            help='To overwrite the training epochs specified in the config.')
        parser.add_argument('--lr_init', default=None, type=float,
                            help='To overwrite the initial learning rate specified in the config.')
        parser.add_argument('--sample_interval', type=int, default=None,
                            help='To overwrite the sample interval specified in the config.')
        parser.add_argument('--save_interval', type=int, default=None,
                            help='To overwrite the save interval specified in the config.')
        parser.add_argument('--resume', type=str, default=None,
                            help='To resume training from the latest checkpoint.')

        # swinGNN options
        parser.add_argument('--feature_dims', type=int, default=None,
                            help='To overwrite the model dimension specified in the config.')
        parser.add_argument('--window_size', type=int, default=None,
                            help='To overwrite the window size specified in the config.')
        parser.add_argument('--patch_size', type=int, default=None,
                            help='To overwrite the patch size specified in the config.')

        # node and edge attribute encoding options
        parser.add_argument('--node_encoding', type=str, default=None,
                            help='To overwrite the node encoding specified in the config.')
        parser.add_argument('--edge_encoding', type=str, default=None,
                            help='To overwrite the edge encoding specified in the config.')

        # special ablations for node and edge attribute encoding
        parser.add_argument('--node_only', default=None, action='store_true',
                            help='To remove edge attributes. Reshape the node attributes in the shape of adj matrix')
        parser.add_argument('--binary_edge', default=None, action='store_true',
                            help='To remove node attributes and only use binary edge attributes (adj topology).')

        args = parser.parse_args()
    elif mode == 'eval':
        parser.add_argument('-p', '--model_path', type=str, default=None, required=True,
                            help="Path of the model")
        parser.add_argument('--search_weights', default=False, action='store_true',
                            help='To search for network weights inside the path.')
        parser.add_argument('--min_epoch', type=int, default=None,
                            help='Select network weights with minimum number of training epochs.')
        parser.add_argument('--max_epoch', type=int, default=None,
                            help='Select network weights with maximum number of training epochs.')
        parser.add_argument('--use_ema', default='all', nargs='+',
                            help='To use EMA version weight with specified coefficients.')
        args = parser.parse_args()

        # handle special use_ema keywords 'all' or 'none'
        _use_ema = args.use_ema
        if (isinstance(_use_ema, list) and len(_use_ema) == 1) or isinstance(_use_ema, str):
            # either 'all', 'none' or a single value; it must be a string
            _use_ema = _use_ema[0] if isinstance(_use_ema, list) else _use_ema
            assert isinstance(_use_ema, str)
            if _use_ema in ['all', 'none']:
                args.use_ema = None if _use_ema == 'none' else 'all'
            else:
                args.use_ema = [float(_use_ema)]
        else:
            # specific EMA coefficients
            _use_ema = []
            for item in args.use_ema:
                # store float number except for special keywords 'all' or 'none'
                _use_ema.append(float(item) if item not in ['all', 'none'] else item)
            args.use_ema = _use_ema  # always a list

        # handle model path and its config file
        assert isinstance(args.model_path, str) and os.path.exists(args.model_path)
        if os.path.isfile(args.model_path):
            # single model file
            config_file = os.path.abspath(os.path.join(os.path.dirname(args.model_path), '../config.yaml'))
            if not os.path.exists(config_file):
                config_file = os.path.abspath(os.path.join(os.path.dirname(args.model_path), 'config.yaml'))
            args.model_path = [args.model_path]
        elif os.path.isdir(args.model_path):
            # multiple model files
            assert args.search_weights, 'Please specify --search_weights to search for model weights.'
            config_file = os.path.abspath(os.path.join(args.model_path, '../config.yaml'))

            _model_path_ls = sorted(glob.glob(os.path.join(args.model_path, '*.pth')))
            min_epoch = 0 if args.min_epoch is None else args.min_epoch
            max_epoch = float('inf') if args.max_epoch is None else args.max_epoch
            model_path_ls = []
            for model_path in _model_path_ls:
                _epoch = os.path.basename(model_path).split('_')[-1].replace('.pth', '')
                if _epoch == 'best':
                    continue
                else:
                    _epoch = int(_epoch)
                if min_epoch <= _epoch <= max_epoch:
                    model_path_ls.append(model_path)
            args.model_path = model_path_ls
        else:
            raise NotImplementedError
        assert os.path.exists(config_file), 'Config file not found: {:s}'.format(config_file)
        args.config_file = config_file
    else:
        raise NotImplementedError
    args.mode = mode

    """load config file and overwrite config parameters"""
    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    config = config_dict.ConfigDict(config)
    config.lock()
    args_dict = vars(args)

    # overwrite mcmc parameter
    if args.num_steps is not None:
        print("Overwriting config file: @MCMC sampling steps: {:04d} ---> {:04d}".format(
            config.mcmc.num_steps, args.num_steps))
        config.mcmc.num_steps = args.num_steps

    # overwrite training parameters
    if mode == 'train':
        # overwrite dataset path
        _dataset_overwrite_keywords = ['dataset_name', 'max_node_num', 'subset']
        for keyword in _dataset_overwrite_keywords:
            if args_dict[keyword] is not None:
                _config_key = keyword if keyword != 'dataset_name' else 'name'
                _original_param = config.dataset[_config_key]
                config.dataset[_config_key] = args_dict[keyword]
                print("Overwriting config file: @dataset: {:s}, {} {:s} {}".format(
                    _config_key, _original_param, '------>', args_dict[keyword]))

        # overwrite training parameters
        _train_overwrite_keywords = ['self_cond', 'max_epoch', 'lr_init', 'batch_size',
                                     'sample_interval', 'save_interval',
                                     'node_encoding', 'edge_encoding',
                                     'node_only', 'binary_edge']
        for keyword in _train_overwrite_keywords:
            if args_dict[keyword] is not None:
                _original_param = config.train[keyword]
                config.train[keyword] = args_dict[keyword]
                print("Overwriting config file: @train: {:s}, {} {:s} {}".format(
                    keyword, _original_param, '------>', args_dict[keyword]))

        # resume training from a checkpoint
        with config.unlocked():
            config.train.resume = args_dict['resume']
            if config.train.resume is not None:
                assert os.path.exists(config.train.resume), 'Resume file not found: {:s}'.format(config.train.resume)

        # swinGNN-specific model parameters
        _model_overwrite_keywords = ['feature_dims', 'window_size', 'patch_size']
        for keyword in _model_overwrite_keywords:
            if args_dict[keyword] is not None:
                _original_param = config.model[keyword]
                if isinstance(config.model[keyword], int):
                    config.model[keyword] = args_dict[keyword]
                elif isinstance(config.model[keyword], list):
                    assert len(config.model[keyword]) == 1
                    config.model[keyword] = [args_dict[keyword]]
                else:
                    raise NotImplementedError
                print("Overwriting config file: @model: {:s}, {} {:s} {}".format(
                    keyword, _original_param, '------>', args_dict[keyword]))
    else:
        # eval mode, reset some training parameters to avoid conflicts
        with config.unlocked():
            config.train.resume = None

    # overwrite sampling (test) parameters
    _test_overwrite_keywords = ['eval_size', 'batch_size']
    for keyword in _test_overwrite_keywords:
        if args_dict[keyword] is not None:
            if keyword in config.test:
                _original_param = config.test[keyword]
                config.test[keyword] = args_dict[keyword]
            else:
                _original_param = 'None'
                with config.unlocked():
                    config.test[keyword] = args_dict[keyword]
            print("Overwriting config file: @test: {:s}, {} {:s} {}".format(
                keyword, _original_param, '------>', args_dict[keyword]))

    # overwrite random seed
    if args_dict['seed'] is not None:
        _original_param = config.seed
        config.seed = args_dict['seed']
        print("Overwriting config file: @seed: {} {:s} {}".format(
            _original_param, '------>', args_dict['seed']))

    # add molecule generation flag
    with config.unlocked():
        if config.dataset.name in ['qm9', 'zinc250k']:
            config.flag_mol = True
        else:
            config.flag_mol = False

    return args, config


def set_seed_and_logger(config, log_level, comment, dist_helper, eval_mode=False):
    """
    Set up random seed number and global logger.
    """
    # Setup random seed
    if dist_helper.is_ddp:
        config.seed += dist.get_rank()
    else:
        pass
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # torch numerical accuracy flags
    # reference: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    # add log directory
    str_subset = 'sub_{:03d}'.format(config.dataset.subset) if config.dataset.subset is not None else None
    str_self_cond = 'self-cond-OFF' if config.train.self_cond is False else 'self-cond-ON'

    if 'swin_gnn' in config.model.name:
        _feature_dims = config.model.feature_dims
        _feature_dims = _feature_dims[0] if isinstance(_feature_dims, list) else _feature_dims
        str_feature_dims = 'feat_dim_' + str(_feature_dims)
        str_window_size = 'window_' + str(config.model.window_size)
        str_patch_size = 'patch_' + str(config.model.patch_size)
    else:
        str_feature_dims = None
        str_window_size = None
        str_patch_size = None

    str_node_encoding, str_edge_encoding = None, None
    if config.flag_mol:
        str_node_encoding = 'node_' + config.train.node_encoding
        str_edge_encoding = 'edge_' + config.train.edge_encoding
    str_comment = comment if len(comment) else None
    str_eval = "eval" if eval_mode else None

    str_folder_name = [
        config.dataset.name, config.model.name, config.mcmc.name,
        str_subset, str_self_cond, str_feature_dims, str_window_size, str_patch_size,
        str_node_encoding, str_edge_encoding, str_comment, str_eval,
        time.strftime('%b-%d-%H-%M-%S')
    ]
    logdir = '_'.join([item for item in str_folder_name if item is not None])
    logdir = os.path.join(config.exp_dir, config.exp_name, logdir)

    with config.unlocked():
        config.logdir = logdir
        config.model_save_dir = os.path.join(logdir, 'models')
        config.model_ckpt_dir = os.path.join(logdir, 'models_ckpt')
        if 'dev' in config:
            # reset device if it is already set
            config.dev = None
        config.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.logdir, exist_ok=True)
    if not eval_mode:
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.model_ckpt_dir, exist_ok=True)

    # dump config to yaml file
    yaml_save_path = os.path.join(config.logdir, 'config.yaml')
    with open(yaml_save_path, 'w') as f:
        config_dict_ = config.to_dict()
        config_dict_['dev'] = str(config.dev)
        yaml.dump(config_dict_, f)

    # setup logger
    if dist_helper.is_ddp:
        log_file = "ddp_rank_{:02d}_".format(dist.get_rank()) + log_level.lower() + ".log"
    else:
        log_file = log_level.lower() + ".log"
    if eval_mode:
        log_file = 'eval_' + log_file
    log_file = os.path.join(logdir, log_file)
    log_format = comment + '| %(asctime)s %(message)s'
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG, format=log_format,
                        datefmt='%m-%d %H:%M:%S',
                        handlers=[fh, logging.StreamHandler(sys.stdout)])

    # avoid excessive logging messages
    logging.getLogger('PIL').setLevel(logging.WARNING)  # avoid PIL logging pollution
    logging.getLogger('matplotlib').setLevel(logging.INFO)  # remove excessive matplotlib messages
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)  # remove excessive matplotlib messages

    logging.info('EXPERIMENT BEGIN: ' + comment)
    logging.info('logging into %s', log_file)

    # setup tensorboard logger
    if get_ddp_save_flag():
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = None
    return writer


def backup_code(config, config_file_path):
    logging.info('Config: \n' + pformat(config))
    if get_ddp_save_flag():
        code_path = os.path.join(config.logdir, 'code')
        dirs_to_save = ['loss', 'model', 'runner', 'utils']
        os.makedirs(code_path, exist_ok=True)
        if config_file_path is not None:
            shutil.copy(os.path.abspath(config_file_path), os.path.join(config.logdir, 'config_original.yaml'))

        os.system('cp ./*py ' + code_path)
        [shutil.copytree(os.path.join('./', this_dir), os.path.join(code_path, this_dir)) for this_dir in dirs_to_save]


def set_training_loss_logger(save_dir):
    """
    Setup separated log files for training time losses.
    """
    log_train_loss = os.path.join(save_dir, 'train_loss.log')
    log_test_loss = os.path.join(save_dir, 'test_loss.log')
    f_train_loss = open(log_train_loss, 'w')
    f_test_loss = open(log_test_loss, 'w')
    logging.info("Training and validation loss are recorded at {:s} and {:s} respectively".format(
        log_train_loss, log_test_loss))
    return f_train_loss, f_test_loss


def get_gpu_memory_status(visible=True):
    """
    Print GPU memory status.
    """
    current_usage = []
    for i in range(torch.cuda.device_count()):
        current_usage.append((torch.cuda.mem_get_info(i)[1] - torch.cuda.mem_get_info(i)[0]) / 1024**2)
        if visible:
            logging.info("GPU ID: {:d}, occupied: {:.1f} MB / {:.1f} MB".format(
                i,
                (torch.cuda.mem_get_info(i)[1] - torch.cuda.mem_get_info(i)[0]) / 1024**2,
                torch.cuda.mem_get_info(i)[1] / 1024**2))
    return current_usage

