import logging

from runner.trainer.trainer_node_adj import node_adj_go_training
from runner.trainer.trainer import go_training
from utils.arg_parser import parse_arguments, set_seed_and_logger, backup_code
from utils.learning_utils import get_network, get_optimizer, get_training_objective_generator, \
                                 get_rainbow_loss, get_ema_helper
from utils.dataloader import load_data
from utils.sampling_utils import get_mc_sampler
from utils.dist_training import DistributedHelper


def init_basics(mode='train'):
    # Initialization
    args, config = parse_arguments(mode=mode)
    dist_helper = DistributedHelper(args.dp, args.ddp, args.ddp_gpu_ids, args.ddp_init_method)
    writer = set_seed_and_logger(config, args.log_level, args.comment, dist_helper, eval_mode=mode == 'eval')
    backup_code(config, args.config_file)
    return args, config, dist_helper, writer


def init_model(config, dist_helper):
    # Initialize training objective generator
    train_obj_gen = get_training_objective_generator(config)

    # Initialize MCMC sampler
    mc_sampler = get_mc_sampler(config)

    # Initialize network model & optimizer
    model = get_network(config, dist_helper)
    optimizer, scheduler = get_optimizer(model, config, dist_helper)

    # Initialize EMA helper
    ema_helper = get_ema_helper(config, model)

    # Initialize loss function
    loss_func = get_rainbow_loss(config)
    return train_obj_gen, mc_sampler, model, optimizer, scheduler, ema_helper, loss_func


def main():
    """
    Training begins here!
    """

    """Initialize basics"""
    args, config, dist_helper, writer = init_basics()

    """Get dataloader"""
    train_dl, test_dl = load_data(config, dist_helper)

    """Get network"""
    train_obj_gen, mc_sampler, model, optimizer, scheduler, ema_helper, loss_func = init_model(config, dist_helper)

    """Go training"""
    if config.flag_mol:
        node_adj_go_training(model, optimizer, scheduler, ema_helper,
                             train_dl, test_dl, train_obj_gen, loss_func, mc_sampler, config, dist_helper, writer)
    else:
        go_training(model, optimizer, scheduler, ema_helper,
                    train_dl, test_dl, train_obj_gen, loss_func, mc_sampler, config, dist_helper, writer)

    # Clean up DDP utilities after training
    dist_helper.clean_up()

    logging.info('TRAINING IS FINISHED.')


if __name__ == "__main__":
    main()
