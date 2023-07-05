import logging
import os
import pdb
import time

import torch
from torch import nn as nn

from runner.sampler.sampler import go_sampling
from runner.sanity_check_helper import sanity_check_training_objectives
from runner.trainer.trainer_utils import get_logger_per_epoch, update_epoch_learning_status, print_epoch_learning_status, \
    check_best_model, save_ckpt_model, permute_aug, compute_gt_score
from utils.arg_parser import set_training_loss_logger
from utils.dist_training import get_ddp_save_flag


def move_forward_one_epoch(model, optimizer, ema_helper, dataloader, train_obj_gen, loss_func, epoch_logger,
                           mode, sanity_check_save_dir=None,
                           flag_matching=False, flag_gt_score_pred=False, flag_permutation_aug=False):
    """
    Go through one epoch of data. Compatible with training and testing.
    """
    assert mode in ['train', 'test']
    epoch_logger[mode]['time_start'] = time.time()
    sanity_check_flag = epoch_logger['epoch'] == 0 and mode == 'train'
    for data_tuple in dataloader:

        """Initialization"""
        if len(data_tuple) == 2:
            adjs_gt, x_gt = data_tuple
            node_flags = adjs_gt.abs().sum(-1).gt(1e-5).to(dtype=torch.float32)  # [B, N]
        elif len(data_tuple) == 3:
            adjs_gt, x_gt, node_flags = data_tuple
        else:
            raise NotImplementedError

        # enforce a large batch size, to stack the graphs multiple times
        if len(adjs_gt) < dataloader.batch_size and dataloader.batch_size % len(adjs_gt) == 0:
            if hasattr(dataloader, 'repeated_data'):
                adjs_gt, x_gt, node_flags = dataloader.repeated_data
            else:
                num_repeat = dataloader.batch_size // len(adjs_gt)
                adjs_gt = adjs_gt.repeat(num_repeat, 1, 1)
                x_gt = x_gt.repeat(num_repeat, 1, 1)
                node_flags = node_flags.repeat(num_repeat, 1)
                repeated_data = [adjs_gt, x_gt, node_flags]
                dataloader.repeated_data = repeated_data

        adjs_gt = adjs_gt.to(train_obj_gen.dev)  # [B, N, N]
        x_gt = x_gt.to(train_obj_gen.dev)  # [B, N, 1], all zeros by default
        node_flags = node_flags.to(train_obj_gen.dev)  # [B, N]

        if train_obj_gen.objective == 'diffusion':
            # for diffusion model:      (noisy adj matrices, diffusion step indexes, prediction target)
            net_input, net_cond, net_target = train_obj_gen.get_input_output(adjs_gt, node_flags)
            cond_val = torch.index_select(train_obj_gen.const_beta_t, 0, net_cond.long())
            weights = None
            # scaling_coef = torch.index_select(train_obj_gen.const_alpha_t_bar, 0, net_cond.long()).sqrt()
        elif train_obj_gen.objective == 'score':
            raise NotImplementedError
            # cond_val = torch.index_select(train_obj_gen.const_sigma_t, 0, net_cond.long())
            # scaling_coef = torch.ones_like(net_cond)
        elif train_obj_gen.objective == 'edm':
            net_input, net_cond, net_target, (c_skip, c_out, c_in, c_noise, sigmas, weights) = train_obj_gen.get_input_output(adjs_gt, node_flags)
            cond_val = None
        else:
            raise NotImplementedError

        """DEBUG OPTIONS"""
        # use the ground-truth score function as training target
        if flag_gt_score_pred:
            net_target = compute_gt_score(adjs_gt, net_input, net_cond, train_obj_gen)

        # permute the network input-output
        if flag_permutation_aug:
            net_input, net_target, node_flags = permute_aug(net_input, net_target, node_flags)

        # sanity check: visualize intermediate states
        if sanity_check_flag and get_ddp_save_flag():
            sanity_check_training_objectives(adjs_gt, node_flags, train_obj_gen, sanity_check_save_dir)
            sanity_check_flag = False  # run at most once

        """Network forward pass"""
        if train_obj_gen.objective == 'diffusion':
            def _diffusion_model_pass():
                net_out = model(net_input, node_flags, net_cond)
                return net_out

            if mode == 'train':
                optimizer.zero_grad(set_to_none=True)
                net_output = _diffusion_model_pass()
            elif mode == 'test':
                with torch.no_grad():
                    net_output = _diffusion_model_pass()

            regression_loss = loss_func(net_pred=net_output,
                                        net_target=net_target,
                                        net_cond=net_cond,
                                        adjs_perturbed=net_input,
                                        adjs_gt=adjs_gt,
                                        node_flags=node_flags,
                                        cond_val=cond_val,
                                        flag_matching=flag_matching,
                                        reduction='none')  # [B]
            loss = regression_loss.mean()
        elif train_obj_gen.objective == 'edm':
            # Network forward pass
            def _edm_model_pass():
                # the model is with the precond module
                net_out = model(x=net_input, node_flags=node_flags, sigmas=sigmas)
                return net_out

            if mode == 'train':
                optimizer.zero_grad(set_to_none=True)
                net_output = _edm_model_pass()
            elif mode == 'test':
                with torch.no_grad():
                    net_output = _edm_model_pass()

            regression_loss = loss_func(net_pred=net_output,
                                        net_target=net_target,
                                        net_cond=net_cond,
                                        adjs_perturbed=net_input,
                                        adjs_gt=adjs_gt,
                                        node_flags=node_flags,
                                        flag_matching=flag_matching,
                                        loss_weight=weights,
                                        reduction='none')  # [B]
            loss = regression_loss.mean()
        else:
            raise NotImplementedError

        """Network backward pass"""
        if mode == 'train':
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)  # clip gradient
            optimizer.step()
            if ema_helper is not None:
                # we maintain a list EMA helper to handle multiple EMA coefficients
                [ema.update() for ema in ema_helper]

        """Record training result per iteration"""
        update_epoch_learning_status(epoch_logger, mode, reg_loss=regression_loss.detach(),
                                     noise_label=net_cond.detach())


def go_training(model, optimizer, scheduler, ema_helper,
                train_dl, test_dl, train_obj_gen, loss_func, mc_sampler, config, dist_helper, writer):
    """
    Core training functions go here.
    """

    """Initialization"""
    lowest_loss = {"epoch": -1, "loss": float('inf')}

    # Build txt loss file handler dedicated to training / evaluation loss per sample
    if get_ddp_save_flag():
        f_train_loss, f_test_loss = set_training_loss_logger(config.logdir)
    else:
        f_train_loss, f_test_loss = None, None

    save_interval = config.train.save_interval
    sample_interval = config.train.sample_interval
    sanity_check_save_dir = os.path.join(config.logdir, 'sanity_check_training_data')

    flag_matching = config.train.matching
    flag_gt_score_pred = config.train.gt_score_pred
    flag_permutation_aug = config.train.permutation_aug

    """Go training"""
    for epoch in range(config.train.max_epoch):
        """Initialization"""
        epoch_logger = get_logger_per_epoch(epoch, flag_node_adj=False)
        if dist_helper.is_ddp:
            train_dl.sampler.set_epoch(epoch)
            test_dl.sampler.set_epoch(epoch)

        """Start learning"""
        # training
        model.train()
        move_forward_one_epoch(model, optimizer, ema_helper, train_dl, train_obj_gen, loss_func, epoch_logger,
                               'train', sanity_check_save_dir, flag_matching, flag_gt_score_pred, flag_permutation_aug)
        scheduler.step()
        logging.debug("epoch: {:05d}| effective learning rate: {:12.6f}".format(epoch, optimizer.param_groups[0]["lr"]))

        # testing
        if epoch % save_interval == save_interval - 1 or epoch == 0:
            if ema_helper is not None:
                test_model = ema_helper[0].ema_model
            else:
                test_model = model
            test_model.eval()

            move_forward_one_epoch(test_model, optimizer, ema_helper, test_dl, train_obj_gen, loss_func, epoch_logger,
                                   'test', sanity_check_save_dir, flag_matching, flag_gt_score_pred, flag_permutation_aug)

            """Network weight saving"""
            # check best model
            check_best_model(model, ema_helper, epoch_logger, lowest_loss, save_interval, config, dist_helper)
            # save checkpoint model
            save_ckpt_model(model, ema_helper, epoch_logger, config, dist_helper)

        dist_helper.ddp_sync()

        # show the training and testing status
        print_epoch_learning_status(epoch_logger, f_train_loss, f_test_loss, writer, config.mcmc.name, flag_node_adj=False)

        """Sampling during training"""
        if epoch % sample_interval == sample_interval - 1 or epoch == 0:
            if ema_helper is not None:
                test_model = ema_helper[-2].ema_model  # use the second last EMA coefficient for sampling
                ema_beta = ema_helper[-2].beta
            else:
                test_model = model
                ema_beta = 1.0
            test_model.eval()
            sampling_params = {'model_nm': 'training_e{:05d}'.format(epoch),
                               'weight_kw': '{:.3f}'.format(ema_beta),
                               'model_path': os.path.join(config.model_ckpt_dir,
                                                          f"{config.dataset.name}_{epoch:05d}.pth")}

            if epoch == 0:
                go_sampling(epoch, test_model, dist_helper, test_dl, mc_sampler, config, writer, sanity_check=True,
                            sampling_params=sampling_params, eval_mode=False)
            else:
                go_sampling(epoch, test_model, dist_helper, test_dl, mc_sampler, config, writer, sanity_check=False,
                            sampling_params=sampling_params, eval_mode=False)

    # Destroy dedicated txt logger
    if get_ddp_save_flag():
        f_train_loss.close()
        f_test_loss.close()
