import copy
import logging
import os
import time

import numpy as np
import torch

from runner.sanity_check_helper import get_all_permutation, compute_score
from utils.dist_training import get_ddp_save_flag, dist_save_model


def get_logger_per_epoch(epoch, flag_node_adj):
    """
    Create dict to save learning status at the beginning of each epoch.
    """
    _loss_status = {
        'summed_loss': [],
        'time_start': None,
        'time_elapsed': None,
        'noise_label': []
    }
    if flag_node_adj:
        _loss_status['reg_loss_adj'] = []
        _loss_status['reg_loss_node'] = []
    else:
        _loss_status['regression_loss'] = []

    loss_status_ls = [copy.deepcopy(_loss_status) for _ in range(2)]

    logger = {'train': loss_status_ls[0],
              'test': loss_status_ls[1],
              'epoch': epoch}
    return logger


def update_epoch_learning_status(epoch_logger, mode, reg_loss=None,
                                 reg_loss_adj=None, reg_loss_node=None, noise_label=None):
    """
    Update learning status dict.
    """
    assert mode == 'train' or 'test'

    if reg_loss is not None:
        assert reg_loss_adj is None and reg_loss_node is None
        epoch_logger[mode]['regression_loss'].append(reg_loss.cpu().numpy())
        epoch_logger[mode]['summed_loss'].append(reg_loss.cpu().numpy())
    else:
        epoch_logger[mode]['reg_loss_adj'].append(reg_loss_adj.cpu().numpy())
        epoch_logger[mode]['reg_loss_node'].append(reg_loss_node.cpu().numpy())
        epoch_logger[mode]['summed_loss'].append((reg_loss_adj + reg_loss_node).cpu().numpy())

    epoch_logger[mode]['noise_label'].append(noise_label.cpu().numpy())
    if epoch_logger[mode]['time_start'] is None:
        epoch_logger[mode]['time_start'] = time.time()
    else:
        # update each time for convenience, only the last timestamp is useful
        epoch_logger[mode]['time_elapsed'] = time.time() - epoch_logger[mode]['time_start']
    return epoch_logger


def print_epoch_learning_status(epoch_logger, f_train_loss, f_test_loss, writer, objective, flag_node_adj):
    """
    Show the learning status of this epoch.
    """
    epoch = epoch_logger['epoch']

    def _write_to_file_handler(np_array_data, file_handler, line_sampling_freq):
        for i_line, line in enumerate(np_array_data):
            if i_line % line_sampling_freq == 0:
                line_str = np.array2string(line, formatter={'float_kind': lambda x: "%.6f" % x}, separator=" ")
                file_handler.write(line_str[1:-1] + '\n')
        file_handler.flush()

    for mode, f_handler in zip(['train', 'test'], [f_train_loss, f_test_loss]):

        flag_empty = len(epoch_logger[mode]['summed_loss']) == 0

        if not flag_empty:
            summed_loss = np.concatenate(epoch_logger[mode]['summed_loss'])  # array, [N]
            time_elapsed = epoch_logger[mode]['time_elapsed']  # scalar
            noise_label = np.concatenate(epoch_logger[mode]['noise_label'])  # array, [N]
            i_iter = epoch_logger['epoch'] * len(summed_loss)

            if flag_node_adj:
                reg_loss_node = np.concatenate(epoch_logger[mode]['reg_loss_node'])  # array, [N]
                reg_loss_adj = np.concatenate(epoch_logger[mode]['reg_loss_adj'])
                logging.info(f'epoch: {epoch:05d}| {mode:5s} | '
                             f'total loss: {np.mean(summed_loss):10.6f} | '
                             f'{objective:s} adj_loss: {np.mean(reg_loss_adj):10.6f} | '
                             f'node_loss: {np.mean(reg_loss_node):10.6f} | '
                             f'time: {time_elapsed:5.2f}s | ')

                down_sampling_freq = 1000
                if get_ddp_save_flag():
                    # record epoch-wise and sample-wise training status into tensorboard
                    cat_loss = np.stack([noise_label, reg_loss_adj, reg_loss_node], axis=1)  # array, [N, X]
                    writer.add_scalar("{:s}_epoch/loss_adj".format(mode), np.mean(reg_loss_adj), epoch)
                    writer.add_scalar("{:s}_epoch/loss_node".format(mode), np.mean(reg_loss_node), epoch)
                    for i in range(len(cat_loss)):
                        if i % down_sampling_freq == 0:
                            writer.add_scalar("{:s}_sample/loss_adj".format(mode), reg_loss_adj[i], i + i_iter)
                            writer.add_scalar("{:s}_sample/loss_node".format(mode), reg_loss_node[i], i + i_iter)
                            writer.add_scalar("{:s}_sample/noise_label".format(mode), noise_label[i], i + i_iter)
                    writer.flush()
            else:
                regression_loss = np.concatenate(epoch_logger[mode]['regression_loss'])  # array, [N]
                logging.info(f'epoch: {epoch:05d}| {mode:5s} | '
                             f'total loss: {np.mean(summed_loss):10.6f} | '
                             f'{objective:s} loss: {np.mean(regression_loss):10.6f} | '
                             f'time: {time_elapsed:5.2f}s | ')

                down_sampling_freq = 1
                if get_ddp_save_flag():
                    # record epoch-wise and sample-wise training status into tensorboard
                    cat_loss = np.stack([noise_label, regression_loss], axis=1)  # array, [N, X]
                    writer.add_scalar("{:s}_epoch/loss".format(mode), np.mean(regression_loss), epoch)
                    for i in range(len(cat_loss)):
                        writer.add_scalar("{:s}_sample/loss".format(mode), regression_loss[i], i + i_iter)
                        writer.add_scalar("{:s}_sample/noise_label".format(mode), noise_label[i], i + i_iter)
                    writer.flush()

            if get_ddp_save_flag():
                # record sample-wise training status into txt file
                _write_to_file_handler(cat_loss, f_handler, down_sampling_freq)


def check_best_model(model, ema_helper, epoch_logger, best_model_status, save_interval, config, dist_helper):
    """
    Check if the latest training leads to a better model.
    """
    if get_ddp_save_flag():
        lowest_loss = best_model_status["loss"]
        mean_train_loss = np.concatenate(epoch_logger['train']['summed_loss']).mean()
        mean_test_loss = np.concatenate(epoch_logger['test']['summed_loss']).mean()
        epoch = epoch_logger['epoch']
        if lowest_loss > mean_test_loss and epoch > save_interval:
            best_model_status["epoch"] = epoch
            best_model_status["loss"] = mean_test_loss
            to_save = get_ckpt_data(model, ema_helper, epoch, mean_train_loss, mean_test_loss, config, dist_helper)

            # save to model checkpoint dir (many network weights)
            to_save_path = os.path.join(config.model_ckpt_dir, f"{config.dataset.name}_best.pth")
            dist_save_model(to_save, to_save_path)
            logging.info(f"epoch: {epoch:05d}| best model updated at {to_save_path:s}")

            # save to best model storage directory (single network weight)
            to_save_path = os.path.join(config.model_save_dir, f"{config.dataset.name}_best.pth")
            dist_save_model(to_save, to_save_path)


def save_ckpt_model(model, ema_helper, epoch_logger, config, dist_helper):
    """
    Save the checkpoint weight.
    """
    mean_train_loss = np.concatenate(epoch_logger['train']['summed_loss']).mean()
    mean_test_loss = np.concatenate(epoch_logger['test']['summed_loss']).mean()
    epoch = epoch_logger['epoch']
    to_save = get_ckpt_data(model, ema_helper, epoch, mean_train_loss, mean_test_loss, config, dist_helper)
    to_save_path = os.path.join(config.model_ckpt_dir, f"{config.dataset.name}_{epoch:05d}.pth")
    dist_save_model(to_save, to_save_path)


def get_ckpt_data(model, ema_helper, epoch, train_loss, test_loss, config, dist_helper):
    """
    Create a dictionary containing necessary stuff to save.
    """
    to_save = {
        'model': model.state_dict(),
        'config': config.to_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'test_loss': test_loss
    }

    if ema_helper is not None:
        for ema in ema_helper:
            beta = ema.beta
            to_save['model_ema_beta_{:.4f}'.format(beta)] = ema.ema_model.state_dict()

    return to_save


def permute_aug(net_input, net_target, node_flags):
    """permute the network input-output"""

    # input-output is [B, N, N]
    num_nodes = node_flags.sum(dim=-1).cpu().numpy().astype('int')  # [B]
    max_nodes = node_flags.size(1)

    def _get_one_permutation_mat(num_node: int, max_node: int):
        to_index = np.random.permutation(num_node)
        from_index = np.arange(num_node)
        permutation_matrix = np.zeros((num_node, num_node), dtype=int)
        permutation_matrix[to_index, from_index] = 1
        permutation_matrix = np.pad(permutation_matrix, (0, max_node - num_node),
                                    mode='constant', constant_values=0)
        return permutation_matrix

    perm_mats = [torch.tensor(_get_one_permutation_mat(n, max_nodes)) for n in num_nodes]
    perm_mats = torch.stack(perm_mats, dim=0).to(net_input)  # [B, N, N]
    net_input = torch.einsum('b i j, b j k, b k l -> b i l', perm_mats, net_input, perm_mats.permute(0, 2, 1))
    net_target = torch.einsum('b i j, b j k, b k l -> b i l', perm_mats, net_target, perm_mats.permute(0, 2, 1))
    node_flags = torch.einsum('b i j, b j -> b i', perm_mats, node_flags)

    return net_input, net_target, node_flags


def compute_gt_score(adjs_gt, net_input, net_cond, train_obj_gen):
    """
    use the ground-truth score function as training target
    """
    alpha_t_bar = torch.index_select(train_obj_gen.const_alpha_t_bar, 0, net_cond.long()).item()
    if net_input.size(0) == 1:
        x_vec = net_input.view(-1)  # [N^2]
        num_nodes = net_input.size(1)

        if not hasattr(train_obj_gen, 'x0_all_perm_vec'):
            all_perm_mats = torch.from_numpy(get_all_permutation(num_nodes)).to(adjs_gt)  # [X, N, N]
            num_perm = all_perm_mats.size(0)
            x0_all_perm = adjs_gt.expand(num_perm, -1, -1)  # [X, N, N]
            x0_all_perm = all_perm_mats @ x0_all_perm @ all_perm_mats.transpose(-1, -2)  # [X, N, N]
            x0_all_perm_vec = x0_all_perm.view(num_perm, -1)  # [X, N^2]
            x0_all_perm_vec = torch.unique(x0_all_perm_vec, dim=0)  # [X, N^2], with a smaller X
            train_obj_gen.x0_all_perm_vec = x0_all_perm_vec
            logging.info("Creating GMM centroids used in the GT score training...")
            logging.info("Number of unique components: {:d}".format(x0_all_perm_vec.size(0)))
        else:
            x0_all_perm_vec = train_obj_gen.x0_all_perm_vec  # [X, N^2]

        mu_vec = x0_all_perm_vec  # [X, N^2]
        gt_score = compute_score(x_vec, mu_vec, alpha_t_bar).view(1, num_nodes,
                                                                  num_nodes)  # [1, N, N] tensor

        # convert back to epsilon prediction target
        net_target = -gt_score * np.sqrt(1.0 - alpha_t_bar)
    else:
        raise NotImplementedError

    return net_target
