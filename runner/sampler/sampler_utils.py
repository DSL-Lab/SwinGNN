import logging
import os

import networkx as nx
import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader, random_split

from matplotlib import pyplot as plt

from evaluation.stats import eval_torch_batch
from utils.graph_utils import mask_adjs
from utils.sampling_utils import eval_sample_batch
from utils.visual_utils import plot_graphs_adj


def plot_interim_adjs(interim_adjs, sampling_steps, save_dir, comment=None):
    """
    Plot adjs in colormap plots.
    """
    fig = plt.figure()
    interim_adjs = interim_adjs.cpu().numpy()
    for in_array, cur_step in zip(interim_adjs, sampling_steps):
        plt.clf()
        plt.imshow(in_array, vmin=-3.0, vmax=3.0, cmap='ocean')
        plt.colorbar()
        cur_step = int(cur_step)
        plt.title("Sampling steps: {}. \n Min: {:.4f}, Max: {:.4f}.".format(
            cur_step, in_array.min(), in_array.max()))
        if comment is None:
            file_name = "sample_at_step_{:04d}.png".format(cur_step)
        else:
            file_name = "{:s}_sample_at_step_{:04d}.png".format(comment, cur_step)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_group_interim_adjs(interim_adjs, node_flags, sampling_steps, save_dir, comment=None):
    """
    Plot adjs in colormap plots.
    @param interim_adjs: [K, B, N, N] tensor.
    @param node_flags: [B, N] tensor.
    @param sampling_steps: [K] list.
    @param save_dir: str.
    @param comment: str.
    """
    interim_adjs = interim_adjs.cpu().numpy()
    node_flags = node_flags.cpu().numpy().astype(bool)
    num_images_to_display = 16

    fig = plt.figure()
    for in_array, cur_step in zip(interim_adjs, sampling_steps):
        plt.clf()
        # create a grid of subplots
        num_rows = int(np.sqrt(num_images_to_display))
        num_cols = int(np.ceil(num_images_to_display / num_rows))
        fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=(20, 10))

        # plot raw adjs
        for i in range(num_images_to_display):
            row = i // num_cols
            col = i % num_cols
            i_image = in_array[i][node_flags[i]][:, node_flags[i]]
            axes[row, col].imshow(i_image, vmin=-3.0, vmax=3.0, cmap='ocean')
            axes[row, col].axis('off')
        
        # plot networkx graphs
        for i in range(num_images_to_display):
            row = i // num_cols
            col = i % num_cols + num_cols
            i_image = in_array[i][node_flags[i]][:, node_flags[i]]

            # use 0.0 as threshold
            adj = (i_image > 0.0).astype(int)
            G = nx.from_numpy_matrix(adj)
            assert isinstance(G, nx.Graph)
            G.remove_edges_from(list(nx.selfloop_edges(G)))
            G.remove_nodes_from(list(nx.isolates(G)))

            options = {
                'node_size': 2,
                'edge_color': 'black',
                'linewidths': 1,
                'width': 0.5
            }
            nx.draw(G, pos=nx.spring_layout(G, seed=2023), with_labels=False, **options, ax=axes[row, col])

        cur_step = int(cur_step)

        # set title on top of the overall figure
        plt.suptitle("Sampling step: {:03d}. \nLeft: generated continuous adjacency matrices. Right: corresponding discrete graphs.".format(cur_step), fontsize=20)
        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.90)

        if comment is None:
            file_name = "sample_at_step_{:04d}.png".format(cur_step)
        else:
            file_name = "{:s}_sample_at_step_{:04d}.png".format(comment, cur_step)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rmse(error_per_steps: torch.Tensor, step_per_interim_adjs, save_dir, comment=None):
    """
    Plot RMSE error of interim samples.
    """
    fig = plt.figure()
    error_per_steps = error_per_steps.cpu().numpy()  # [K] steps errors
    sampling_steps = (np.arange(len(error_per_steps)) * step_per_interim_adjs).astype(int)  # [K] steps indices

    ax = plt.gca()
    plt.plot(sampling_steps, error_per_steps, '.-', linewidth=1, markersize=4)
    ax.set_xlabel('Denoising/Sampling step')
    ax.set_ylabel('RMSE (in range of x_0)')

    if comment is None:
        ax.set_title('Interim sample error w.r.t. the x_0')
        file_name = "sample_RMSE.png"
    else:
        ax.set_title('Interim sample error w.r.t. the x_0\n Comment:{:s}'.format(comment))
        file_name = "sample_RMSE_{:s}.png".format(comment)
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_interim_adjs(interim_samples, gt_samples, node_flags, objective, step_per_interim_adjs, plot_save_dir):
    """
    Visualize the interim adjs.
    @param interim_samples: [K, B, N, N] tensor
    @param gt_samples: [B, N, N] tensor
    @param node_flags: [B, N]
    @param objective: str
    @param step_per_interim_adjs: int
    @param plot_save_dir: str
    """

    # Initialization
    interim_samples = interim_samples.cpu() if isinstance(interim_samples, torch.Tensor
                                                          ) else [elem.cpu() for elem in interim_samples]
    _interim_samples = interim_samples
    if objective == 'score':
        num_interim_samples = interim_samples.size(0)
    elif objective == 'diffusion':
        interim_x_t, interim_x_0_pred = interim_samples
        num_interim_samples = interim_x_t.size(0)
    elif objective == 'edm':
        interim_x_t = interim_samples
        num_interim_samples = interim_x_t.size(0)
    else:
        raise NotImplementedError

    num_snapshots = 256
    if num_interim_samples > num_snapshots:
        select_steps = np.linspace(0, num_interim_samples, num_snapshots).astype(int).clip(
            max=num_interim_samples - 1, min=0)
    else:
        select_steps = np.arange(num_interim_samples)

    # Plot interim samples
    logging.info('Visualizing interim adjs...')
    os.makedirs(plot_save_dir, exist_ok=True)

    if objective == 'score':
        interim_samples = interim_samples[:, 0, :, :]  # [K, N, N] <- [K, B, N, N]
        interim_samples = torch.index_select(interim_samples, dim=0, index=torch.tensor(select_steps).long())
        plot_interim_adjs(interim_samples, select_steps * step_per_interim_adjs, plot_save_dir)
    elif objective == 'diffusion':
        interim_x_t = interim_x_t[:, 0, :, :]  # [K, N, N] <- [K, B, N, N]
        interim_x_t = torch.index_select(interim_x_t, dim=0, index=torch.tensor(select_steps).long())
        plot_interim_adjs(interim_x_t, select_steps * step_per_interim_adjs, plot_save_dir, comment='ddpm_x_t')

        interim_x_0_pred = interim_x_0_pred[:, 0, :, :]  # [K, N, N] <- [K, B, N, N]
        interim_x_0_pred = torch.index_select(interim_x_0_pred, dim=0, index=torch.tensor(select_steps).long())
        plot_interim_adjs(interim_x_0_pred, select_steps * step_per_interim_adjs, plot_save_dir, comment='ddpm_x_0_pred')
    elif objective == 'edm':
        interim_x_t = interim_x_t[:, 0, :, :]  # [K, N, N] <- [K, B, N, N]
        interim_x_t = torch.index_select(interim_x_t, dim=0, index=torch.tensor(select_steps).long())
        plot_interim_adjs(interim_x_t, select_steps * step_per_interim_adjs, plot_save_dir, comment='edm_x_t')
        # interim_x_t = torch.index_select(interim_x_t, dim=0, index=torch.tensor(select_steps).long())
        # plot_group_interim_adjs(interim_x_t, node_flags, select_steps * step_per_interim_adjs, plot_save_dir, comment='edm_x_t_group')

    # Evaluate and show interim sample quality (RMSE w.r.t. ground-truth data [-1, 1])
    logging.info('Evaluate interim adjs error...')
    if objective == 'score':
        error_samples = gt_samples.unsqueeze(0) - _interim_samples  # [K, B, N, N]
        error_samples = mask_adjs(error_samples.transpose(0, 1), node_flags)  # [B, K, N, N]
        rmse_per_steps = (error_samples.square().sum(dim=[-1, -2]) / node_flags.sum(dim=1, keepdim=True)).sqrt().mean(
            dim=0)  # [K]
        plot_rmse(rmse_per_steps, step_per_interim_adjs, plot_save_dir, comment=None)
    elif objective == 'diffusion':
        interim_x_t, interim_x_0_pred = _interim_samples
        error_x_t = gt_samples.unsqueeze(0) - interim_x_t  # [K, B, N, N]
        error_x_t = mask_adjs(error_x_t.transpose(0, 1), node_flags)  # [B, K, N, N]
        rmse_x_t = (error_x_t.square().sum(dim=[-1, -2]) / node_flags.sum(dim=1, keepdim=True)).sqrt().mean(
            dim=0)  # [K]
        plot_rmse(rmse_x_t, step_per_interim_adjs, plot_save_dir, comment='diffusion_x_t')

        error_x_0_pred = gt_samples.unsqueeze(0) - interim_x_0_pred  # [K, B, N, N]
        error_x_0_pred = mask_adjs(error_x_0_pred.transpose(0, 1), node_flags)  # [B, K, N, N]
        rmse_x_0_pred = (error_x_0_pred.square().sum(dim=[-1, -2]) / node_flags.sum(dim=1, keepdim=True)).sqrt().mean(
            dim=0)  # [K]
        plot_rmse(rmse_x_0_pred, step_per_interim_adjs, plot_save_dir, comment='diffusion_x_0')
    elif objective == 'edm':
        interim_x_t = _interim_samples
        error_x_t = gt_samples.unsqueeze(0) - interim_x_t  # [K, B, N, N]
        error_x_t = mask_adjs(error_x_t.transpose(0, 1), node_flags)  # [B, K, N, N]
        rmse_x_t = (error_x_t.square().sum(dim=[-1, -2]) / node_flags.sum(dim=1, keepdim=True)).sqrt().mean(dim=0)  # [K]
        plot_rmse(rmse_x_t, step_per_interim_adjs, plot_save_dir, comment='diffusion_x_t')
    else:
        raise NotImplementedError


def visualize_evaluate_final_adjs(init_adjs, sampled_adjs, gt_adjs, node_flags, mc_sampler, title, save_dir):
    """
    Visualize and evaluate the final sample.
    """
    sampled_adjs_int = mc_sampler.adj_to_int(sampled_adjs, node_flags, 0.5)
    init_adjs_int = mc_sampler.adj_to_int(init_adjs, node_flags, 0.5)
    gt_adjs_int = mc_sampler.adj_to_int(gt_adjs, node_flags, 0.5)

    eval_sample_batch(sampled_adjs_int, gt_adjs_int, init_adjs_int, save_dir, title=title, threshold=0.5)

    plot_graphs_adj(init_adjs_int, title=title.replace('.png', '_init.png'), save_dir=save_dir)
    plot_graphs_adj(gt_adjs_int, title=title.replace('.png', '_gt.png'), save_dir=save_dir)

    result_dict = eval_torch_batch(gt_adjs_int, sampled_adjs_int, methods=None)
    return result_dict


def split_test_set(test_dl, total_samples, batch_size, dist_helper, seed=None):
    """
    Split the testing dataset to match the number of samples to be generated.
    """
    # dataset_select, dataset_discard = None, None
    if total_samples <= len(test_dl.dataset):
        # to generate fewer samples than the test set, we can just randomly select a subset of the test set
        split_seed = 42 if seed is None else seed
        dataset_select, dataset_discard = random_split(test_dl.dataset, [total_samples, len(test_dl.dataset) - total_samples],
                                                       generator=torch.Generator().manual_seed(split_seed))
    else:
        # to generate more samples than the test set, we need to repeat the test set
        _num_residue = total_samples % len(test_dl.dataset)
        _num_repeat = total_samples // len(test_dl.dataset)
        if _num_residue == 0:
            dataset_select = torch.utils.data.ConcatDataset([test_dl.dataset] * _num_repeat)
        else:
            _num_repeat = total_samples // len(test_dl.dataset)
            dataset_residue, _ = random_split(test_dl.dataset, [_num_residue, len(test_dl.dataset) - _num_residue], generator=torch.Generator().manual_seed(42))
            dataset_select = torch.utils.data.ConcatDataset([test_dl.dataset] * _num_repeat + [dataset_residue])

    if dist_helper.is_ddp:
        sampler = DistributedSampler(dataset_select)
        batch_size_per_gpu = max(1, batch_size // dist.get_world_size())
        sampler_dl = DataLoader(dataset_select, sampler=sampler, batch_size=batch_size_per_gpu,
                                pin_memory=False, num_workers=min(6, os.cpu_count()))
    else:
        sampler_dl = DataLoader(dataset_select, batch_size=batch_size, shuffle=False,
                                pin_memory=False, num_workers=min(6, os.cpu_count()))

    return sampler_dl
