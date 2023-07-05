import os

import math
import random
import numpy as np
import networkx as nx
from sympy.utilities.iterables import multiset_permutations
from scipy.stats import norm

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from runner.objectives.diffusion import get_alpha_t_bar
from runner.objectives.edm import get_edm_params, get_preconditioning_params


def _compute_kl_std_normal(mean, std):
    assert len(mean) == len(std)
    kl = np.sum(mean ** 2 + std ** 2 - np.log(std ** 2) - 1, axis=(-1, -2)) * 0.5  # [B]
    return kl


def _to_numpy_array(in_tensor):
    if isinstance(in_tensor, torch.Tensor):
        if in_tensor.numel() > 1:
            return in_tensor.cpu().numpy()
        else:
            return in_tensor.item()
    elif isinstance(in_tensor, tuple) or isinstance(in_tensor, list):
        return (_to_numpy_array(item) for item in in_tensor)
    elif isinstance(in_tensor, np.ndarray):
        return in_tensor


def get_all_permutation(num_node):
    """
    Create all possible permutation matrices given the number of nodes.
    Note: this function does not consider automorphism group.
    """
    perm_mat = []
    all_perm_vecs = multiset_permutations(np.arange(num_node))
    for perm_vec in all_perm_vecs:
        a = np.zeros((num_node, num_node))
        # [N, N]
        np.put_along_axis(a, np.array(perm_vec).reshape(-1, 1), 1.0, axis=1)
        perm_mat.append(a)
    perm_mat = np.stack(perm_mat, axis=0).astype(float)  # [B, N, N]
    assert len(perm_mat) == np.math.factorial(num_node)
    return perm_mat  # [B, N, N], numpy array


def get_random_permutation(num_node, num_perm, seed=2023):
    """
    Create some permutation matrices given the number of nodes.
    Note: this function does not consider automorphism group.
    """
    # control reproducibility
    random.seed(seed)
    np.random.seed(seed)

    perm_mat = []
    random_perm_vecs = [np.random.permutation(np.arange(num_node)) for _ in range(num_perm)]
    for perm_vec in random_perm_vecs:
        a = np.zeros((num_node, num_node))
        # [N, N]
        np.put_along_axis(a, np.array(perm_vec).reshape(-1, 1), 1.0, axis=1)
        perm_mat.append(a)
    perm_mat = np.stack(perm_mat, axis=0).astype(float)  # [B, N, N]
    return perm_mat  # [B, N, N], numpy array


def gmm_log_prob(x_vec, mu_vec, alpha_t_bar):
    """
    Evaluate the log probability of the GMM model for permutation equivariant noisy data distribution,
    where the covariance matrix is isotropic.
    @param x_vec:   [Y, D]
    @param mu_vec:  [X, D]
    @param alpha_t_bar: float
    @return log_prob: [Y], float
    """
    dim = x_vec.size(-1)

    alpha_t_bar_sqrt = np.sqrt(alpha_t_bar)
    coef_one_minus_alpha_t_bar = 1.0 - alpha_t_bar

    scaled_mu_vec = mu_vec * alpha_t_bar_sqrt  # [X, D]
    covariance = coef_one_minus_alpha_t_bar  # scalar

    part_0 = dim * np.log(2 * math.pi * covariance)  # scalar
    diff = x_vec[:, None, :] - scaled_mu_vec[None, :, :]  # [Y, X, D]
    part_1 = diff.square().sum(dim=-1) / covariance  # [Y, X]

    num_perm = np.math.factorial(np.sqrt(dim))
    log_prob_mu = -0.5 * (part_0 + part_1) - np.log(num_perm)  # [Y, X]

    log_prob = torch.logsumexp(log_prob_mu, dim=1)  # [Y]
    log_prob = log_prob.clip(max=0.0)  # enforce log-prob property

    # const_max = log_prob_mu.max()
    # prob_mu = torch.log(torch.exp(log_prob_mu - const_max).sum()) + const_max
    #
    # prob_mu = (2 * math.pi * covariance) ** (-dim / 2) * torch.exp(-0.5 * part_1)  # [X]

    return log_prob


def compute_score(x_vec, mu_vec, alpha_t_bar):
    """
    Compute the ground-truth score function for O(n!) GMM-based probability distribution.
    @param x_vec:   [D]
    @param mu_vec:  [X, D], where X means the number of possible permutations
    @param alpha_t_bar: float
    @return prob: [D]
    """
    alpha_t_bar_sqrt = np.sqrt(alpha_t_bar)
    coef_one_minus_alpha_t_bar = 1.0 - alpha_t_bar

    scaled_mu_vec = mu_vec * alpha_t_bar_sqrt  # [X, D]
    dist_x_to_mu = x_vec[None, :] - scaled_mu_vec  # [X, D]

    logits = dist_x_to_mu.square().sum(dim=-1)  # [X]
    logits = -logits / 2 / coef_one_minus_alpha_t_bar  # [X]
    coef_softmax = -1.0 / coef_one_minus_alpha_t_bar * torch.softmax(logits, dim=0)  # [X]

    gt_score = (coef_softmax[:, None] * dist_x_to_mu).sum(dim=0)  # [D] <- [X, D]
    return gt_score


def plot_diffusion_noisy_data(noisy_adjs, adjs_gt, timesteps, scaling_coef, noise_coef, plot_dir):
    """
    Plot the noisy data and some of its augmentations.
    @param noisy_adjs:                  [B, N, N]
    @param adjs_gt:                     [N, N]
    @param timesteps:                   [B]
    @param scaling_coef:                [B], square root of alpha_t_bar
    @param noise_coef:                  [B], 1 - alpha_t_bar
    @param plot_dir:                    str
    """

    """Init"""
    num_node = adjs_gt.size(-1)
    os.makedirs(plot_dir, exist_ok=True)

    noisy_adjs_ = noisy_adjs / scaling_coef.view(-1, 1, 1)                    # [B, N, N]
    noisy_adjs_scaled = noisy_adjs_ / 2.0 + 0.5                               # [B, N, N], no clipping
    noisy_adjs_shrunk = torch.clip(noisy_adjs_scaled, min=0.0, max=1.0)       # [B, N, N], clipped
    # normalization
    adjs_de_deg_mat = torch.diag_embed(noisy_adjs_shrunk.sum(dim=-1)).pow(-0.5)  # [B, N, N]
    adjs_de_deg_mat.masked_fill_(adjs_de_deg_mat == float('inf'), 0)
    noisy_adjs_normal = adjs_de_deg_mat @ noisy_adjs_shrunk @ adjs_de_deg_mat.transpose(-1, -2)  # [B, N, N]

    """Plot loop"""
    n_rows, n_cols = 2, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 10))
    i_iter = 0
    in_data_gt = adjs_gt.cpu().numpy()  # [N, N]
    for in_data_shrunk, in_data_scaled, in_data_normal, in_data_raw, in_cond in zip(
            noisy_adjs_shrunk, noisy_adjs_scaled, noisy_adjs_normal, noisy_adjs, timesteps):

        [axes[i, j].clear() for i in range(n_rows) for j in range(n_cols)]
        cb_ls = []
        in_data_shrunk, in_data_scaled, in_data_normal, in_data_raw = _to_numpy_array(
            (in_data_shrunk, in_data_scaled, in_data_normal, in_data_raw))
        discrete_thresholds = np.linspace(-1.0, 1.0, n_cols + 2)[1:-1]

        for row in range(n_rows):
            for col in range(n_cols):
                if row == 0:
                    # plot original matrix
                    if col == 0:
                        data, v_min, v_max, str_title = in_data_shrunk, 0.0, 1.0, 'Shrunk adjs (clipped)'
                    elif col == 1:
                        data, v_min, v_max, str_title = in_data_scaled, 0.0, 1.0, 'Scaled adjs (no clipping)'
                    elif col == 2:
                        data, v_min, v_max, str_title = in_data_normal, 0.0, 1.0, 'Normalized adjs'
                    elif col == 3:
                        data, v_min, v_max, str_title = in_data_raw, -3.0, 3.0, 'Raw adjs'
                    elif col == 4:
                        in_data_x0 = in_data_gt * scaling_coef[i_iter].item()
                        data, v_min, v_max, str_title = in_data_x0, -3.0, 3.0, 'X0 part'
                    elif col == 5:
                        in_data_eps = in_data_raw - in_data_x0
                        data, v_min, v_max, str_title = in_data_eps, -3.0, 3.0, 'Noise part'
                    else:
                        raise NotImplementedError

                    im = axes[row, col].imshow(data, vmin=v_min, vmax=v_max, cmap='ocean')
                    divider = make_axes_locatable(axes[row, col])
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    cb = plt.colorbar(im, cax=cax, orientation='vertical')
                    cb_ls.append(cb)
                    axes[row, col].set_title("{:s} Diffusion step: {:03d}."
                                             "\nMin: {:.4f}, Max: {:.4f}.".format(str_title, in_cond,
                                                                                  data.min(), data.max()))
                elif row == 1:
                    # plot discretized adjs
                    data_discrete = in_data_raw.copy() >= discrete_thresholds[col]
                    data, v_min, v_max = data_discrete, 0.0, 1.0

                    im = axes[row, col].imshow(data, vmin=v_min, vmax=v_max, cmap='ocean')
                    divider = make_axes_locatable(axes[row, col])
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    cb = plt.colorbar(im, cax=cax, orientation='vertical')
                    cb_ls.append(cb)
                    axes[row, col].set_title("Discretized adjs th={:.4f}\n"
                                             "Diffusion step: {:03d}".format(discrete_thresholds[col], in_cond))

        plot_path = os.path.join(plot_dir, "training_sample_step_{:03d}.png".format(in_cond))
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        for cb in cb_ls:
            cb.remove()
        i_iter += 1
    plt.close(fig)


def plot_gt_score_sampling(noisy_adjs, adjs_gt, timesteps, scaling_coef, noise_coef, plot_dir,
                           flag_noise_injection, flag_repetitive):
    """
    Plot the sampling results using ground-truth score function at a single diffusion step.
    Note: we could repeat the denoising steps multiple times but the scaling and sigma coefficients are preserved.
    @param noisy_adjs:                  [B, N, N]
    @param adjs_gt:                     [N, N]
    @param timesteps:                   [B]
    @param scaling_coef:                [B], square root of alpha_t_bar
    @param noise_coef:                  [B], 1 - alpha_t_bar
    @param plot_dir:                    str
    @param flag_noise_injection:        bool
    @param flag_repetitive:             bool
    """

    """Init"""
    num_node = adjs_gt.size(-1)
    os.makedirs(plot_dir, exist_ok=True)

    # compute all permuted clean data
    all_perm_mats = torch.from_numpy(get_all_permutation(num_node)).to(adjs_gt)  # [X, N, N]
    num_perm = all_perm_mats.size(0)
    x0_all_perm = adjs_gt.repeat(num_perm, 1, 1)  # [X, N, N]
    x0_all_perm = all_perm_mats @ x0_all_perm @ all_perm_mats.transpose(-1, -2)  # [X, N, N]
    x0_all_perm_vec = x0_all_perm.view(num_perm, -1)  # [X, N^2]
    x0_all_perm_vec = torch.unique(x0_all_perm_vec, dim=0)  # [X, N^2], with a smaller X

    """Plot loop"""
    n_rows, n_cols = 4, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(45, 25))
    i_iter = 0
    in_data_gt = adjs_gt.cpu().numpy()  # [N, N]
    consecutive_xt_ls = []

    # flip the order, start from the largest diffusion step
    noisy_adjs = noisy_adjs.flip(dims=[0])
    timesteps = timesteps.flip(dims=[0])
    scaling_coef = scaling_coef.flip(dims=[0])
    noise_coef = noise_coef.flip(dims=[0])

    for in_data_raw, in_cond, in_scaling_coef, in_noise_coef in zip(noisy_adjs, timesteps, scaling_coef, noise_coef):

        # init each figure
        [axes[i, j].clear() for i in range(n_rows) for j in range(n_cols)]
        cb_ls = []
        in_data_raw, in_scaling_coef, in_noise_coef = _to_numpy_array((in_data_raw, in_scaling_coef, in_noise_coef))

        # load params related to diffusion step
        alpha_t_bar = np.square(in_scaling_coef)
        coef_xt = 1.0 / np.sqrt(alpha_t_bar)
        coef_score = (1.0 - alpha_t_bar) / np.sqrt(alpha_t_bar)
        mu_t = in_data_gt * in_scaling_coef

        # repetitive denoising using gt score
        if flag_repetitive:
            # load the noisy data from diffusion process
            input_xt_sample = in_data_raw
        else:
            # load the noisy data from last time's denoising result
            if i_iter == 0:
                consecutive_xt_ls.append(in_data_raw)
            input_xt_sample = consecutive_xt_ls[-1]
        rep_xt, rep_x0, rep_gt_score = [input_xt_sample], [input_xt_sample / in_scaling_coef], []

        for i in range(20):
            this_xt_est = rep_xt[-1]  # [N, N] np array
            this_gt_score = compute_score(x_vec=torch.tensor(this_xt_est).view(-1).to(adjs_gt),  # [N*N] tensor
                                          mu_vec=x0_all_perm_vec,  # [X, N*N] tensor
                                          alpha_t_bar=alpha_t_bar  # float
                                          ).view(num_node, num_node).cpu().numpy()  # [N, N] np array

            this_x0_est = coef_xt * this_xt_est + coef_score * this_gt_score  # [N, N] np array
            new_xt_est = this_x0_est * in_scaling_coef  # [N, N] np array
            if flag_noise_injection:
                eps_ = np.random.randn(*new_xt_est.shape) * np.sqrt(1.0 - alpha_t_bar)  # [N, N]
                eps_ = 0.5 * (eps_ + eps_.transpose((-1, -2)))
                new_xt_est += eps_  # [N, N]
            rep_xt.append(new_xt_est)
            rep_x0.append(this_x0_est)
            rep_gt_score.append(this_gt_score)
        if not flag_repetitive:
            consecutive_xt_ls.append(rep_xt[1])

        rep_xt_log_prob = gmm_log_prob(x_vec=torch.from_numpy(np.stack(rep_xt)).view(-1, num_node**2).to(adjs_gt),
                                       mu_vec=x0_all_perm_vec,
                                       alpha_t_bar=alpha_t_bar).cpu().numpy()  # [D] steps x_t log prob

        # evaluate likelihood of all scaled permuted data
        # note: we only need to evaluate one permutation, because this distribution is equivariant!
        mu_t_log_prob = gmm_log_prob(x_vec=x0_all_perm_vec[0:1] * in_scaling_coef,
                                     mu_vec=x0_all_perm_vec,
                                     alpha_t_bar=alpha_t_bar).item()  # float

        # evaluate likelihood of the collapsed averaged permuted data
        avg_mu_t = x0_all_perm_vec.mean(dim=0, keepdim=True) * in_scaling_coef
        avg_mu_t_log_prob = gmm_log_prob(x_vec=avg_mu_t,
                                         mu_vec=x0_all_perm_vec,
                                         alpha_t_bar=alpha_t_bar).item()  # scalar

        for row in range(n_rows):
            for col in range(n_cols):
                if row == 0:
                    if col == 0:
                        data, v_min, v_max, str_title = input_xt_sample, -3.0, 3.0, 'Noisy adjs x_t (input)'
                    elif col == 1:
                        data, v_min, v_max, str_title = rep_gt_score[0], -3.0, 3.0, 'GT score'
                    elif col == 2:
                        data, v_min, v_max, str_title = rep_xt[1], -3.0, 3.0, 'One-step denoised x_t'
                    elif col == 3:
                        avg_mu_t = avg_mu_t.view(num_node, num_node).cpu().numpy()
                        data, v_min, v_max, str_title = avg_mu_t, -1.0, 1.0, 'GT Avg. permuted mu_t'
                    else:
                        data, v_min, v_max, str_title = None, None, None, None
                elif row == 1:
                    if col == 0:
                        data, v_min, v_max, str_title = mu_t, -1.0, 1.0, 'GT adjs mu_t'
                    elif col == 1:
                        data, v_min, v_max, str_title = rep_xt[1], -1.0, 1.0, 'x_t denoising (1)'
                    elif col == 2:
                        data, v_min, v_max, str_title = rep_xt[5], -1.0, 1.0, 'x_t denoising (5)'
                    elif col == 3:
                        data, v_min, v_max, str_title = rep_xt[10], -1.0, 1.0, 'x_t denoising (10)'
                    elif col == 4:
                        data, v_min, v_max, str_title = rep_xt[20], -1.0, 1.0, 'x_t denoising (20)'
                    else:
                        data, v_min, v_max, str_title = None, None, None, None
                elif row == 2:
                    if col == 0:
                        data, v_min, v_max, str_title = in_data_gt, -1.0, 1.0, 'GT adjs x_0'
                    elif col == 1:
                        data, v_min, v_max, str_title = rep_x0[1], -1.0, 1.0, 'x_0 denoising (1)'
                    elif col == 2:
                        data, v_min, v_max, str_title = rep_x0[5], -1.0, 1.0, 'x_0 denoising (5)'
                    elif col == 3:
                        data, v_min, v_max, str_title = rep_x0[10], -1.0, 1.0, 'x_0 denoising (10)'
                    elif col == 4:
                        data, v_min, v_max, str_title = rep_x0[20], -1.0, 1.0, 'x_0 denoising (20)'
                    else:
                        data, v_min, v_max, str_title = None, None, None, None
                    if data is not None:
                        nx.draw(nx.from_numpy_array(data > 0.0), ax=axes[row + 1, col])

                if col < n_cols - 1:
                    if row < 3:
                        if data is None:
                            axes[row, col].axis('off')
                        else:
                            im = axes[row, col].imshow(data, vmin=v_min, vmax=v_max, cmap='ocean')
                            divider = make_axes_locatable(axes[row, col])
                            cax = divider.append_axes('right', size='3%', pad=0.05)
                            cb = plt.colorbar(im, cax=cax, orientation='vertical')
                            cb_ls.append(cb)
                            axes[row, col].set_title("{:s} \nDiffusion step: {:03d}."
                                                     "\nMin: {:.4f}, Max: {:.4f}.".format(
                                                        str_title, in_cond, data.min(), data.max()))
                else:
                    if row == 1:
                        axes[row, col].plot(rep_xt_log_prob, '-o', label='x_t samples')
                        axes[row, col].axhline(y=mu_t_log_prob, linestyle='-.', color='blue', label='mu_t (non-collapse)')
                        axes[row, col].axhline(y=avg_mu_t_log_prob, linestyle='--', color='red', label='avg_mu_t (collapse)')
                        axes[row, col].set_xlabel("Step of repetitive denoising")
                        axes[row, col].set_ylabel("Log prob at diffusion step {:03d}".format(in_cond))
                        axes[row, col].legend(loc='lower right')
                    else:
                        axes[row, col].axis('off')

        if flag_repetitive:
            prefix = "repetitive_"
        else:
            prefix = "consecutive_"
        if flag_noise_injection:
            prefix += 'with_noise_'
        else:
            prefix += 'no_noise_'
        suffix = "gt_score_sampling_step_{:03d}.png".format(in_cond)
        plot_file_name = prefix + suffix

        plot_path = os.path.join(plot_dir, plot_file_name)
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        for cb in cb_ls:
            cb.remove()
        i_iter += 1
    plt.close(fig)


def sanity_check_diffusion_training(adjs_gt, node_flags, train_obj_gen, sanity_check_save_dir):
    """
    Sanity check for diffusion model.
    """

    """Init"""
    max_steps = train_obj_gen.max_steps
    # num_snapshots = min(max_steps, 20)
    # num_snapshots = 10
    num_snapshots = max_steps
    timesteps = torch.linspace(0, max_steps, num_snapshots).long().clip(max=max_steps - 1)
    aug_adjs_gt = adjs_gt[0:1].repeat(num_snapshots, 1, 1)
    aug_node_flags = node_flags[0:1].repeat(num_snapshots, 1)
    legit_num_nodes = aug_node_flags[0].sum().long().item()
    os.makedirs(sanity_check_save_dir, exist_ok=True)

    """Plot noise injection coefficient curve"""
    scaling_coef = train_obj_gen.const_alpha_t_bar.sqrt().cpu().numpy()  # sqrt of alpha_t_bar
    noise_coef = (1.0 - train_obj_gen.const_alpha_t_bar).sqrt().cpu().numpy()  # sqrt of 1.0 - alpha_t_bar

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(scaling_coef, label='Data coef., alpha_t_bar')
    ax.plot(noise_coef, label='Noise coef., sqrt(1 - alpha_t)')
    ax.legend()
    plot_path = os.path.join(sanity_check_save_dir, "training_coef_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    """Plot noisy data"""
    scaling_coef = torch.index_select(train_obj_gen.const_alpha_t_bar, 0, timesteps.to(adjs_gt).long()).sqrt()
    noise_coef = (1.0 - scaling_coef.square()).sqrt()

    training_noisy_adjs, _ = train_obj_gen.get_network_input(aug_adjs_gt, aug_node_flags, timesteps.to(adjs_gt))
    training_noisy_adjs = training_noisy_adjs[:, :legit_num_nodes, :legit_num_nodes]  # remove inactive nodes

    # plot_diffusion_noisy_data(training_noisy_adjs, adjs_gt[0], timesteps, scaling_coef, noise_coef,
    #                           sanity_check_save_dir)

    """Visualize repetitive denoising sampling using ground-truth score function"""
    # plot_gt_score_sampling(training_noisy_adjs, adjs_gt[0], timesteps, scaling_coef, noise_coef,
    #                        sanity_check_save_dir, flag_noise_injection=True, flag_repetitive=True)
    # plot_gt_score_sampling(training_noisy_adjs, adjs_gt[0], timesteps, scaling_coef, noise_coef,
    #                        sanity_check_save_dir, flag_noise_injection=False, flag_repetitive=True)

    """Visualize consecutive denoising sampling using ground-truth score function"""
    # plot_gt_score_sampling(training_noisy_adjs, adjs_gt[0], timesteps, scaling_coef, noise_coef,
    #                        sanity_check_save_dir, flag_noise_injection=True, flag_repetitive=False)
    #
    # plot_gt_score_sampling(training_noisy_adjs, adjs_gt[0], timesteps, scaling_coef, noise_coef,
    #                        sanity_check_save_dir, flag_noise_injection=False, flag_repetitive=False)

    # pdb.set_trace()

    """Visualize the eigenspace"""
    # legit_num_nodes = aug_node_flags[0].sum().long().item()
    # training_noisy_adjs = training_noisy_adjs[:, :legit_num_nodes, :legit_num_nodes]  # remove inactive nodes
    # training_noisy_adjs = torch.tensor(training_noisy_adjs).to(adjs_gt)  # [B, N, N]
    # scaling_coef = torch.index_select(train_obj_gen.const_alpha_t_bar, 0, timesteps.to(adjs_gt).long()).sqrt()
    # training_noisy_adjs_shrink = training_noisy_adjs / scaling_coef.view(-1, 1, 1)  # [B, N, N]
    # training_noisy_adjs_shrink = torch.clip(training_noisy_adjs_shrink / 2.0 + 0.5, min=0.0, max=1.0)  # [B, N, N]
    #
    # # Plot eigenspace visuals
    # n_rows, n_cols = 4, 3
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 24))
    # for in_data_shrink, in_data_raw, in_cond in zip(training_noisy_adjs_shrink, training_noisy_adjs, timesteps):
    #
    #     laplacian = get_laplacian(in_data_shrink.unsqueeze(0), normalized=True, exponential=False)[0]  # [N, N]
    #     eig_vals_r, eig_vecs_r = torch.linalg.eigh(in_data_raw)     # [N] + [N, N]
    #     eig_vals_s, eig_vecs_s = torch.linalg.eigh(in_data_shrink)  # [N] + [N, N]
    #     eig_vals_l, eig_vecs_l = torch.linalg.eigh(laplacian)       # [N] + [N, N]
    #
    #     # convert to numpy array
    #     in_data_raw, in_data_shrink, laplacian = _to_numpy_array((in_data_raw, in_data_shrink, laplacian))
    #
    #     [axes[i, j].clear() for i in range(n_rows) for j in range(n_cols)]
    #
    #     for row in range(n_rows):
    #         for col in range(n_cols):
    #             # plot original matrix
    #             if row == 0:
    #                 if col == 0:
    #                     data, v_min, v_max = in_data_raw, -3.0, 3.0
    #                 elif col == 1:
    #                     data, v_min, v_max = in_data_shrink, 0.0, 1.0
    #                 elif col == 2:
    #                     data, v_min, v_max = laplacian, 0.0, 1.0
    #                 else:
    #                     raise NotImplementedError
    #                 im = axes[row, col].imshow(data, vmin=v_min, vmax=v_max, cmap='ocean')
    #                 divider = make_axes_locatable(axes[row, col])
    #                 cax = divider.append_axes('right', size='3%', pad=0.05)
    #                 plt.colorbar(im, cax=cax, orientation='vertical')
    #                 axes[row, col].set_title("Diffusion step: {:03d}."
    #                                          "\nMin: {:.4f}, Max: {:.4f}.".format(in_cond,
    #                                                                               data.min(), data.max()))
    #             # plot eigenvalue histogram
    #             elif row == 1:
    #                 if col == 0:
    #                     data = eig_vals_r
    #                 elif col == 1:
    #                     data = eig_vals_s
    #                 elif col == 2:
    #                     data = eig_vals_l
    #                 else:
    #                     raise NotImplementedError
    #                 data = _to_numpy_array(data)
    #                 axes[row, col].hist(data, bins=50, density=True, stacked=True)
    #                 axes[row, col].set_xlabel('Value')
    #                 if col == 0:
    #                     axes[row, col].set_ylabel('Density')
    #                 axes[row, col].set_title('Histogram of eigenvalues')
    #
    #             # plot eigenvalue scatter
    #             elif row == 2:
    #                 if col == 0:
    #                     data = eig_vals_r
    #                 elif col == 1:
    #                     data = eig_vals_s
    #                 elif col == 2:
    #                     data = eig_vals_l
    #                 else:
    #                     raise NotImplementedError
    #                 data = np.sort(_to_numpy_array(data))
    #                 axes[row, col].plot(data, 'D-', linewidth=1.0, markersize=1.0)
    #                 axes[row, col].set_xlabel('Rank')
    #                 if col == 0:
    #                     axes[row, col].set_ylabel('Value')
    #                 axes[row, col].set_title('Sorted eigenvalues')
    #
    #             # plot eigen vector matrix
    #             elif row == 3:
    #                 if col == 0:
    #                     data = eig_vecs_r
    #                 elif col == 1:
    #                     data = eig_vecs_s
    #                 elif col == 2:
    #                     data = eig_vecs_l
    #                 else:
    #                     raise NotImplementedError
    #                 v_min, v_max = -0.1, 0.1
    #                 data = _to_numpy_array(data)  # [N, N]
    #                 im = axes[row, col].imshow(data, vmin=v_min, vmax=v_max, cmap='ocean')
    #                 divider = make_axes_locatable(axes[row, col])
    #                 cax = divider.append_axes('right', size='3%', pad=0.05)
    #                 plt.colorbar(im, cax=cax, orientation='vertical')
    #                 axes[row, col].set_title("Eigenvector map "
    #                                          "\nMin: {:.4f}, Max: {:.4f}.".format(
    #                                                 data.min(), data.max()))
    #                 axes[row, col].set_xlabel('Order of eigenfunction')
    #                 if col == 0:
    #                     axes[row, col].set_ylabel("Order of node")
    #
    #     os.makedirs(sanity_check_save_dir, exist_ok=True)
    #     plot_path = os.path.join(sanity_check_save_dir, "eigenmap_training_sample_sigma_step_{:03d}.png".format(in_cond))
    #     plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    # plt.close(fig)

    """Visualize KL divergence of the q(x_T|x_0) || N(0, I)"""
    max_beta_ls = np.linspace(0.01, 0.05, 50)
    max_step_ls = np.linspace(400, 1600, 50).astype(int)
    kl_loss = np.zeros([len(max_step_ls), len(max_beta_ls)])
    x0 = adjs_gt.cpu().numpy()  # [B, N, N]
    aug_node_flags = node_flags.cpu().numpy()  # [B, N]

    # Compute KL divergence
    for i, max_step in enumerate(max_step_ls):
        for j, max_beta in enumerate(max_beta_ls):
            alpha_bar = get_alpha_t_bar(timesteps=[max_step - 1],
                                        max_steps=max_step,
                                        beta_min=train_obj_gen.beta_min, beta_max=max_beta,
                                        schedule=train_obj_gen.schedule).item()  # [T] steps

            mean = np.sqrt(alpha_bar) * x0.copy()  # [B, N, N]
            std = np.sqrt(1.0 - alpha_bar) * np.ones_like(mean)  # [B, N, N]

            mean[~aug_node_flags.astype(bool)] = 0.0
            std[~aug_node_flags.astype(bool)] = 1.0
            kl = _compute_kl_std_normal(mean=mean, std=std)  # [B]
            kl_loss[i, j] = kl.mean()

    # Plot KL divergence surface
    def log_tick_formatter(val, pos=None):
        return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    x_mesh, y_mesh = np.meshgrid(max_step_ls, max_beta_ls)
    surf = ax.plot_surface(x_mesh, y_mesh,
                           np.log10(kl_loss.clip(min=1e-9)).clip(min=1e-6),
                           cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_xlabel("Max. #steps (T)")
    ax.set_ylabel("Max. beta_T")
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.view_init(None, 45)

    os.makedirs(sanity_check_save_dir, exist_ok=True)
    plot_path = os.path.join(sanity_check_save_dir, "KL_divergence_surface.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def sanity_check_score_training(adjs_gt, node_flags, train_obj_gen, sanity_check_save_dir):
    """
    Sanity check for score-based model.
    """

    """Visualize noise-perturbed data"""
    # Generate noisy data
    sigma_num_slices = train_obj_gen.sigma_num_slices
    num_snapshots = min(sigma_num_slices, 20)

    sigma_idxs = np.linspace(0, sigma_num_slices, num_snapshots).astype(int).clip(max=sigma_num_slices - 1)
    sigma_values = torch.index_select(train_obj_gen.const_sigma_t.cpu(), 0, torch.tensor(sigma_idxs).long()).numpy()

    aug_adjs_gt = torch.stack([adjs_gt[0].cpu()] * len(sigma_idxs))
    aug_node_flags = torch.stack([node_flags[0].cpu()] * len(sigma_idxs))
    training_noisy_adjs, _ = train_obj_gen.get_network_input(aug_adjs_gt, aug_node_flags,
                                                             torch.tensor(sigma_idxs).to(adjs_gt).long())
    training_noisy_adjs = training_noisy_adjs.cpu().numpy()  # [X, N, N]

    # Plot noisy data
    fig = plt.figure()
    for in_data, in_cond, true_sigma in zip(training_noisy_adjs, sigma_idxs, sigma_values):
        plt.clf()
        plt.imshow(in_data, vmin=-5.0, vmax=5.0, cmap='ocean')
        plt.colorbar()
        plt.title("Score noise perturbation sigma idx: {:03d} ({:.04f}). \n Min: {:.4f}, Max: {:.4f}.".format(
            in_cond, true_sigma, in_data.min(), in_data.max()))
        os.makedirs(sanity_check_save_dir, exist_ok=True)
        plot_path = os.path.join(sanity_check_save_dir, "training_sample_sigma_no{:03d}_{:.04f}.png".format(
            in_cond, true_sigma))
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_edm_coefs(num_steps, plot_dir):
    """Plot network target at different time-steps (sampling discretization scheme)"""
    step_indices = torch.arange(num_steps, dtype=torch.float)
    edm_params = get_edm_params()
    rho = edm_params.rho
    sampling_sigma_steps = (edm_params.sigma_max_sampling ** (1 / rho) + step_indices / (num_steps - 1) * (
            edm_params.sigma_min_sampling ** (1 / rho) - edm_params.sigma_max_sampling ** (1 / rho))) ** rho   # [N]

    c_skip, c_out, c_in, c_noise = get_preconditioning_params('edm', sampling_sigma_steps, None, None, edm_params)  # [N]

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(c_in, label='c_in (scaling)')
    ax.plot(c_in * sampling_sigma_steps, label='Gaussian noise std, c_in*sigma')
    ax.plot((1.0 - c_skip) / c_out, label='Target-clean data coef, (1-c_skip)/c_out')
    ax.plot(-c_skip / c_out * sampling_sigma_steps, label='Target-Gaussian noise std, -c_skip/c_out*sigma')
    ax.legend()
    ax.set_xlabel("Sampling step (starts from 0)")
    plot_path = os.path.join(plot_dir, "sampling_scheme_main_coef.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(sampling_sigma_steps, label='Sigma')
    ax.legend()
    ax.set_xlabel("Sampling step (starts from 0)")
    plot_path = os.path.join(plot_dir, "sampling_scheme_sigma_coef.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    """Plot network target at different time-steps (training sigma distribution scheme)"""
    training_sigma_steps = np.linspace(0.002, 20, 1000)
    p = norm.pdf(x=np.log(training_sigma_steps))
    training_sigma_steps = torch.as_tensor(training_sigma_steps)
    c_skip, c_out, c_in, c_noise = get_preconditioning_params('edm', training_sigma_steps, None, None, edm_params)  # [N]

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(training_sigma_steps, c_in, label='c_in (scaling)')
    ax.plot(training_sigma_steps, c_in * training_sigma_steps, label='Gaussian noise std, c_in*sigma')
    ax.plot(training_sigma_steps, (1.0 - c_skip) / c_out, label='Target-clean data coef, (1-c_skip)/c_out')
    ax.plot(training_sigma_steps, -c_skip / c_out * training_sigma_steps, label='Target-Gaussian noise std, -c_skip/c_out*sigma')
    ax.legend()
    ax.set_xlabel("Sigma")
    plot_path = os.path.join(plot_dir, "training_scheme_sigma_vs_other_coef.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(c_in, p, label='c_in (scaling)')
    ax.plot(c_in * training_sigma_steps, p, label='Gaussian noise std, c_in*sigma')
    ax.plot((1.0 - c_skip) / c_out, p, label='Target-clean data coef, (1-c_skip)/c_out')
    ax.plot(-c_skip / c_out * training_sigma_steps, p, label='Target-Gaussian noise std, -c_skip/c_out*sigma')
    ax.legend()
    ax.set_xlabel("Value")
    ax.set_ylabel("PDF")
    plot_path = os.path.join(plot_dir, "training_scheme_main_coef.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(training_sigma_steps, p, 'r-', alpha=0.6, label='sigma pdf')
    ax.legend()
    ax.set_xlabel("Sigma")
    plot_path = os.path.join(plot_dir, "training_scheme_sigma_coef.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_edm_noisy_data(precond_in, precond_target, sigmas, probs, c_skip, c_out, c_in, plot_dir):
    """
    Plot the noisy data and some of its augmentations.
    @param precond_in:                  [B, N, N],  x = y+n
    @param precond_target:              [N, N],     y
    @param sigmas:                      [B]
    @param probs:                       [B]
    @param c_skip:                      [B]
    @param c_out:                       [B]
    @param c_in:                        [B]
    @param plot_dir:                    str
    """

    """Init"""
    num_node = precond_target.size(-1)
    os.makedirs(plot_dir, exist_ok=True)

    model_in = precond_in * c_in[:, None, None]  # [B, N, N]
    model_target = 1.0 / c_out[:, None, None] * (precond_target.unsqueeze(0) - c_skip[:, None, None] * precond_in)  # [B, N, N]

    """Plot loop"""
    n_rows, n_cols = 1, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 10))
    i_iter = 0
    in_data_gt = precond_target.cpu().numpy()  # [N, N]
    for in_precond_in, in_model_in, in_model_target, in_sigma, in_prob in zip(
            precond_in, model_in, model_target, sigmas, probs):

        # [axes[i, j].clear() for i in range(n_rows) for j in range(n_cols)]
        [axes[j].clear() for j in range(n_cols)]
        cb_ls = []
        in_precond_in, in_model_in, in_model_target = _to_numpy_array(
            (in_precond_in, in_model_in, in_model_target))

        for row in range(n_rows):
            for col in range(n_cols):
                if row == 0:
                    # plot original matrix
                    if col == 0:
                        data, v_min, v_max, str_title = in_precond_in, -3.0, 3.0, 'y+n, corrupted input'
                    elif col == 1:
                        data, v_min, v_max, str_title = in_data_gt, -3.0, 3.0, 'y, clean data'
                    elif col == 2:
                        data, v_min, v_max, str_title = in_model_in, -3.0, 3.0, 'c_in*(y+n), effective input'
                    elif col == 3:
                        data, v_min, v_max, str_title = in_model_target, -3.0, 3.0, 'Effective target'
                    else:
                        raise NotImplementedError

                    im = axes[col].imshow(data, vmin=v_min, vmax=v_max, cmap='ocean')
                    divider = make_axes_locatable(axes[col])
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    cb = plt.colorbar(im, cax=cax, orientation='vertical')
                    cb_ls.append(cb)
                    axes[col].set_title("{:s} EDM training, sigma={:.3f}, p={:.4f}"
                                        "\nMin: {:.4f}, Max: {:.4f}.".format(str_title, in_sigma, in_prob,
                                                                             data.min(), data.max()))

        plot_path = os.path.join(plot_dir, "training_sample_sigma_{:.3f}_p_{:.4f}.png".format(in_sigma, in_prob))
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        for cb in cb_ls:
            cb.remove()
        i_iter += 1
    plt.close(fig)


def sanity_check_edm_training(adjs_gt, node_flags, train_obj_gen, sanity_check_save_dir):
    """
    Sanity check for diffusion model.
    """

    """Init"""
    max_steps = 79  # default params
    num_snapshots = max_steps
    # num_snapshots = min(max_steps, 20)
    # num_snapshots = 10
    aug_adjs_gt = adjs_gt[0:1].repeat(num_snapshots, 1, 1)
    aug_node_flags = node_flags[0:1].repeat(num_snapshots, 1)
    legit_num_nodes = aug_node_flags[0].sum().long().item()
    os.makedirs(sanity_check_save_dir, exist_ok=True)

    """Plot network coefficients"""
    plot_edm_coefs(max_steps, sanity_check_save_dir)

    """Plot edm input-output"""
    # this may be a bit time-consuming
    precond_in, _, precond_target, (c_skip, c_out, c_in, c_noise, sigmas, weights) = train_obj_gen.get_input_output(
        aug_adjs_gt, aug_node_flags)
    precond_in = precond_in[:, :legit_num_nodes, :legit_num_nodes]  # remove inactive nodes
    precond_target = precond_target[0, :legit_num_nodes, :legit_num_nodes]  # precond target is clean data, all same
    probs = norm.pdf(x=np.log(sigmas.cpu().numpy()))
    # plot_edm_noisy_data(precond_in, precond_target, sigmas, probs, c_skip, c_out, c_in, sanity_check_save_dir)


def sanity_check_training_objectives(adjs_gt, node_flags, train_obj_gen, sanity_check_save_dir, other_params=None):
    """
    Visualize the training data for sanity check.
    """

    objective = train_obj_gen.objective
    if objective == 'diffusion':
        sanity_check_diffusion_training(adjs_gt, node_flags, train_obj_gen, sanity_check_save_dir)
    elif objective == 'score':
        sanity_check_score_training(adjs_gt, node_flags, train_obj_gen, sanity_check_save_dir)
    elif objective == 'edm':
        sanity_check_edm_training(adjs_gt, node_flags, train_obj_gen, sanity_check_save_dir)
    else:
        raise NotImplementedError
