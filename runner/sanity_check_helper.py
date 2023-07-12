import os

import numpy as np
from sympy.utilities.iterables import multiset_permutations
from scipy.stats import norm

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
