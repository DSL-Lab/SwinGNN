import pdb

import torch
import torch.nn as nn
import numpy as np
import logging

from torch import nn as nn

from utils.graph_utils import mask_adjs, check_adjs_symmetry, get_sym_normal_noise, mask_nodes
from runner.objectives.edm import get_vp_sigma_from_t, get_vp_sigma_deriv_t, get_vp_t_from_sigma, \
    get_ve_sigma_from_t, get_ve_sigma_deriv_t, get_ve_t_from_sigma, \
    get_edm_sigma_from_t, get_edm_sigma_deriv_t, get_edm_t_from_sigma, get_vp_params, get_ve_params, get_edm_params
from . import GeneralSampler


class EDMSampler(GeneralSampler):
    """
    MCMC sampler for EDM.
    """

    def __init__(self,
                 *,
                 sigma_min=None, sigma_max=None,
                 solver='heun', discretization='edm', schedule='linear', scaling='none',
                 C_1=0.001, C_2=0.008, M=1000, alpha=1,
                 # num_steps=18, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
                 # EDM ImageNet parameters
                 num_steps=256, S_churn=40, S_min=0.05, S_max=50, S_noise=1.003,
                 clip_samples, clip_samples_min, clip_samples_max, clip_samples_scope,
                 self_condition, dev, objective='edm'):
        super().__init__(clip_samples, clip_samples_min, clip_samples_max, objective, dev)

        assert clip_samples_scope == 'x_0'

        assert solver in ['euler', 'heun']
        assert discretization in ['vp', 've', 'iddpm', 'edm']
        assert schedule in ['vp', 've', 'linear']
        assert scaling in ['vp', 'none']

        self.solver = solver
        self.discretization = discretization
        self.schedule = schedule
        self.scaling = scaling

        self.num_steps = num_steps
        self.alpha = alpha
        self.dev = dev

        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

        self.vp_params = get_vp_params()
        self.ve_params = get_ve_params()
        self.edm_params = get_edm_params()

        self.self_condition = self_condition

        # Select default noise level range based on the specified time step discretization.
        if sigma_min is None:
            sigma_min = {'vp': self.vp_params.sigma_min_sampling,
                         've': self.ve_params.sigma_min_sampling,
                         'iddpm': 0.002,
                         'edm': self.edm_params.sigma_min_sampling}[discretization]
        if sigma_max is None:
            sigma_max = {'vp': self.vp_params.sigma_max_sampling,
                         've': self.ve_params.sigma_max_sampling,
                         'iddpm': 81,
                         'edm': self.edm_params.sigma_max_sampling}[discretization]

        # Define time steps in terms of noise level.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=dev)
        if discretization == 'vp':
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (self.vp_params.epsilon_s - 1)
            sigma_steps = get_vp_sigma_from_t(t=orig_t_steps)
        elif discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = get_ve_sigma_from_t(orig_t_steps)
        elif discretization == 'iddpm':
            u = torch.zeros(M + 1, dtype=torch.float64, device=dev)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=dev):  # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
        else:
            assert discretization == 'edm'
            rho = self.edm_params.rho
            sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # Define noise level schedule.
        if schedule == 'vp':
            sigma = get_vp_sigma_from_t
            sigma_deriv = get_vp_sigma_deriv_t
            sigma_inv = get_vp_t_from_sigma
        elif schedule == 've':
            sigma = get_ve_sigma_from_t
            sigma_deriv = get_ve_sigma_deriv_t
            sigma_inv = get_ve_t_from_sigma
        else:
            assert schedule == 'linear'
            sigma = get_edm_sigma_from_t
            sigma_deriv = get_edm_sigma_deriv_t
            sigma_inv = get_edm_t_from_sigma

        # Define scaling schedule.
        if scaling == 'vp':
            s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
            s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
        else:
            assert scaling == 'none'
            s = lambda t: 1
            s_deriv = lambda t: 0

        self.sigma = sigma
        self.sigma_inv = sigma_inv
        self.sigma_deriv = sigma_deriv
        self.s = s
        self.s_deriv = s_deriv
        self.sigma_steps = sigma_steps

    def sample(self, model, node_flags, init_adjs=None, sanity_check_gt_adjs=None,
               flag_interim_adjs=False, max_num_interim_adjs=None, flag_use_double=False):
        """
        Generate samples for DDPM.
        @param model: preconditioned network whose input arguments are [x, adjs, node_flags, conditions]
        @param node_flags: [B, N]
        @param init_adjs: None for drawing initial adjs from pure noise, otherwise use the given adjs
        @param sanity_check_gt_adjs: None for model inference, otherwise compute ground-truth target from gt adjs.
        @param flag_interim_adjs: bool, return the interim adjs
        @param max_num_interim_adjs: None for unlimited num of adjs, otherwise take evenly-separated snapshots.
        @param flag_use_double: bool, to use float64
        """

        # Compute final time steps based on the corresponding noise levels.
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            func_round_sigma = model.module.round_sigma
        else:
            func_round_sigma = model.round_sigma
        t_steps = self.sigma_inv(func_round_sigma(self.sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
        if flag_use_double:
            t_steps = t_steps.to(torch.float64)
        else:
            t_steps = t_steps.to(torch.float32)

        if init_adjs is None:
            init_adjs = self.gen_init_sample(node_flags, folded_norm=True)  # [B, N, N], already masked
        adjs = init_adjs
        adjs_ls = [init_adjs.cpu()]
        if max_num_interim_adjs is None:
            timesteps_snapshot = np.arange(self.num_steps)
        else:
            timesteps_snapshot = np.linspace(0, self.num_steps, max_num_interim_adjs).astype(
                                    int).clip(max=self.num_steps-1)

        # Main sampling loop, the iteration is over the time-signal.
        # we use notation x to represent the adjacency matrix data
        t_next = t_steps[0]
        if flag_use_double:
            x_next = init_adjs.to(torch.float64) * (self.sigma(t_next) * self.s(t_next))
        else:
            x_next = init_adjs * (self.sigma(t_next) * self.s(t_next))
        x_self_cond = None
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= self.sigma(t_cur) <= self.S_max else 0
            t_hat = self.sigma_inv(func_round_sigma(self.sigma(t_cur) + gamma * self.sigma(t_cur)))
            x_hat = self.s(t_hat) / self.s(t_cur) * x_cur + (self.sigma(t_hat) ** 2 - self.sigma(t_cur) ** 2).clip(min=0).sqrt() * self.s(
                t_hat) * self.S_noise * get_sym_normal_noise(x_cur)
            x_hat = mask_adjs(x_hat, node_flags)  # [B, N, N]

            # Euler step.
            h = t_next - t_hat
            sigma_tensors = self.sigma(t_hat).view(-1).expand(node_flags.size(0))
            if sanity_check_gt_adjs is None:
                with torch.no_grad():
                    denoised = model(x_hat / self.s(t_hat), node_flags, sigma_tensors, x_self_cond)
            else:
                denoised = sanity_check_gt_adjs
            if flag_use_double:
                denoised = denoised.to(torch.float64)
            denoised = mask_adjs(denoised, node_flags)  # [B, N, N]
            d_cur = (self.sigma_deriv(t_hat) / self.sigma(t_hat) + self.s_deriv(t_hat) / self.s(t_hat)) * x_hat - self.sigma_deriv(t_hat) * self.s(
                t_hat) / self.sigma(t_hat) * denoised
            d_cur = mask_adjs(d_cur, node_flags)  # [B, N, N]
            x_prime = x_hat + self.alpha * h * d_cur
            t_prime = t_hat + self.alpha * h

            # Apply 2nd order correction.
            if self.solver == 'euler' or i == self.num_steps - 1:
                x_next = x_hat + h * d_cur
            else:
                assert self.solver == 'heun'
                # denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
                sigma_tensors = self.sigma(t_hat).view(-1).expand(node_flags.size(0))
                if sanity_check_gt_adjs is None:
                    x_self_cond = denoised if self.self_condition else None
                    with torch.no_grad():
                        denoised = model(x_hat / self.s(t_hat), node_flags, sigma_tensors, x_self_cond)
                else:
                    denoised = sanity_check_gt_adjs
                if flag_use_double:
                    denoised = denoised.to(torch.float64)
                d_cur = mask_adjs(d_cur, node_flags)  # [B, N, N]
                d_prime = (self.sigma_deriv(t_prime) / self.sigma(t_prime) + self.s_deriv(t_prime) / self.s(
                    t_prime)) * x_prime - self.sigma_deriv(t_prime) * self.s(t_prime) / self.sigma(t_prime) * denoised
                x_next = x_hat + h * ((1 - 1 / (2 * self.alpha)) * d_cur + 1 / (2 * self.alpha) * d_prime)

            x_next = mask_adjs(x_next, node_flags)
            x_self_cond = denoised if self.self_condition else None
            check_adjs_symmetry(x_next)
            adjs = x_next
            if flag_interim_adjs:
                if i in timesteps_snapshot:
                    adjs_ls.append(adjs.cpu())
            logging.debug("EDM MCMC: step {:5d} | avg. #edges@0.0 of x_t: {:08d} |"
                          .format(i,
                                  int(EDMSampler.get_num_edges(adjs, node_flags, 0.0).mean().item())
                                  )
                          )

        adjs = adjs.cpu()
        if flag_interim_adjs:
            return adjs, torch.stack(adjs_ls)
        else:
            return adjs


class NodeAdjEDMSampler(EDMSampler):
    """
    MCMC sampler for EDM framework that generates node and adjacency matrix samples simultaneously.
    """

    def __init__(self,
                 *,
                 sigma_min=None, sigma_max=None,
                 solver='heun', discretization='edm', schedule='linear', scaling='none',
                 C_1=0.001, C_2=0.008, M=1000, alpha=1,
                 # num_steps=18, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
                 # EDM ImageNet parameters
                 num_steps=256, S_churn=40, S_min=0.05, S_max=50, S_noise=1.003,
                 clip_samples, clip_samples_min, clip_samples_max, clip_samples_scope,
                 self_condition, dev, objective='edm', symmetric_noise=True):

        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max,
                         solver=solver, discretization=discretization, schedule=schedule, scaling=scaling,
                         C_1=C_1, C_2=C_2, M=M, alpha=alpha,
                         num_steps=num_steps, S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise,
                         clip_samples=clip_samples, clip_samples_min=clip_samples_min,
                         clip_samples_max=clip_samples_max, clip_samples_scope=clip_samples_scope,
                         self_condition=self_condition, dev=dev, objective=objective)

        self.symmetric_noise = symmetric_noise

    def gen_init_sample(self, node_flags, folded_norm=False,
                        flag_node_multi_channel=False, flag_adj_multi_channel=False,
                        num_node_chan=150, num_edge_chan=51, ):
        """
        Generate initial samples. [overridden]
        @param node_flags: [B, N] or [B, N, N]
        @param folded_norm: bool
        @param flag_node_multi_channel: bool, to use multiple channels for node attributes
        @param flag_adj_multi_channel: bool, to use multiple channels for edge attributes
        @param num_node_chan: int, #node types
        @param num_edge_chan: int, #edge types
        @return init_adjs: [B, *, N, N] with proper masking.
        @return init_nodes: [B, N, *] with proper masking.
        """
        # initialize noisy adjacency matrix
        batch_size, max_node_num = node_flags.shape[:2]
        if self.symmetric_noise:
            init_adjs = torch.randn((batch_size, num_edge_chan, max_node_num, max_node_num)
                                    ).triu(diagonal=1).to(self.dev)  # [B, C, N, N]
            init_adjs = init_adjs.abs() if folded_norm else init_adjs
            init_adjs = init_adjs + init_adjs.transpose(-1, -2)
        else:
            init_adjs = torch.randn((batch_size, num_edge_chan, max_node_num, max_node_num)).to(self.dev)
        init_adjs = mask_adjs(init_adjs, node_flags)
        if num_edge_chan == 1:
            init_adjs = init_adjs.squeeze(1)  # [B, N, N] <- [B, C=1, N, N]

        # initialize noisy node features
        init_nodes = torch.randn((batch_size, max_node_num, num_node_chan)).to(self.dev)
        init_nodes = mask_nodes(init_nodes, node_flags)
        if num_node_chan == 1:
            init_nodes = init_nodes.squeeze(-1)  # [B, N] <- [B, N, F=1]
        return init_adjs, init_nodes

    def sample(self, model, node_flags, init_adjs=None, init_nodes=None,
               sanity_check_gt_adjs=None, sanity_check_gt_nodes=None,
               flag_interim_adjs=False, max_num_interim_adjs=None, flag_use_double=False,
               flag_node_multi_channel=False, flag_adj_multi_channel=False,
               num_node_chan=150, num_edge_chan=51, ):
        """
        Generate samples for DDPM.
        @param model: preconditioned network whose input arguments are [nodes, adjs, node_flags, conditions]
        @param node_flags: [B, N]
        @param init_adjs: None for drawing initial adjs from pure noise, otherwise use the given adjs
        @param init_nodes: None for drawing initial node feat from pure noise, otherwise use the given nodes
        @param sanity_check_gt_adjs: None for model inference, otherwise compute ground-truth target from gt adjs.
        @param sanity_check_gt_nodes: None for model inference, otherwise compute ground-truth target from gt nodes.
        @param flag_interim_adjs: bool, return the interim adjs
        @param max_num_interim_adjs: None for unlimited num of adjs, otherwise take evenly-separated snapshots.
        @param flag_use_double: bool, to use float64
        @param flag_node_multi_channel: bool, to use multiple channels for node attributes
        @param flag_adj_multi_channel: bool, to use multiple channels for edge attributes
        @param num_node_chan: int, #node types
        @param num_edge_chan: int, #edge types
        """

        # Compute final time steps based on the corresponding noise levels.
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            func_round_sigma = model.module.round_sigma
        else:
            func_round_sigma = model.round_sigma
        t_steps = self.sigma_inv(func_round_sigma(self.sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
        if flag_use_double:
            t_steps = t_steps.to(torch.float64)
        else:
            t_steps = t_steps.to(torch.float32)

        if init_adjs is None or init_nodes is None:
            init_adjs, init_nodes = self.gen_init_sample(node_flags, folded_norm=False,
                                                         flag_node_multi_channel=flag_node_multi_channel,
                                                         flag_adj_multi_channel=flag_adj_multi_channel,
                                                         num_node_chan=num_node_chan, num_edge_chan=num_edge_chan)
        adjs = init_adjs
        nodes = init_nodes
        adjs_ls = [init_adjs.cpu()]
        nodes_ls = [init_nodes.cpu()]
        if max_num_interim_adjs is None:
            timesteps_snapshot = np.arange(self.num_steps)
        else:
            timesteps_snapshot = np.linspace(0, self.num_steps, max_num_interim_adjs).astype(
                                    int).clip(max=self.num_steps-1)

        # Main sampling loop, the iteration is over the time-signal.
        t_next = t_steps[0]
        if flag_use_double:
            adjs_next = init_adjs.to(torch.float64) * (self.sigma(t_next) * self.s(t_next))
            nodes_next = init_nodes.to(torch.float64) * (self.sigma(t_next) * self.s(t_next))
        else:
            adjs_next = init_adjs * (self.sigma(t_next) * self.s(t_next))
            nodes_next = init_nodes * (self.sigma(t_next) * self.s(t_next))
        nodes_self_cond = None
        adjs_self_cond = None
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            adjs_cur = adjs_next
            nodes_cur = nodes_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= self.sigma(t_cur) <= self.S_max else 0
            t_hat = self.sigma_inv(func_round_sigma(self.sigma(t_cur) + gamma * self.sigma(t_cur)))
            if self.symmetric_noise:
                adjs_hat = self.s(t_hat) / self.s(t_cur) * adjs_cur + (self.sigma(t_hat) ** 2 - self.sigma(t_cur) ** 2).clip(min=0).sqrt() * self.s(
                    t_hat) * self.S_noise * get_sym_normal_noise(adjs_cur)
            else:
                adjs_hat = self.s(t_hat) / self.s(t_cur) * adjs_cur + (self.sigma(t_hat) ** 2 - self.sigma(t_cur) ** 2).clip(min=0).sqrt() * self.s(
                    t_hat) * self.S_noise * torch.randn_like(adjs_cur)
            nodes_hat = self.s(t_hat) / self.s(t_cur) * nodes_cur + (self.sigma(t_hat) ** 2 - self.sigma(t_cur) ** 2).clip(min=0).sqrt() * self.s(
                t_hat) * self.S_noise * torch.randn_like(nodes_cur)
            adjs_hat = mask_adjs(adjs_hat, node_flags)
            nodes_hat = mask_nodes(nodes_hat, node_flags)

            # Euler step.
            h = t_next - t_hat
            # denoised = net(nodes_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
            sigma_tensors = self.sigma(t_hat).view(-1).expand(node_flags.size(0))
            if sanity_check_gt_adjs is None:
                with torch.no_grad():
                    denoised_adjs, denoised_nodes = model(adjs_hat / self.s(t_hat), nodes_hat / self.s(t_hat), node_flags, sigma_tensors, adjs_self_cond, nodes_self_cond)
            else:
                denoised_adjs = sanity_check_gt_adjs
                denoised_nodes = sanity_check_gt_nodes
            if flag_use_double:
                denoised_adjs = denoised_adjs.to(torch.float64)
                denoised_nodes = denoised_nodes.to(torch.float64)
            denoised_adjs = mask_adjs(denoised_adjs, node_flags)
            denoised_nodes = mask_nodes(denoised_nodes, node_flags)

            d_cur_adjs = (self.sigma_deriv(t_hat) / self.sigma(t_hat) + self.s_deriv(t_hat) / self.s(t_hat)) * adjs_hat - self.sigma_deriv(t_hat) * self.s(t_hat) / self.sigma(t_hat) * denoised_adjs
            d_cur_nodes = (self.sigma_deriv(t_hat) / self.sigma(t_hat) + self.s_deriv(t_hat) / self.s(t_hat)) * nodes_hat - self.sigma_deriv(t_hat) * self.s(t_hat) / self.sigma(t_hat) * denoised_nodes
            d_cur_adjs = mask_adjs(d_cur_adjs, node_flags)
            d_cur_nodes = mask_nodes(d_cur_nodes, node_flags)

            nodes_prime = nodes_hat + self.alpha * h * d_cur_nodes
            adjs_prime = adjs_hat + self.alpha * h * d_cur_adjs
            t_prime = t_hat + self.alpha * h

            # Apply 2nd order correction.
            if self.solver == 'euler' or i == self.num_steps - 1:
                nodes_next = nodes_hat + h * d_cur_nodes
                adjs_next = adjs_hat + h * d_cur_adjs
            else:
                assert self.solver == 'heun'
                # denoised = net(nodes_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
                sigma_tensors = self.sigma(t_hat).view(-1).expand(node_flags.size(0))
                if sanity_check_gt_adjs is None:
                    nodes_self_cond = denoised_nodes if self.self_condition else None
                    adjs_self_cond = denoised_adjs if self.self_condition else None
                    with torch.no_grad():
                        denoised_adjs, denoised_nodes = model(adjs_hat / self.s(t_hat), nodes_hat / self.s(t_hat), node_flags, sigma_tensors, adjs_self_cond, nodes_self_cond)
                else:
                    denoised_adjs = sanity_check_gt_adjs
                    denoised_nodes = sanity_check_gt_nodes
                if flag_use_double:
                    denoised_adjs = denoised_adjs.to(torch.float64)
                    denoised_nodes = denoised_nodes.to(torch.float64)
                denoised_adjs = mask_adjs(denoised_adjs, node_flags)
                denoised_nodes = mask_nodes(denoised_nodes, node_flags)
                d_prime_nodes = (self.sigma_deriv(t_prime) / self.sigma(t_prime) + self.s_deriv(t_prime) / self.s(
                    t_prime)) * nodes_prime - self.sigma_deriv(t_prime) * self.s(t_prime) / self.sigma(t_prime) * denoised_nodes
                d_prime_adjs = (self.sigma_deriv(t_prime) / self.sigma(t_prime) + self.s_deriv(t_prime) / self.s(
                    t_prime)) * adjs_prime - self.sigma_deriv(t_prime) * self.s(t_prime) / self.sigma(t_prime) * denoised_adjs
                nodes_next = nodes_hat + h * ((1 - 1 / (2 * self.alpha)) * d_cur_nodes + 1 / (2 * self.alpha) * d_prime_nodes)
                adjs_next = adjs_hat + h * ((1 - 1 / (2 * self.alpha)) * d_cur_adjs + 1 / (2 * self.alpha) * d_prime_adjs)

            adjs_next = mask_adjs(adjs_next, node_flags)
            nodes_next = mask_nodes(nodes_next, node_flags)
            adjs_self_cond = denoised_adjs if self.self_condition else None
            nodes_self_cond = denoised_nodes if self.self_condition else None
            if self.symmetric_noise:
                check_adjs_symmetry(adjs_next)
            adjs = adjs_next
            nodes = nodes_next
            if flag_interim_adjs:
                if i in timesteps_snapshot:
                    adjs_ls.append(adjs.cpu())
                    nodes_ls.append(nodes.cpu())
            logging.debug("EDM-NodeAdj MCMC: step {:5d} | avg. #edges@0.0 of x_t: {:08d} |"
                          .format(i, int(NodeAdjEDMSampler.get_num_edges(adjs, node_flags, 0.0).mean().item())))

        logging.info("Done with EDM-NodeAdj MCMC.")
        adjs = adjs.cpu()
        nodes = nodes.cpu()
        if flag_interim_adjs:
            if flag_adj_multi_channel:
                return adjs, nodes, [None], torch.stack(nodes_ls)
            else:
                return adjs, nodes, torch.stack(adjs_ls), torch.stack(nodes_ls)
        else:
            return adjs, nodes
