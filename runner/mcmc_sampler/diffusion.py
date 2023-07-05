import torch
import numpy as np
import logging

from runner.sanity_check_helper import compute_score, get_all_permutation
from utils.graph_utils import mask_adjs, add_sym_normal_noise
from runner.objectives.diffusion import get_sigma_t, get_alpha_t, get_alpha_t_bar, get_beta_t
from . import GeneralSampler


class DiffusionSampler(GeneralSampler):
    """
    MCMC sampler for DDPM.
    """

    def __init__(self,
                 max_steps,
                 beta_min,
                 beta_max,
                 schedule,
                 pred_target,
                 clip_samples,
                 clip_samples_min,
                 clip_samples_max,
                 clip_samples_scope,
                 dev,
                 objective='diffusion'):
        super().__init__(clip_samples, clip_samples_min, clip_samples_max, objective, dev)

        assert clip_samples_scope == 'x_0'

        self.max_steps = max_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.schedule = schedule
        assert schedule in ['cosine', 'linear']

        self.pred_target = pred_target
        assert pred_target in ['epsilon', 'mu', 'x_0']

        self._const_beta_t = get_beta_t(torch.arange(max_steps),
                                        max_steps, beta_min, beta_max, schedule).to(self.dev)
        self._const_alpha_t = get_alpha_t(torch.arange(max_steps),
                                          max_steps, beta_min, beta_max, schedule).to(self.dev)
        self._const_alpha_t_bar = get_alpha_t_bar(torch.arange(max_steps),
                                                  max_steps, beta_min, beta_max, schedule).to(self.dev)
        self._const_sigma_t = get_sigma_t(torch.arange(max_steps),
                                          max_steps, beta_min, beta_max, schedule).to(self.dev)

    def sample(self, model, node_flags, init_adjs=None, sanity_check_gt_adjs=None,
               flag_interim_adjs=False, max_num_interim_adjs=None,
               gt_score_min_step=None, gt_score_max_step=None):
        """
        Generate samples for DDPM.
        @param model: network whose input arguments are [x, adjs, node_flags, conditions]
        @param node_flags: [B, N]
        @param init_adjs: None for drawing initial adjs from pure noise, otherwise use the given adjs
        @param sanity_check_gt_adjs: None for model inference, otherwise compute ground-truth target from gt adjs.
        @param flag_interim_adjs: bool, return the interim adjs
        @param max_num_interim_adjs: None for unlimited num of adjs, otherwise take evenly-separated snapshots.
        @param gt_score_min_step: gt score sampling interval min diffusion step
        @param gt_score_max_step: gt score sampling interval max diffusion step
        """
        if init_adjs is None:
            init_adjs = self.gen_init_sample(node_flags, folded_norm=True)  # [B, N, N], already masked
        adjs = init_adjs
        adjs_ls, x_0_pred_ls = [init_adjs.cpu()], [init_adjs.cpu()]
        i_step = 0
        if max_num_interim_adjs is None:
            timesteps_snapshot = np.arange(self.max_steps)
        else:
            timesteps_snapshot = np.linspace(0, self.max_steps, max_num_interim_adjs).astype(
                                    int).clip(max=self.max_steps-1)
        for timestep in np.arange(self.max_steps)[::-1]:
            # the adjs and x_0_pred in each loop are already masked
            if gt_score_min_step is None and gt_score_max_step is None:
                adjs, x_0_pred = self._step_sample(model, adjs, node_flags, torch.tensor([timestep]).to(self.dev),
                                                   sanity_check_gt_adjs=sanity_check_gt_adjs)
            elif gt_score_min_step is not None and gt_score_max_step is not None:
                # DEBUG: use ground-truth score function in the epsilon objective
                assert sanity_check_gt_adjs is not None
                adjs, x_0_pred = self._step_sample(model, adjs, node_flags, torch.tensor([timestep]).to(self.dev),
                                                   sanity_check_gt_adjs=sanity_check_gt_adjs,
                                                   gt_score_min_step=gt_score_min_step,
                                                   gt_score_max_step=gt_score_max_step)
            else:
                raise NotImplementedError
            if flag_interim_adjs:
                if timestep in timesteps_snapshot:
                    adjs_ls.append(adjs.cpu())
                    x_0_pred_ls.append(x_0_pred.cpu())
            logging.debug("DDPM MCMC: step {:5d} | avg. #edges@0.0 of x_t & x_0_pred: {:08d} & {:08d} |"
                          .format(i_step,
                                  int(DiffusionSampler.get_num_edges(adjs, node_flags, 0.0).mean().item()),
                                  int(DiffusionSampler.get_num_edges(x_0_pred, node_flags, 0.0).mean().item())
                                  )
                          )
            i_step += 1
        adjs = adjs.cpu()
        if flag_interim_adjs:
            return adjs, (torch.stack(adjs_ls), torch.stack(x_0_pred_ls))
        else:
            return adjs

    def _step_sample(self, model, input_adjs, node_flags, timestep, sanity_check_gt_adjs=None,
                     gt_score_min_step=None, gt_score_max_step=None):
        batch_size = input_adjs.size(0)

        timesteps = timestep.expand(batch_size)
        alpha_t = torch.index_select(self._const_alpha_t, 0, timesteps).to(input_adjs).view(-1, 1, 1)  # [B, 1, 1]
        alpha_t_bar = torch.index_select(self._const_alpha_t_bar, 0, timesteps).to(input_adjs
                                                                                   ).view(-1, 1, 1)  # [B, 1, 1]
        beta_t = torch.index_select(self._const_beta_t, 0, timesteps).to(input_adjs).view(-1, 1, 1)  # [B, 1, 1]
        sigma_t = torch.index_select(self._const_sigma_t, 0, timesteps).to(input_adjs)  # [B]

        if timestep > 0:
            alpha_t_minus_one_bar = torch.index_select(self._const_alpha_t_bar, 0, timesteps - 1
                                                       ).view(-1, 1, 1)  # [B, 1, 1]
            coef_x0 = alpha_t_minus_one_bar.sqrt() * beta_t / (1.0 - alpha_t_bar)  # [B, 1, 1]
            coef_xt = alpha_t.sqrt() * (1.0 - alpha_t_minus_one_bar) / (1.0 - alpha_t_bar)  # [B, 1, 1]
        else:
            alpha_t_minus_one_bar, coef_x0, coef_xt = None, None, None

        # DEBUG mode: get gt score, this only works for epsilon prediction objective
        _flag_use_gt_score_ever = gt_score_min_step is not None and gt_score_max_step is not None
        _flag_use_gt_score_this_step = False
        if _flag_use_gt_score_ever:
            assert sanity_check_gt_adjs is not None
            assert self.pred_target == 'epsilon'
            assert input_adjs.size(0) == 1

            # only do the debugging for batch size 1
            if gt_score_min_step <= timestep.item() <= gt_score_max_step:
                _flag_use_gt_score_this_step = True
                x_vec = input_adjs.view(-1)  # [N^2]
                num_nodes = input_adjs.size(1)

                if not hasattr(self, 'x0_all_perm_vec'):
                    all_perm_mats = torch.from_numpy(get_all_permutation(num_nodes)).to(input_adjs)  # [X, N, N]
                    num_perm = all_perm_mats.size(0)
                    x0_all_perm = sanity_check_gt_adjs.expand(num_perm, -1, -1)  # [X, N, N]
                    x0_all_perm = all_perm_mats @ x0_all_perm @ all_perm_mats.transpose(-1, -2)  # [X, N, N]
                    x0_all_perm_vec = x0_all_perm.view(num_perm, -1)  # [X, N^2]
                    x0_all_perm_vec = torch.unique(x0_all_perm_vec, dim=0)  # [X, N^2], with a smaller X
                    self.x0_all_perm_vec = x0_all_perm_vec
                    logging.info("Creating GMM centroids used in the GT score training...")
                    logging.info("Number of unique components: {:d}".format(x0_all_perm_vec.size(0)))
                else:
                    x0_all_perm_vec = self.x0_all_perm_vec  # [X, N^2]

                mu_vec = x0_all_perm_vec  # [X, N^2]
                gt_score = compute_score(x_vec, mu_vec, alpha_t_bar.item()
                                         ).view(1, num_nodes, num_nodes)  # [1, N, N]

                # convert back to epsilon prediction target
                eps_pred_from_gt_score = -gt_score * np.sqrt(1.0 - alpha_t_bar.item())
            else:
                eps_pred_from_gt_score = None

        # do reverse process for one step, get the mean of the posterior
        scaling_coef = alpha_t_bar.view(-1).sqrt()  # [B]
        if self.pred_target == 'epsilon':
            if _flag_use_gt_score_ever:
                if _flag_use_gt_score_this_step:
                    eps_pred = eps_pred_from_gt_score
                else:
                    with torch.no_grad():
                        eps_pred = model(input_adjs, node_flags, timesteps)
            else:
                if sanity_check_gt_adjs is None:
                    with torch.no_grad():
                        eps_pred = model(input_adjs, node_flags, timesteps)
                else:
                    eps_pred = 1.0 / torch.sqrt(1.0 - alpha_t_bar) * (
                            input_adjs - alpha_t_bar.sqrt() * sanity_check_gt_adjs)

            x_0_pred = (1.0 / alpha_t_bar.sqrt()) * (input_adjs - torch.sqrt(1.0 - alpha_t_bar) * eps_pred)
            if timestep > 0 and self.clip_samples:
                # rewrite the predicted x_0 sample to do clipping
                # only activated when doing denoising for t > 0
                x_0_pred.clamp_(min=self.clip_samples_min, max=self.clip_samples_max)
                mu_pred = x_0_pred * coef_x0 + input_adjs * coef_xt  # [B, N, N]
            else:
                scale_coef = 1.0 / alpha_t.sqrt()  # [B]
                eps_pred_coef = (1.0 - alpha_t) / (1.0 - alpha_t_bar).sqrt()  # [B]
                mu_pred = scale_coef * (input_adjs - eps_pred_coef * eps_pred)  # [B, N, N]
        elif self.pred_target == 'mu':
            if sanity_check_gt_adjs is None:
                with torch.no_grad():
                    mu_pred = model(input_adjs, node_flags, timesteps)  # [B, N, N]
            else:
                mu_pred = coef_x0 * sanity_check_gt_adjs + coef_xt * input_adjs
            if timestep > 0:
                x_0_pred = (mu_pred - coef_xt * input_adjs) / coef_x0
            else:
                x_0_pred = mu_pred
        elif self.pred_target == 'x_0':
            if sanity_check_gt_adjs is None:
                with torch.no_grad():
                    x_0_pred = model(input_adjs, node_flags, timesteps)  # [B, N, N]
            else:
                x_0_pred = sanity_check_gt_adjs
            if timestep > 0:
                # only activated when doing denoising for t > 0
                if self.clip_samples:
                    x_0_pred.clamp_(min=self.clip_samples_min, max=self.clip_samples_max)
                mu_pred = x_0_pred * coef_x0 + input_adjs * coef_xt
            else:
                mu_pred = x_0_pred  # [B, N, N]
        else:
            raise NotImplementedError

        # add noise to the posterior mean
        if timestep > 0:
            output_adjs, _ = add_sym_normal_noise(mu_pred, torch.ones_like(sigma_t), sigma_t, node_flags)
        elif timestep == 0:
            output_adjs = mu_pred
        else:
            raise NotImplementedError

        output_adjs = mask_adjs(output_adjs, node_flags)
        x_0_pred = mask_adjs(x_0_pred, node_flags)
        return output_adjs, x_0_pred

