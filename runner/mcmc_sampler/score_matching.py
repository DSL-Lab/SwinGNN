import torch
import numpy as np
import logging

from utils.graph_utils import check_adjs_symmetry, mask_adjs, add_sym_normal_noise
from . import GeneralSampler


class ScoreMatchingSampler(GeneralSampler):
    """
    MCMC sampler for score-based model.
    """

    def __init__(self,
                 eps_step_size,
                 eps_noise,
                 steps_per_sigma,
                 sigma_num_slices,
                 sigma_min,
                 sigma_max,
                 sigma_preset,
                 clip_samples,
                 clip_samples_min,
                 clip_samples_max,
                 clip_samples_scope,
                 dev,
                 objective='score'):
        super().__init__(clip_samples, clip_samples_min, clip_samples_max, objective, dev)

        assert clip_samples_scope == 'x_t'

        self.eps_step_size = eps_step_size
        self.eps_noise = eps_noise
        self.steps_per_sigma = steps_per_sigma

        self.sigma_num_slices = sigma_num_slices
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        if sigma_preset is None:
            self._const_sigma_t = torch.tensor(np.geomspace(sigma_min, sigma_max, sigma_num_slices)).to(self.dev
                                                                                                        ).flip(0)
        else:
            self._const_sigma_t = torch.tensor(sigma_preset).to(self.dev).view(-1).sort(descending=True)[0]
            self.sigma_num_slices = len(self._const_sigma_t)
            self.sigma_min = self._const_sigma_t.min().item()
            self.sigma_max = self._const_sigma_t.max().itme()

    def sample(self, model, node_flags, init_adjs=None, sanity_check_gt_adjs=None,
               flag_interim_adjs=False):
        """
        Generate samples for score-based model.
        @param model: network whose input arguments are [x, adjs, node_flags, conditions]
        @param node_flags: [B, N]
        @param init_adjs: None for drawing initial adjs from pure noise, otherwise use the given adjs
        @param sanity_check_gt_adjs: None for model inference, otherwise compute ground-truth target from gt adjs.
        @param flag_interim_adjs: bool, return the interim adjs
        """
        if init_adjs is None:
            init_adjs = self.gen_init_sample(node_flags, folded_norm=True)  # [B, N, N]
        adjs = init_adjs
        i_step = 0
        adjs_ls = [adjs.cpu()]
        for i_sigma in range(self.sigma_num_slices):
            for j_iter in range(self.steps_per_sigma):
                # sigma becomes smaller when i_sigma increases
                adjs = self._step_sample(model, adjs, node_flags, torch.tensor([i_sigma]).to(self.dev),
                                         sanity_check_gt_adjs=sanity_check_gt_adjs)
                if flag_interim_adjs:
                    adjs_ls.append(adjs.cpu())
                logging.debug("ScoreMatching MCMC | total step: {:5d} | current sigma: {:3d} ({:.5f}) iter: {:04d} | "
                              "avg. #edges@0.5 of x_t: {:08d} | "
                              .format(i_step,
                                      i_sigma,
                                      self._const_sigma_t[i_sigma].item(),
                                      j_iter,
                                      int(ScoreMatchingSampler.get_num_edges(adjs, node_flags, 0.5).mean().item())
                                      )
                              )
                i_step += 1
        adjs = self._step_sample(model, adjs, node_flags, torch.tensor([self.sigma_num_slices-1]).to(self.dev),
                                 last_step=False, sanity_check_gt_adjs=sanity_check_gt_adjs)
        adjs = adjs.cpu()
        if flag_interim_adjs:
            return adjs, torch.stack(adjs_ls)
        else:
            return adjs

    def _step_sample(self, model, input_adjs, node_flags, sigma_idx, last_step=False, sanity_check_gt_adjs=None):
        batch_size = input_adjs.size(0)
        sigma_value = torch.index_select(self._const_sigma_t, 0, sigma_idx.long()).to(input_adjs)  # [1]
        sigma_idxs = sigma_idx.expand(batch_size)  # [B]
        sigma_values = sigma_value.expand(batch_size)  # [B]

        if not last_step:
            step_sizes = self.eps_step_size * (sigma_values / self.sigma_min) ** 2  # [B]
            sigma_sampling_noises = (step_sizes * 2.0).sqrt() * self.eps_noise  # [B]
        else:
            step_sizes = sigma_values ** 2  # [B]
            sigma_sampling_noises = step_sizes * 0.0  # [B]

        if sanity_check_gt_adjs is None:
            with torch.no_grad():
                score_pred = model(None, input_adjs, node_flags, sigma_idxs)
        else:
            score_pred = -(input_adjs - sanity_check_gt_adjs) / (sigma_values ** 2)

        output_adjs = input_adjs + step_sizes[:, None, None] * score_pred

        if not last_step:
            output_adjs, _ = add_sym_normal_noise(output_adjs,
                                                  torch.ones_like(sigma_sampling_noises),
                                                  sigma_sampling_noises,
                                                  node_flags)

        output_adjs = mask_adjs(output_adjs, node_flags)
        return output_adjs

    @property
    def const_sigma_t(self):
        return self._const_sigma_t
