import pdb

import torch
import numpy as np

from utils.graph_utils import check_adjs_symmetry, mask_adjs, mask_nodes, add_sym_normal_noise, mask_incs
from . import TrainingObjectiveGenerator


def get_beta_t(timesteps, max_steps, beta_min, beta_max, schedule):
    """
    Get the beta_t parameter for diffusion process.
    @param timesteps: [N] tensors
    @param max_steps: scalar, int
    @param beta_min:  scalar, float
    @param beta_max:  scalar, float
    @param schedule:  str, must be in ['cosine', 'linear']
    @return ret_betas: [N] tensors
    """
    if isinstance(timesteps, list) or isinstance(timesteps, np.ndarray):
        timesteps = torch.tensor(timesteps).view(-1)

    if schedule == 'linear':
        all_betas = torch.linspace(beta_min, beta_max, max_steps)
    elif schedule == 'cosine':
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        as implemented in https://github.com/lucidrains/denoising-diffusion-pytorch 
        """
        s = 0.008
        steps = max_steps + 1
        x = torch.linspace(0, max_steps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / max_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        all_betas = torch.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError

    ret_betas = torch.index_select(all_betas, 0, timesteps.long())
    return ret_betas


def get_alpha_t_bar(timesteps, max_steps, beta_min, beta_max, schedule):
    """
    Get alpha_t^bar based on beta_t configuration.
    """
    if isinstance(timesteps, list) or isinstance(timesteps, np.ndarray):
        timesteps = torch.tensor(timesteps).view(-1)
    all_betas = get_beta_t(torch.arange(max_steps), max_steps, beta_min, beta_max, schedule)
    all_alphas = 1.0 - all_betas
    all_alphas_bar = torch.cumprod(all_alphas, dim=0)
    ret_alphas_bar = torch.index_select(all_alphas_bar, 0, timesteps.long())
    return ret_alphas_bar


def get_alpha_t(timesteps, max_steps, beta_min, beta_max, schedule):
    """
    Get alpha_t based on beta_t configuration.
    """
    return 1.0 - get_beta_t(timesteps, max_steps, beta_min, beta_max, schedule)


def get_sigma_t(timesteps, max_steps, beta_min, beta_max, schedule):
    """
    Get sigma_t based on beta_t configuration.
    """
    return get_beta_t(timesteps, max_steps, beta_min, beta_max, schedule).sqrt()


class DiffusionObjectiveGenerator(TrainingObjectiveGenerator):
    """
    Training objective generator for diffusion model at the adjacency matrix space.
    """

    def __init__(self,
                 max_steps,
                 beta_min,
                 beta_max,
                 schedule,
                 pred_target,
                 other_params,
                 dev,
                 objective='diffusion'):
        super().__init__(objective, dev)

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

        self.other_params = other_params

        if self.max_steps == 1:
            self.noisy_adjs, self.noise_added = None, None

    def _get_forward_diffusion_latent(self, adjs, node_flags, scale_coefs, sigmas):
        """
        Forward diffusion process: scale down the original data and add Gaussian noise.
        """

        if self.max_steps == 1:
            # debug mode
            noisy_adjs, noise_added = add_sym_normal_noise(adjs,
                                                           scales=torch.ones_like(scale_coefs) * self.other_params.scaling_coef,
                                                           sigmas=torch.ones_like(scale_coefs) * self.other_params.betas.min,
                                                           node_flags=node_flags)
            if self.other_params.noise_ctrl == 'fixed':
                if self.noisy_adjs is None and self.noise_added is None:
                    self.noisy_adjs, self.noise_added = noisy_adjs.to(sigmas.device), noise_added.to(sigmas.device)
                else:
                    noisy_adjs, noise_added = self.noisy_adjs.to(sigmas.device), self.noise_added.to(sigmas.device)
            elif self.other_params.noise_ctrl.startswith('fixed_'):
                num_noisy_adjs = int(self.other_params.noise_ctrl.split('_')[-1])
                if self.noisy_adjs is None and self.noise_added is None:
                    self.noisy_adjs, self.noise_added = [], []
                    for _ in range(num_noisy_adjs):
                        # create and store many noise data
                        new_noisy_adjs, new_noise_added = add_sym_normal_noise(
                            adjs, scales=torch.ones_like(scale_coefs) * self.other_params.scaling_coef,
                            sigmas=torch.ones_like(scale_coefs) * self.other_params.betas.min,
                            node_flags=node_flags)
                        self.noisy_adjs.append(new_noisy_adjs)
                        self.noise_added.append(new_noise_added)

                assert len(self.noisy_adjs) == num_noisy_adjs
                noise_idx = np.random.choice(num_noisy_adjs)  # integer
                noisy_adjs, noise_added = self.noisy_adjs[noise_idx].to(sigmas.device), self.noise_added[noise_idx].to(sigmas.device)
            elif self.other_params.noise_ctrl == 'random':
                pass
            else:
                raise NotImplementedError
        else:
            # normal mode
            noisy_adjs, noise_added = add_sym_normal_noise(adjs, scale_coefs, sigmas, node_flags)

        check_adjs_symmetry(noisy_adjs)

        # keep unbounded noise injection
        noise_target = noise_added

        noise_target /= sigmas[:, None, None]  # normalize the Gaussian noise
        noisy_adjs = mask_adjs(noisy_adjs, node_flags)
        noise_target = mask_adjs(noise_target, node_flags)
        return noisy_adjs, noise_target

    def get_network_input(self, input_adjs, node_flags, timesteps):

        assert len(timesteps) == len(input_adjs)

        alpha_bars = torch.index_select(self._const_alpha_t_bar, 0, timesteps.long()).to(input_adjs)  # [B]

        # fwd_latent = noisy_adjs  # q(x_t | x_0), forward latents, computed w/ stochasticity
        noisy_adjs, noise_target = self._get_forward_diffusion_latent(input_adjs,
                                                                      node_flags,
                                                                      torch.sqrt(alpha_bars),
                                                                      torch.sqrt(1.0 - alpha_bars))
        return noisy_adjs, noise_target

    def get_conditions(self, num_samples):
        """
        Get random diffusion steps.
        """
        return torch.randint(low=0, high=self.max_steps, size=(num_samples,)).to(self.dev)

    def get_network_target(self, input_adjs, noisy_adjs, noise_target, timesteps):
        """
        Get network output targets.
        """
        if self.pred_target == 'epsilon':
            return noise_target
        elif self.pred_target == 'mu':
            # q(x_t-1 | x_t, x_0), posterior forward latents, computed deterministically

            flag_t_is_zero = timesteps == 0
            timesteps = timesteps.clamp(min=1).long()

            alpha_t_minus_one_bar = torch.index_select(self._const_alpha_t_bar, 0, timesteps-1)  # [B]
            alpha_t_bar = torch.index_select(self._const_alpha_t_bar, 0, timesteps)  # [B]
            beta_t = torch.index_select(self._const_beta_t, 0, timesteps)  # [B]
            alpha_t = torch.index_select(self._const_alpha_t, 0, timesteps)  # [B]

            coef_x0 = alpha_t_minus_one_bar.sqrt() * beta_t / (1.0 - alpha_t_bar)  # [B]
            coef_xt = alpha_t.sqrt() * (1.0 - alpha_t_minus_one_bar) / (1.0 - alpha_t_bar)  # [B]
            mu_pred = input_adjs * coef_x0[:, None, None] + noisy_adjs * coef_xt[:, None, None]

            mu_pred[flag_t_is_zero] = input_adjs[flag_t_is_zero]  # make t == 0 terms equivalent to x0 prediction
            return mu_pred
        elif self.pred_target == 'x_0':
            return input_adjs
        else:
            raise NotImplementedError

    def get_input_output(self, input_adjs, node_flags):
        batch_size = input_adjs.size(0)
        timesteps = self.get_conditions(batch_size)

        noisy_adjs, noise_target = self.get_network_input(input_adjs, node_flags, timesteps)
        net_target = self.get_network_target(input_adjs, noisy_adjs, noise_target, timesteps)

        # always return net_input, net_condition, net_target
        return noisy_adjs, timesteps, net_target

    @property
    def const_beta_t(self):
        return self._const_beta_t

    @property
    def const_alpha_t_bar(self):
        return self._const_alpha_t_bar

    @property
    def const_alpha_t(self):
        return self._const_alpha_t

