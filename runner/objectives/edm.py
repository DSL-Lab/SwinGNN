from collections import namedtuple
import torch

from utils.graph_utils import check_adjs_symmetry, mask_adjs, mask_nodes, add_sym_normal_noise
from . import TrainingObjectiveGenerator

VP_PARAMS = namedtuple('vp_params',
                       [
                           'beta_d',                # Extent of the noise level schedule.
                           'beta_min',              # Initial slope of the noise level schedule.
                           'epsilon_t',             # Minimum t-value used during training.
                           'M',                     # Original number of timesteps in the DDPM formulation.
                           'epsilon_s',             # Sampler discretization parameter.
                           'sigma_min_training',    # Minimum supported noise level for training.
                           'sigma_max_training',    # Maximum supported noise level for training.
                           'sigma_min_sampling',    # Minimum supported noise level for sampling.
                           'sigma_max_sampling'     # Maximum supported noise level for sampling.
                       ])

VE_PARAMS = namedtuple('ve_params',
                       [
                           'sigma_min_training',    # Minimum supported noise level for training.
                           'sigma_max_training',    # Maximum supported noise level for training.
                           'sigma_min_sampling',    # Minimum supported noise level for sampling.
                           'sigma_max_sampling'     # Maximum supported noise level for sampling.
                       ])

EDM_PARAMS = namedtuple('edm_params',
                        [
                            'sigma_min_training',   # Minimum supported noise level for training.
                            'sigma_max_training',   # Maximum supported noise level for training.
                            'sigma_min_sampling',   # Minimum supported noise level for sampling.
                            'sigma_max_sampling',   # Maximum supported noise level for sampling.
                            'sigma_data',           # Expected standard deviation of the training data.
                            'P_mean',               # Sigma's log-normal distribution parameter.
                            'P_std',                # Sigma's log-normal distribution parameter.
                            'rho'                   # Sampler discretization parameter.
                        ])


def get_vp_params():
    epsilon_t, epsilon_s = 1e-5, 1e-3
    vp_sigma_min_training = float(get_vp_sigma_from_t(t=epsilon_t))
    vp_sigma_max_training = float(get_vp_sigma_from_t(t=1.0))
    vp_sigma_min_sampling = float(get_vp_sigma_from_t(t=epsilon_s))
    vp_sigma_max_sampling = float(get_vp_sigma_from_t(t=1.0))

    assert vp_sigma_min_sampling >= vp_sigma_min_training

    return VP_PARAMS(beta_d=19.9, beta_min=0.1, epsilon_t=epsilon_t, M=1000, epsilon_s=epsilon_s,
                     sigma_min_training=vp_sigma_min_training, sigma_max_training=vp_sigma_max_training,
                     sigma_min_sampling=vp_sigma_min_sampling, sigma_max_sampling=vp_sigma_max_sampling)


def get_ve_params():
    return VE_PARAMS(sigma_min_training=0.02, sigma_max_training=100.0,
                     sigma_min_sampling=0.02, sigma_max_sampling=100.0)


def get_edm_params():
    return EDM_PARAMS(sigma_min_training=0.0, sigma_max_training=float('inf'),
                      sigma_min_sampling=0.002, sigma_max_sampling=80.0,
                      sigma_data=0.5, P_mean=-1.2, P_std=1.2, rho=7)


def get_vp_sigma_from_t(t, beta_d=19.9, beta_min=0.1):
    t = torch.as_tensor(t)
    return ((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1).sqrt()


def get_vp_sigma_deriv_t(t, beta_d=19.9, beta_min=0.1):
    t = torch.as_tensor(t)
    return 0.5 * (beta_min + beta_d * t) * (get_vp_sigma_from_t(t) + 1 / get_vp_sigma_from_t(t))


def get_vp_t_from_sigma(sigma, beta_d=19.9, beta_min=0.1):
    sigma = torch.as_tensor(sigma)
    return ((beta_min ** 2 + 2 * beta_d * (1 + sigma ** 2).log()).sqrt() - beta_min) / beta_d


def get_ve_sigma_from_t(t):
    t = torch.as_tensor(t)
    return t.sqrt()


def get_ve_sigma_deriv_t(t):
    t = torch.as_tensor(t)
    return 0.5 / t.sqrt()


def get_ve_t_from_sigma(sigma):
    sigma = torch.as_tensor(sigma)
    return sigma ** 2


def get_edm_sigma_from_t(t):
    t = torch.as_tensor(t)
    return t


def get_edm_sigma_deriv_t(t):
    t = torch.as_tensor(t)
    return torch.ones_like(t)


def get_edm_t_from_sigma(sigma):
    sigma = torch.as_tensor(sigma)
    return sigma


def get_preconditioning_params(precond, sigmas, vp_params, ve_params, edm_params):
    if precond == 'vp':
        c_skip = 1
        c_out = -sigmas
        c_in = 1 / (sigmas ** 2 + 1).sqrt()
        c_noise = (vp_params.M - 1) * get_vp_t_from_sigma(sigmas)
    elif precond == 've':
        c_skip = 1
        c_out = sigmas
        c_in = 1
        c_noise = (0.5 * sigmas).log()
    elif precond == 'edm':
        c_skip = edm_params.sigma_data ** 2 / (sigmas ** 2 + edm_params.sigma_data ** 2)
        c_out = sigmas * edm_params.sigma_data / (sigmas ** 2 + edm_params.sigma_data ** 2).sqrt()
        c_in = 1 / (edm_params.sigma_data ** 2 + sigmas ** 2).sqrt()
        c_noise = sigmas.log() / 4
    else:
        raise NotImplementedError
    return c_skip, c_out, c_in, c_noise


class EDMObjectiveGenerator(TrainingObjectiveGenerator):
    """
    Training objective generator for diffusion model at the adjacency matrix space.
    """

    def __init__(self,
                 precond,
                 sigma_dist,
                 # other params
                 *,
                 other_params,
                 dev,
                 objective="edm"):
        super().__init__(objective, dev)

        self.precond = precond
        self.sigma_dist = sigma_dist

        assert precond in ['vp', 've', 'edm']
        assert sigma_dist in ['vp', 've', 'edm']

        self.vp_params = get_vp_params()
        self.ve_params = get_ve_params()
        self.edm_params = get_edm_params()

        self.other_params = other_params

    """helper functions regarding preconditioning and loss"""
    def get_training_sigmas_weights(self, num_samples):
        """
        Training (Section 5) tuning.
        """
        if self.sigma_dist == 'vp':
            rnd_uniform = torch.rand(num_samples, device=self.dev)
            sigmas = get_vp_sigma_from_t(1 + rnd_uniform * (self.vp_params.epsilon_t - 1))
            weights = 1 / sigmas ** 2
        elif self.sigma_dist == 've':
            rnd_uniform = torch.rand(num_samples, device=self.dev)
            sigmas = self.ve_params.sigma_min_training * (
                    (self.ve_params.sigma_max_training / self.ve_params.sigma_min_training) ** rnd_uniform)
            weights = 1 / sigmas ** 2
        elif self.sigma_dist == 'edm':
            rnd_normal = torch.randn(num_samples, device=self.dev)
            sigmas = (rnd_normal * self.edm_params.P_std + self.edm_params.P_mean).exp()
            weights = (sigmas ** 2 + self.edm_params.sigma_data ** 2) / (sigmas * self.edm_params.sigma_data) ** 2
        else:
            raise NotImplementedError
        return sigmas, weights

    def get_network_input(self, clean_adjs, node_flags, sigmas, *args, **kwargs):
        assert len(sigmas) == len(clean_adjs)
        unit_scales = torch.ones_like(sigmas)  # [B]
        noisy_adjs, noise_added = add_sym_normal_noise(clean_adjs, unit_scales, sigmas, node_flags)  # noise is masked
        check_adjs_symmetry(noisy_adjs)
        return noisy_adjs, noise_added

    def get_input_output(self, clean_adjs, node_flags, *args, **kwargs):
        """
        Get training time network input and output.
        """

        batch_size = clean_adjs.size(0)
        """get training sigmas and weights"""
        sigmas, weights = self.get_training_sigmas_weights(batch_size)  # [B] + [B]

        """get preconditioning coefficients"""
        c_skip, c_out, c_in, c_noise = get_preconditioning_params(self.precond, sigmas,
                                                                  self.vp_params, self.ve_params, self.edm_params)

        """create the raw noisy input"""
        # x = y + n + n, where y is the clean data
        noisy_adjs, noise_added = self.get_network_input(clean_adjs, node_flags, sigmas)

        """rearrange input-output from the training objective generator"""
        # these signals are for the preconditioned D_x network, not the raw network F_x input-output!
        net_input = noisy_adjs      # D_x input, x = y+n
        net_cond = sigmas           # D_x conditional signal
        net_target = clean_adjs     # D_x target, equivalent to DDPM x0-prediction target

        # always return net_input, net_condition, net_target, (c_skip, c_out, c_in, c_noise, sigmas, weights)
        return net_input, net_cond, net_target, (c_skip, c_out, c_in, c_noise, sigmas, weights)


class NodeAdjEDMObjectiveGenerator(EDMObjectiveGenerator):
    """
    Training objective generator for diffusion model for both node and adjacency matrix attributes.
    """

    def __init__(self,
                 precond,
                 sigma_dist,
                 # other params
                 *,
                 other_params,
                 dev,
                 objective="edm",
                 symmetric_noise=True):
        super().__init__(precond, sigma_dist, other_params=other_params, dev=dev, objective=objective)
        self.symmetric_noise = symmetric_noise

    """helper functions regarding preconditioning and loss"""
    def get_network_input(self, clean_adjs, clean_x=None, node_flags=None, sigmas=None, *args, **kwargs):
        assert len(sigmas) == len(clean_adjs)
        # add noise to the adjacency matrix
        unit_scales = torch.ones_like(sigmas)  # [B]
        noisy_adjs, noise_added_to_adjs = add_sym_normal_noise(clean_adjs, unit_scales, sigmas, node_flags,
                                                               non_symmetric=not self.symmetric_noise)
        if self.symmetric_noise:
            check_adjs_symmetry(noisy_adjs)

        # add noise to the node attributes
        _sigmas_shape = [sigmas.shape[0]] + [1] * (len(clean_x.shape) - 1)  # clean_x has shape [B, N] or [B, N, F]
        noise_added_to_x = torch.randn_like(clean_x) * sigmas.view(_sigmas_shape)
        noise_added_to_x = mask_nodes(noise_added_to_x, node_flags)

        noisy_x = clean_x + noise_added_to_x

        return noisy_adjs, noise_added_to_adjs, noisy_x, noise_added_to_x

    def get_input_output(self, clean_adjs, clean_x=None, node_flags=None, *args, **kwargs):
        """
        Get training time network input and output.
        """

        batch_size = clean_adjs.size(0)
        """get training sigmas and weights"""
        sigmas, weights = self.get_training_sigmas_weights(batch_size)  # [B] + [B]

        """get preconditioning coefficients"""
        c_skip, c_out, c_in, c_noise = get_preconditioning_params(self.precond, sigmas,
                                                                  self.vp_params, self.ve_params, self.edm_params)

        """create the raw noisy input"""
        # x = y + n + n, where y is the clean data
        noisy_adjs, noise_added_to_adjs, noisy_x, noise_added_to_x = self.get_network_input(clean_adjs, clean_x,
                                                                                            node_flags, sigmas)
        """rearrange input-output from the training objective generator"""
        # these signals are for the preconditioned D_x network, not the raw network F_x input-output!
        net_input_a = noisy_adjs        # D_x input, x = y+n
        net_cond = sigmas               # D_x conditional signal
        net_target_a = clean_adjs       # D_x target, equivalent to DDPM x0-prediction target
        net_input_x = noisy_x
        net_target_x = clean_x
        # always return net_input, net_condition, net_target, (c_skip, c_out, c_in, c_noise, sigmas, weights)
        return net_input_a, net_input_x, net_cond, net_target_a, net_target_x, (c_skip, c_out, c_in, c_noise, sigmas, weights)
