import pdb

import torch
import numpy as np

from utils.graph_utils import check_adjs_symmetry, mask_adjs, add_sym_normal_noise
from . import TrainingObjectiveGenerator


class ScoreMatchingObjectiveGenerator(TrainingObjectiveGenerator):
    """
    Training objective generator for score-based model.
    """

    def __init__(self,
                 sigma_num_slices,
                 sigma_min,
                 sigma_max,
                 sigma_preset,
                 dev,
                 objective='score'):
        super().__init__(objective, dev)

        self.sigma_num_slices = sigma_num_slices
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # sigmas are arranged from large to small, sigma_0 > sigma_1 > sigma_2 > ...
        if sigma_preset is None:
            self._const_sigma_t = torch.tensor(np.geomspace(sigma_min, sigma_max, sigma_num_slices)).to(self.dev
                                                                                                        ).flip(0)
        else:
            self._const_sigma_t = torch.tensor(sigma_preset).to(self.dev).view(-1).sort(descending=True)[0]
            self.sigma_num_slices = len(self._const_sigma_t)
            self.sigma_min = self._const_sigma_t.min().item()
            self.sigma_max = self._const_sigma_t.max().itme()

    def get_network_input(self, input_adjs, node_flags, sigma_idxs):

        assert len(sigma_idxs) == len(input_adjs)
        sigma_values = torch.index_select(self._const_sigma_t, 0, sigma_idxs.long()).to(input_adjs)  # [B]

        noisy_adjs, noise_added = add_sym_normal_noise(input_adjs, torch.ones_like(sigma_values),
                                                       sigma_values, node_flags)

        # original implementation
        grad_log_noise = - noise_added / (sigma_values[:, None, None] ** 2)

        noisy_adjs = mask_adjs(noisy_adjs, node_flags)
        grad_log_noise = mask_adjs(grad_log_noise, node_flags)

        return noisy_adjs, grad_log_noise

    def get_conditions(self, num_samples):
        """
        Get random sigma indexes.
        """
        return torch.randint(low=0, high=self.sigma_num_slices, size=(num_samples,)).to(self.dev)

    def get_network_target(self, grad_log_noise):
        """
        Get network output targets.
        """
        return grad_log_noise

    def get_input_output(self, input_adjs, node_flags):
        batch_size = input_adjs.size(0)
        sigma_idxs = self.get_conditions(batch_size)

        noisy_adjs, grad_log_noise = self.get_network_input(input_adjs, node_flags, sigma_idxs)
        net_target = self.get_network_target(grad_log_noise)

        # always return net_input, net_condition, net_target
        # note: sigma indexes are used as network conditions, rather than the actual values
        return noisy_adjs, sigma_idxs, net_target

    @property
    def const_sigma_t(self):
        return self._const_sigma_t
