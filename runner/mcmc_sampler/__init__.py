import torch

from utils.graph_utils import mask_adjs


class GeneralSampler(object):
    """
    Template for MCMC sampler.
    """
    def __init__(self,
                 clip_samples,
                 clip_samples_min,
                 clip_samples_max,
                 objective,
                 dev,
                 **kwargs):
        super().__init__()

        self.objective = objective
        assert objective in ['diffusion', 'score', 'edm']

        self.dev = dev

        self.clip_samples = clip_samples
        self.clip_samples_min = clip_samples_min
        self.clip_samples_max = clip_samples_max

    def gen_init_sample(self, node_flags, folded_norm=False):
        """
        Generate initial samples.
        @param node_flags: [B, N]
        @param folded_norm: bool
        @return adjs_init: [B, N, N] with proper masking.
        """
        batch_size, max_node_num = node_flags.shape
        init_adjs = torch.randn((batch_size, max_node_num, max_node_num)
                                ).triu(diagonal=1).to(self.dev)
        init_adjs = init_adjs.abs() if folded_norm else init_adjs
        init_adjs = init_adjs + init_adjs.transpose(-1, -2)
        init_adjs = mask_adjs(init_adjs, node_flags)
        return init_adjs

    @staticmethod
    def adj_to_int(adjs_cont, node_flags, threshold):
        adjs_disc = torch.where(adjs_cont < threshold, torch.zeros_like(adjs_cont), torch.ones_like(adjs_cont))
        adjs_disc = mask_adjs(adjs_disc, node_flags)
        return adjs_disc

    @staticmethod
    def get_num_edges(adjs_cont, node_flags, threshold):
        adjs_disc = GeneralSampler.adj_to_int(adjs_cont, node_flags, threshold)
        return (adjs_disc > 0.0).sum([-1, -2]).float() / 2.0

    def sample(self, **kwargs):
        pass

    def _step_sample(self, **kwargs):
        pass
