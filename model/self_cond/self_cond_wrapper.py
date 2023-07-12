import numpy as np
import torch
import torch.nn as nn


class SelfCondWrapper(nn.Module):
    """
    Wrapper to accommodate self-conditioning trick in DDP mode.
    Note: for EDM framework, this wrapper is not necessary as the precond wrapper of EDM is doing the same job.
    """
    def __init__(self, model, self_condition):
        super().__init__()

        self.model = model
        self.self_condition = self_condition

    def forward(self, net_input, node_flags, neg_cond, **model_kwargs):
        self_cond = None
        if self.self_condition and np.random.rand() < 0.5:
            with torch.no_grad():
                self_cond = self.model(net_input, node_flags, neg_cond, None, **model_kwargs)
                self_cond.detach_()

        output = self.model(net_input, node_flags, neg_cond, self_cond, **model_kwargs)
        return output
