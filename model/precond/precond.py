import pdb
import numpy as np
import torch
import torch.nn as nn
from torch import nn as nn

from utils.graph_utils import mask_adjs, check_adjs_symmetry, mask_nodes
from runner.objectives.edm import get_vp_params, get_ve_params, get_edm_params, get_preconditioning_params


class Precond(nn.Module):
    def __init__(self, precond, model, self_condition):
        super().__init__()
        self.precond = precond
        assert precond in ['vp', 've', 'edm']

        self.model = model
        self.self_condition = self_condition

        self.vp_params = get_vp_params()
        self.ve_params = get_ve_params()
        self.edm_params = get_edm_params()

    def forward(self, x, node_flags, sigmas, self_cond=None, *args, **model_kwargs):
        # here x means the adjacency matrix data
        c_skip, c_out, c_in, c_noise = get_preconditioning_params(self.precond, sigmas,
                                                                  self.vp_params, self.ve_params, self.edm_params)

        def _expand_tensor_shape(in_tensors):
            if isinstance(in_tensors, torch.Tensor):
                return in_tensors.view(-1, 1, 1)
            elif isinstance(in_tensors, list):
                return [_expand_tensor_shape(item) for item in in_tensors]
            else:
                raise NotImplementedError
        c_skip, c_out, c_in = _expand_tensor_shape([c_skip, c_out, c_in])

        # if len(c_noise.shape) == 0:
        #     c_noise = c_noise.view(-1)

        self_cond = None
        if self.self_condition and np.random.rand() < 0.5:
            with torch.no_grad():
                self_cond = self.model(c_in * x, node_flags, c_noise, None, **model_kwargs)
                self_cond = c_skip * x + c_out * self_cond.to(torch.float32)
                self_cond = mask_adjs(self_cond, node_flags)
                self_cond.detach_()

        F_x = self.model(c_in * x, node_flags, c_noise, self_cond, **model_kwargs)

        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        D_x = mask_adjs(D_x, node_flags)
        check_adjs_symmetry(D_x)
        return D_x

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)


class NodeAdjPrecond(Precond):
    def __init__(self, precond, model, self_condition, symmetric_noise=True):
        super().__init__(precond, model, self_condition)
        self.symmetric_noise = symmetric_noise

    def forward(self, adjs, nodes=None, node_flags=None, sigmas=None, self_cond_adjs=None, self_cond_nodes=None,
                *args, **model_kwargs):
        c_skip, c_out, c_in, c_noise = get_preconditioning_params(self.precond, sigmas,
                                                                  self.vp_params, self.ve_params, self.edm_params)

        def _expand_tensor_shape(in_tensors):
            if isinstance(in_tensors, torch.Tensor):
                return in_tensors.view(-1, 1, 1)
            elif isinstance(in_tensors, list):
                return [_expand_tensor_shape(item) for item in in_tensors]
            else:
                raise NotImplementedError
        c_skip, c_out, c_in = _expand_tensor_shape([c_skip, c_out, c_in])

        # if len(c_noise.shape) == 0:
        #     c_noise = c_noise.view(-1)
        self_cond_adjs = self_cond_adjs
        self_cond_nodes = self_cond_nodes
        c_in_x = c_in.unsqueeze(-1) if len(adjs.shape) == 4 else c_in
        c_skip_x = c_skip.unsqueeze(-1) if len(adjs.shape) == 4 else c_skip
        c_out_x = c_out.unsqueeze(-1) if len(adjs.shape) == 4 else c_out

        c_in_f = c_in.squeeze(-1) if len(nodes.shape) == 2 else c_in
        c_skip_f = c_skip.squeeze(-1) if len(nodes.shape) == 2 else c_skip
        c_out_f = c_out.squeeze(-1) if len(nodes.shape) == 2 else c_out
        if self.self_condition and np.random.rand() < 0.5:
            with torch.no_grad():
                self_cond_adjs, self_cond_nodes = self.model(c_in_x * adjs, c_in_f * nodes, node_flags, c_noise, self_cond_adjs, self_cond_nodes, **model_kwargs)
                self_cond_adjs = c_skip_x * adjs + c_out_x * self_cond_adjs.to(torch.float32)
                self_cond_nodes = c_skip_f * nodes + c_out_f * self_cond_nodes.to(torch.float32)
                self_cond_adjs = mask_adjs(self_cond_adjs, node_flags)
                self_cond_nodes = mask_nodes(self_cond_nodes, node_flags)
                self_cond_adjs.detach_()
                self_cond_nodes.detach_()

        F_x, F_feat = self.model(c_in_x * adjs, c_in_f * nodes, node_flags, c_noise, self_cond_adjs, self_cond_nodes, **model_kwargs)

        D_x = c_skip_x * adjs + c_out_x * F_x.to(torch.float32)
        D_feat = c_skip_f * nodes + c_out_f * F_feat.to(torch.float32)
        D_x = mask_adjs(D_x, node_flags)
        D_feat = mask_nodes(D_feat, node_flags)
        if self.symmetric_noise:
            check_adjs_symmetry(D_x)
        else:
            pass
        return D_x, D_feat

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)
