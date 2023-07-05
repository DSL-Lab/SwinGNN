import logging
import pdb

import numpy as np
from sympy.utilities.iterables import multiset_permutations

import torch
import torch.nn as nn
from torch import nn as nn

from utils.graph_utils import mask_adjs, mask_nodes
from runner.sanity_check_helper import get_all_permutation


def get_reweight_coef(adjs_gt, node_flags, objective):
    """
    Get reweight coefficients. The rule-of-thumb is to increase weight for entries == 1 as they are very sparse.
    @param adjs_gt:         [B, N, N]
    @param node_flags:      [B, N]
    @param objective:       string, ['diffusion', 'score']
    @return reweight_coef:  [B, N, N]
    """
    reweight_coef = torch.zeros_like(adjs_gt)
    node_flags = node_flags.bool()
    if objective == 'score':
        value_zeros, value_ones = 0.0, 1.0
    elif objective in ['diffusion', 'edm']:
        # data scaling for DDPM or EDM
        value_zeros, value_ones = -1.0, 1.0
    else:
        raise NotImplementedError
    num_zeros = torch.sum(adjs_gt[node_flags] == value_zeros)  # occurrence of zero entries
    num_ones = torch.sum(adjs_gt[node_flags] == value_ones)  # occurrence of one entries
    ratio_zero_over_one = num_zeros / num_ones  # scalar
    assert ratio_zero_over_one > 1.0 and num_zeros > 0 and num_ones > 0

    reweight_coef[adjs_gt == value_zeros] = 1.0                               # [B, N, N]
    reweight_coef[adjs_gt == value_ones] = ratio_zero_over_one                # [B, N, N]
    reweight_coef = mask_adjs(reweight_coef, node_flags)                      # [B, N, N]
    return reweight_coef


class RainbowLoss(nn.Module):
    def __init__(self, regression_loss_weight, flag_reweight, objective):
        """
        Rainbow loss with multiple ingredients.
        - Reweight regularization
        - Graph matching loss (debug)
        """

        super(RainbowLoss, self).__init__()

        self.regression_loss_weight = regression_loss_weight

        self.flag_reweight = flag_reweight

        self.objective = objective

        assert objective in ['score', 'diffusion', 'edm'], "Loss mode {:s} is not supported!".format(objective)
        self.all_perm_mat = None

    def forward(self, net_pred, net_target, net_cond, adjs_perturbed, adjs_gt, node_flags, loss_weight=None,
                cond_val=None, flag_matching=False, reduction='mean'):
        # DEBUG: permute the target by graph matching
        if flag_matching:
            net_target = self.graph_matching(net_pred, net_target, adjs_perturbed, adjs_gt, node_flags,)

        # [B, N, N] or None
        reweight_coef = get_reweight_coef(adjs_gt, node_flags, self.objective) if self.flag_reweight else None

        regression_loss = self.get_regression_loss(net_pred, net_target, net_cond, node_flags, reweight_coef, loss_weight, cond_val, reduction)

        return regression_loss

    def get_regression_loss(self, predictions, targets, conditions, node_flags, reweight_coef, loss_weight,
                            condition_true_values, reduction):
        """
        Compute regression loss for score estimation or epsilon-noise prediction.
        @param predictions:             [B, N, N]
        @param targets:                 [B, N, N]
        @param conditions:              [B]
        @param node_flags:              [B, N]
        @param reweight_coef:           [B, N, N]
        @param loss_weight:             [B]
        @param condition_true_values:   [B]
        @param reduction:               str
        @return score_loss:             scalar or [B], loss per entry
        """
        loss_weight = torch.ones_like(conditions).float() if loss_weight is None else loss_weight  # [B]
        loss_weight = loss_weight[:, None, None]
        reweight_coef = 1.0 if reweight_coef is None else reweight_coef
        square_loss = (predictions - targets) ** 2  # [B, N, N]

        # tensor shape reduction
        assert len(node_flags.shape) == 2
        num_adj_entries = node_flags.sum(dim=-1) ** 2  # [B]

        def _reduction(matrix_form_loss):
            # matrix_form_loss: [B, N, N]
            if reduction == 'mean':
                # return scalar
                matrix_form_loss = matrix_form_loss.sum() / num_adj_entries * self.regression_loss_weight
            elif reduction is None or reduction == 'none':
                # return [B]
                matrix_form_loss = matrix_form_loss.sum(dim=[-1, -2]) / num_adj_entries * self.regression_loss_weight
            return matrix_form_loss

        if self.objective == "score":
            score_loss = 0.5 * (condition_true_values ** 2)[:, None, None] * square_loss  # [B, N, N]
            score_loss = score_loss * reweight_coef * loss_weight  # [B, N, N]
            score_loss = mask_adjs(score_loss, node_flags)  # [B, N, N]
            return _reduction(score_loss)
        elif self.objective in ["diffusion", 'edm']:
            diffusion_loss = square_loss * reweight_coef * loss_weight  # [B, N, N]
            diffusion_loss = mask_adjs(diffusion_loss, node_flags)  # [B, N, N]
            return _reduction(diffusion_loss)
        else:
            raise NotImplementedError

    def graph_matching(self, net_pred, net_target, adjs_perturbed, adjs_gt, node_flags,):
        """
        Graph mathing ops for debug purpose.
        """
        """scipy's simple graph matching loss using approximation algorithm"""
        # warning: it is not so good. post-optimization results may be even worse sometimes.
        # from scipy.optimize import quadratic_assignment
        # import numpy as np
        # res = quadratic_assignment(net_pred.cpu().detach().numpy()[0], net_target.cpu().detach().numpy()[0], method='faq', options={"maximize": True})
        # perm = res['col_ind']  # [N]
        # perm_mat = torch.from_numpy(np.eye(net_pred.size(1), dtype=int)[perm]).unsqueeze(0).to(net_pred)  # [1, N, N]
        #
        # err_before = torch.linalg.matrix_norm((net_pred - net_target).detach()).item()
        # err_after = torch.linalg.matrix_norm((net_pred - perm_mat @ net_target @ perm_mat.transpose(-1, -2)).detach()).item()
        # print("ERR before matching: {:.3f}, after matching: {:.3f}".format(err_before, err_after))
        #
        # net_target = perm_mat @ net_target @ perm_mat.transpose(-1, -2)  # match the loss first, w/o SGD

        """brute-force graph matching"""
        # this should only be enabled when the number of nodes is small, otherwise it's intractable!
        assert net_target.size(0) == 1  # batch size = 1, debug mode

        if self.all_perm_mat is None:
            self.all_perm_mat = torch.from_numpy(get_all_permutation(net_target.size(-1))).to(net_target)  # [X, N, N]
        all_perm_mat = self.all_perm_mat

        # permute based on the prediction only, this is buggy
        # err_before = torch.linalg.matrix_norm((net_pred - net_target).detach()).item()
        # net_target_perm = all_perm_mat @ net_target @ all_perm_mat.transpose(-1, -2)  # [X, N, N]
        # pred_target_dist = (net_pred - net_target_perm).square()  # [X, N, N]
        # pred_target_dist = mask_adjs(pred_target_dist, node_flags).sum(dim=[-1, -2])  # [X]
        # i_best_matching = pred_target_dist.argmin().item()  # integer
        # net_target = net_target_perm[i_best_matching]

        # permute the network target such that it's closest to the input (perturbed adjs)
        err_before = torch.linalg.matrix_norm((net_pred - net_target).detach()).item()
        adjs_gt_perm = all_perm_mat @ adjs_gt @ all_perm_mat.transpose(-1, -2)  # [X, N, N]
        adjs_noisy_gt_dist = (adjs_perturbed - adjs_gt_perm).square()  # [X, N, N]
        adjs_noisy_gt_dist = mask_adjs(adjs_noisy_gt_dist, node_flags).sum(dim=[-1, -2])  # [X]
        i_best_matching = adjs_noisy_gt_dist.argmin().item()  # integer
        net_target = all_perm_mat[i_best_matching] @ net_target @ all_perm_mat[i_best_matching].transpose(-1, -2)  # [1, N, N]

        err_after = torch.linalg.matrix_norm((net_pred - net_target).detach()).item()
        logging.info("ERR before matching: {:.3f}, after matching: {:.3f}".format(err_before, err_after))

        return net_target


class NodeAdjRainbowLoss(nn.Module):
    def __init__(self, edge_loss_weight, node_loss_weight, objective, flag_reweight=False,):
        """
        Rainbow loss with multiple ingredients.
        - Reweight regularization
        - Graph matching loss (debug)
        """

        super(NodeAdjRainbowLoss, self).__init__()

        self.edge_loss_weight = edge_loss_weight
        self.node_loss_weight = node_loss_weight
        self.flag_reweight = flag_reweight

        self.objective = objective

        assert objective in ['score', 'diffusion', 'edm'], "Loss mode {:s} is not supported!".format(objective)
        self.all_perm_mat = None

    def forward(self, net_pred_a, net_pred_x, net_target_a, net_target_x,  net_cond,
                adjs_perturbed, adjs_gt, x_perturbed, x_gt, node_flags,
                loss_weight=None, cond_val=None, flag_matching=False,
                reduction='mean'):
        if flag_matching:
            raise ValueError("Graph matching is not supported for node-adj loss!")
        reweight_coef = None
        regression_loss = self.get_regression_loss(net_pred_a, net_pred_x, net_target_a, net_target_x, net_cond,
                                                   node_flags, reweight_coef, loss_weight, cond_val, reduction)

        return regression_loss

    def get_regression_loss(self, pred_adj, pred_node, target_adj, target_node, net_cond,
                            node_flags, reweight_coef, loss_weight,
                            condition_true_values, reduction):
        """
        Compute regression loss for score estimation or epsilon-noise prediction.
        @param pred_adj:                [B, N, N] or [B, C, N, N]
        @param pred_node:               [B, N] or [B, C, N]
        @param target_adj:              [B, N, N] or [B, C, N, N]
        @param target_node:             [B, N] or [B, C, N]
        @param net_cond:                [B]
        @param node_flags:              [B, N] or [B, N, N]
        @param reweight_coef:           [B, N, N]
        @param loss_weight:             [B]
        @param condition_true_values:   [B]
        @param reduction:               str
        @return score_loss:             scalar or [B], loss per entry
        """
        loss_weight = torch.ones_like(net_cond).float() if loss_weight is None else loss_weight  # [B]
        _loss_weight = loss_weight.view(-1)
        batch_size = len(_loss_weight)
        # loss_weight = loss_weight[:, None, None]  # [B, N, N]
        if self.objective == "score":
            raise NotImplementedError
        elif self.objective in ["diffusion", 'edm']:
            square_loss_adj = (pred_adj - target_adj) ** 2  # [B, N, N] or [B, C, N, N]
            square_loss_node = (pred_node - target_node) ** 2  # [B, N] or [B, N, C]
            reweight_coef = 1.0 if reweight_coef is None else reweight_coef

            # [B, N, N] or [B, C, N, N]
            _loss_weight_shape = [batch_size] + [1] * (len(square_loss_adj.shape) - 1)
            square_loss_adj = square_loss_adj * reweight_coef * loss_weight.view(_loss_weight_shape)

            # [B, N] or [B, N, C]
            _loss_weight_shape = [batch_size] + [1] * (len(square_loss_node.shape) - 1)
            square_loss_node = square_loss_node * reweight_coef * loss_weight.view(_loss_weight_shape)

            square_loss_adj = mask_adjs(square_loss_adj, node_flags)  # [B, N, N] or [B, C, N, N]
            square_loss_node = mask_nodes(square_loss_node, node_flags)  # [B, N] or [B, N, C]

            # tensor shape reduction
            if len(node_flags.shape) == 2:
                num_adj_entries = node_flags.sum(dim=-1) ** 2       # [B]
                num_node_entries = node_flags.sum(dim=-1)           # [B]
            else:
                num_adj_entries = node_flags.sum(dim=[-1, -2])      # [B]
                num_node_entries = node_flags.sum(dim=[-1, -2])     # [B]

            if reduction == 'mean':
                square_loss_adj = square_loss_adj.sum() / num_adj_entries * self.edge_loss_weight       # scalar
                square_loss_node = square_loss_node.sum() / num_node_entries * self.edge_loss_weight    # scalar
            elif reduction is None or reduction == 'none':
                # keep the output in the shape of [B]
                if len(square_loss_adj.shape) == 3:
                    square_loss_adj = square_loss_adj.sum(dim=[-1, -2]) / num_adj_entries
                elif len(square_loss_adj.shape) == 4:
                    square_loss_adj = square_loss_adj.sum(dim=[-1, -2, -3]) / num_adj_entries / square_loss_adj.size(1)
                square_loss_adj = square_loss_adj * self.edge_loss_weight  # [B]

                if len(square_loss_node.shape) == 2:
                    square_loss_node = square_loss_node.sum(dim=-1) / num_node_entries
                elif len(square_loss_node.shape) == 3:
                    square_loss_node = square_loss_node.sum(dim=[-1, -2]) / num_node_entries / square_loss_node.size(-1)
                square_loss_node = square_loss_node * self.node_loss_weight  # [B]
            return square_loss_adj, square_loss_node
        else:
            raise NotImplementedError
