import pdb

import numpy as np
import torch


def node_feature_to_matrix(x):
    """
    Aggregate node feature vector into matrix.
    Original implementation of EDP-GNN.
    :param x:  BS x N x F
    :return:
    x_pair: BS x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(-1, -1, x.size(1), -1)  # BS x N x N x F
    x_b_t = x_b.transpose(1, 2)  # BS x N x N x F
    x_pair = torch.cat([x_b, x_b_t], dim=-1)  # BS x N x N x 2F
    return x_pair


def _node_feature_to_matrix_add(input_feat, triu_sparse=False):
    """
    Aggregate node feature vector into matrix by addition.
    @param input_feat: [B, N, D]
    @return: [B, N*(N+1)/2, D] if sparse or [B, N, N, D] if not sparse
    """
    if triu_sparse:
        n = input_feat.size(1)
        rows, cols = torch.triu_indices(n, n).to(input_feat.device).unbind()  # each index vector [N*(N+1)/2]

        # gather ver
        output_feat = torch.gather(input_feat, dim=1,
                                   index=rows.view(1, -1, 1).expand(
                                       input_feat.size(0), -1, input_feat.size(-1))  # [B, N*(N+1)/2, D]
                                   )  # [B, N*(N+1)/2, D]

        output_feat += torch.gather(input_feat, dim=1,
                                    index=cols.view(1, -1, 1).expand(
                                        input_feat.size(0), -1, input_feat.size(-1))  # [B, N*(N+1)/2, D]
                                    )  # [B, N*(N+1)/2, D]
        # index select ver
        # output_feat = torch.index_select(input_feat, dim=1, index=rows)  # [B, N*(N+1)/2, D]
        # output_feat += torch.index_select(input_feat, dim=1, index=cols)
    else:
        output_feat = input_feat[:, None, :, :] + input_feat[:, :, None, :]  # [B, N, N, D], symmetry is ensured
    return output_feat


def _extract_triu(input_mat):
    """
    Extract the upper triangular part of the symmetric matrix into vectors.
    @param input_mat: [*, N, N, D], must be symmetric in the -2 and -3 dimensions
    @return: [*, N(N+1)/2, D] in upper triangular form
    """
    input_mat_shape = input_mat.shape
    triu_mask = torch.triu(torch.ones(input_mat_shape[:-1])).bool().to(input_mat.device)  # [*, N, N]
    out_vec = torch.masked_select(input_mat, triu_mask.unsqueeze(-1))
    out_vec = out_vec.view(*input_mat_shape[:-3], -1, input_mat_shape[-1])  # [*, N(N+1)/2, D]
    # assert out_vec.size(-2) == input_mat_shape[-2] * (input_mat_shape[-2] + 1) / 2  # N(N+1)/2
    return out_vec


def _expand_triu(input_vec):
    """
    Expand the vectors into the corresponding symmetric matrix.
    @param input_vec: [*, N(N+1)/2, D], expand back to matrix according to triu indexing
    @return: [*, N, N, D], symmetry is ensured
    """
    s, d = input_vec.size(-2), input_vec.size(-1)
    n = int((-1.0 + np.sqrt(1.0 + 8.0 * s)) / 2)
    # assert s == n * (n + 1) / 2  # s == n * (n + 1) / 2

    input_vec_shape = input_vec.shape
    output_mat = torch.zeros(*input_vec_shape[:-2], n, n, d).to(input_vec)  # [*, N, N, D]
    i, j = torch.triu_indices(n, n)

    if len(input_vec_shape) == 3:
        output_mat[:, i, j] = input_vec  # [B, N, N, D]
        output_mat[:, j, i] = input_vec
    else:
        raise NotImplementedError
    return output_mat


def edge_feat_fast_update(layer, adjs, keep_triu_vec=False):
    """
    Update the multichannel adj. efficiently
    @param layer: MLP layer
    @param adjs: [B, C, N, N]
    @param keep_triu_vec: flag to return sparse vector or full matrix
    """
    adjs = adjs.permute(0, 2, 3, 1)  # [B, N, N, C]
    mlp_in = _extract_triu(adjs)  # [B, N*(N+1)/2, C]
    mlp_out = layer(mlp_in)  # [B, N*(N+1)/2, C_out]
    if keep_triu_vec:
        return mlp_out  # [B, N*(N+1)/2, C_out]
    else:
        mlp_out = _expand_triu(mlp_out)  # [B, N, N, C_out]
        return mlp_out.permute(0, 3, 1, 2)  # [B, C_out, N, N]
