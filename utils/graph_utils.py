import numpy as np
import torch


def mask_adjs(adjs, node_flags, value=0.0, in_place=True, col_only=False):
    """
    Masking out adjs according to node flags.
    @param adjs:        [B, N, N] or [B, C, N, N]
    @param node_flags:  [B, N] or [B, N, N]
    @param value:       scalar
    @param in_place:    flag of in place operation
    @param col_only:    masking in the column direction only
    @return adjs:       [B, N, N] or [B, C, N, N]
    """
    # assert node_flags.sum(-1).gt(2-1e-5).all(), f"{node_flags.sum(-1).cpu().numpy()}, {adjs.cpu().numpy()}"
    if len(node_flags.shape) == 2:
        # mask adjs by columns and/or by rows, [B, N] shape
        if len(adjs.shape) == 4:
            node_flags = node_flags.unsqueeze(1)  # [B, 1, N]
        if in_place:
            if not col_only:
                adjs.masked_fill_(torch.logical_not(node_flags).unsqueeze(-1), value)
            adjs.masked_fill_(torch.logical_not(node_flags).unsqueeze(-2), value)
        else:
            if not col_only:
                adjs = adjs.masked_fill(torch.logical_not(node_flags).unsqueeze(-1), value)
            adjs = adjs.masked_fill(torch.logical_not(node_flags).unsqueeze(-2), value)
    elif len(node_flags.shape) == 3:
        # mask adjs element-wisely, [B, N, N] shape
        assert node_flags.size(1) == node_flags.size(2) and node_flags.size(1) == adjs.size(2)
        assert not col_only
        if len(adjs.shape) == 4:
            node_flags = node_flags.unsqueeze(1)  # [B, 1, N, N]
        if in_place:
            adjs.masked_fill_(torch.logical_not(node_flags), value)  # [B, N, N] or [B, C, N, N]
        else:
            adjs = adjs.masked_fill(torch.logical_not(node_flags), value)  # [B, N, N] or [B, C, N, N]
    return adjs


def mask_nodes(nodes, node_flags, value=0.0, in_place=True, along_dim=None):
    """
    Masking out node embeddings according to node flags.
    @param nodes:        [B, N] or [B, N, D] by default, [B, *, N, *] if along_dim is specified
    @param node_flags:   [B, N] or [B, N, N]
    @param value:        scalar
    @param in_place:     flag of in place operation
    @param along_dim:    along certain specified dimension
    @return NODES:       [B, N] or [B, N, D]
    """
    if len(node_flags.shape) == 3:
        # raise ValueError("node_flags should be [B, N] or [B, N, D]")
        # if node_flags is [B, N, N], then we don't apply any mask
        return nodes
    elif len(node_flags.shape) == 2:
        if along_dim is None:
            # mask along the second dimension by default
            if len(nodes.shape) == 2:
                pass
            elif len(nodes.shape) == 3:
                node_flags = node_flags.unsqueeze(-1)  # [B, N, 1]
            else:
                raise NotImplementedError
        else:
            assert nodes.size(along_dim) == len(node_flags)
            shape_ls = list(node_flags.shape)
            assert len(shape_ls) == 2
            for i, dim in enumerate(nodes.shape):
                if i == 0:
                    pass
                else:
                    if i < along_dim:
                        shape_ls.insert(1, 1)  # insert 1 at the second dim
                    elif i == along_dim:
                        assert shape_ls[i] == dim  # check the length consistency
                    elif i > along_dim:
                        shape_ls.insert(len(shape_ls), 1)  # insert 1 at the end
            node_flags = node_flags.view(*shape_ls)  # [B, *, N, *]

        if in_place:
            nodes.masked_fill_(torch.logical_not(node_flags), value)
        else:
            nodes = nodes.masked_fill(torch.logical_not(node_flags), value)
    else:
        raise NotImplementedError
    return nodes


def mask_incs(incs, node_flags, edge_flags, value=0.0, in_place=True):
    """
    Masking incidence matrix according to node and edge flags.
    @param incs:        B x N x M or B x N x M x F
    @param node_flags:  B x N
    @param edge_flags:  B x M
    @param value:       scalar
    @param in_place:    flag of in place operation
    @return incs:       B x N x M or B x N x M x F
    """
    b, n, m = incs.shape[:3]
    assert node_flags.size(1) == n and edge_flags.size(1) == m

    node_flags = node_flags.view(b, n, 1)
    edge_flags = edge_flags.view(b, 1, m)

    if len(incs.shape) == 4:
        node_flags = node_flags.unsqueeze(-1)
        edge_flags = edge_flags.unsqueeze(-1)

    if in_place:
        incs.masked_fill_(torch.logical_not(node_flags), value)
        incs.masked_fill_(torch.logical_not(edge_flags), value)
    else:
        incs = incs.masked_fill(torch.logical_not(node_flags), value)
        incs = incs.masked_fill(torch.logical_not(edge_flags), value)
    return incs


def check_adjs_symmetry(adjs):
    """
    Check if adjs is symmetric along the last two dimensions.
    """
    tr_adjs = adjs.transpose(-1, -2)
    assert (adjs - tr_adjs).abs().sum() < 1e-2


def pad_adjs(ori_adj, node_number):
    """
    Pad the adjacency matrices with zeros.
    """
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    # a = np.logical_or(a, np.identity(node_number))
    return a


def get_laplacian(adjs, normalized=True, exponential=False):
    """
    Compute laplacian matrix for the given adjacency matrix.
    @param adjs:                    [B, N, N] adjacency matrix.
    @param normalized:              flag of normalization.
    @param exponential:             flag of exponential operation.
    @return lap_mat:                [B, N, N] laplacian matrix.
    """
    if exponential:
        temp = 1.0
        adjs = torch.exp(adjs / temp)
    deg_mat = adjs.sum(dim=-1)  # [B, N]
    deg_mat = torch.diag_embed(deg_mat)  # [B, N, N]
    lap_mat = deg_mat - adjs  # [B, N, N]
    if normalized:
        deg_mat_inv = deg_mat.pow(-0.5)
        deg_mat_inv.masked_fill_(deg_mat_inv == float('inf'), 0)
        lap_mat = torch.einsum('b m n, b n j, b j k -> b m k', deg_mat_inv, lap_mat, deg_mat_inv)
    return lap_mat


def augment_adjs(adjs, node_flags, adjs_power, power=1):
    """
    Initialize edge features by augmenting the raw adjacency matrix.
    @param adjs:                    [B, N, N]
    @param node_flags:              [B, N]
    @param adjs_power:              [B, N, N]
    @param power:                   int
    """
    laplacian_adjs = get_laplacian(adjs, normalized=True, exponential=True)  # [B, N, N]
    adjs = torch.stack([adjs, 1. - adjs, laplacian_adjs], dim=1)  # [B, 3, N, N]
    if power == 1:
        pass
    else:
        adjs_power = torch.stack([torch.matrix_power(adjs_power, p) for p in torch.arange(2, power+1)], dim=1)  # [B, P, N, N]
        adjs = torch.cat([adjs, adjs_power], dim=1)  # [B, 3+P, N, N]
    adjs = mask_adjs(adjs, node_flags)  # non-active nodes have zero-value at their adj matrices
    return adjs


def get_sym_normal_noise(like_this_tensor):
    """
    Generate symmetric normal distribution noise tensor.
    """
    noise = torch.randn_like(like_this_tensor).triu(1)
    noise_s = noise + noise.transpose(-1, -2)
    return noise_s


def add_sym_normal_noise(in_tensors, scales, sigmas, node_flags=None, non_symmetric=False):
    """
    Add (usually symmetric) Gaussian noise to the input tensors.
    @param in_tensors: [B, *]
    @param scales: [B]
    @param sigmas: [B]
    @param node_flags: [B, N] or None
    @param non_symmetric: inject non-symmetric noise forcefully
    @return out_tensors: [B, *]
    """
    num_dims = len(in_tensors.shape)

    scales = scales.view(scales.shape + (1, ) * (num_dims - 1))  # [B, *]
    sigmas = sigmas.view(sigmas.shape + (1, ) * (num_dims - 1))  # [B, *]
    in_tensors = in_tensors * scales
    if non_symmetric:
        # node vector representation
        noise_s = torch.randn_like(in_tensors) * sigmas
    else:
        # adjacency matrix representation
        noise_s = get_sym_normal_noise(in_tensors) * sigmas
    out_tensors = in_tensors + noise_s
    if node_flags is not None:
        # [B, C, N, N] shape or [B, N, N]
        if len(in_tensors.shape) == 4 or (len(in_tensors.shape) == 3 and in_tensors.size(-1) == in_tensors.size(-2)):
            out_tensors = mask_adjs(out_tensors, node_flags)
            noise_s = mask_adjs(noise_s, node_flags)
        else:
            out_tensors = mask_nodes(out_tensors, node_flags)
            noise_s = mask_nodes(noise_s, node_flags)
    return out_tensors, noise_s


def eigen_to_matrix(eig_vecs, eig_vals, node_flags, deg_vec=None, return_lap=False):
    """
    Convert the eigen-decomposition terms back to the adjacency matrix.
    @param eig_vecs: [B, N, K]
    @param eig_vals: [B, K] or [B, K, 1]
    @param node_flags: [B, N] or None
    @param return_lap: bool
    @param deg_vec: [B, K] or [B, K, 1] or None
    """
    if len(eig_vals.shape) == 2:
        pass
    elif len(eig_vals.shape) == 3:
        assert eig_vals.size(-1) == 1
        eig_vals = eig_vals.squeeze(-1)
    laplacian = torch.einsum('b n k, b k, b m k -> b n m', eig_vecs, eig_vals, eig_vecs)  # [B, N, N]
    if deg_vec is not None:
        # to get the un-normalized Laplacian
        assert deg_vec.min() >= 0.0
        deg_vec = deg_vec.squeeze(-1) if len(deg_vec.shape) == 3 else deg_vec  # [B, N]
        deg_mat = torch.diag_embed(deg_vec)  # [B, N, N]
        deg_mat_half = deg_mat.pow(0.5)
        laplacian = torch.einsum('b a b, b b c, b c d -> b a d', deg_mat_half, laplacian, deg_mat_half)  # [B, N, N]
    if return_lap:
        if node_flags is None:
            return laplacian
        else:
            return mask_adjs(laplacian, node_flags)
    else:
        adjs = torch.diag_embed(torch.diagonal(laplacian, dim1=-1, dim2=-2)) - laplacian  # [B, N, N]
        if node_flags is None:
            return adjs
        else:
            return mask_adjs(adjs, node_flags)


def inc_to_adjs(incs, node_flags, deg_vec=None, return_lap=False):
    """
    Convert the incidence matrix back to the adjacency matrix.
    @param incs: [B, N, M]
    @param node_flags: [B, N]
    @param return_lap: bool
    @param deg_vec: [B, N]
    """
    laplacian = torch.matmul(incs, incs.transpose(-1, -2))  # [B, N, N], L = D - A
    if return_lap:
        if node_flags is None:
            return laplacian
        else:
            return mask_adjs(laplacian, node_flags)
    else:
        if deg_vec is None:
            deg_mat = torch.diag_embed(torch.diagonal(laplacian, dim1=-1, dim2=-2))
        else:
            deg_mat = torch.diag_embed(deg_vec)
        # A = D - L
        adjs = deg_mat - laplacian  # [B, N, N]
        if node_flags is None:
            return adjs
        else:
            return mask_adjs(adjs, node_flags)
