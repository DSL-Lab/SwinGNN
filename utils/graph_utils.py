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

