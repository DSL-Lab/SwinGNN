import os
import pdb
import time
import numpy as np
import torch

import sys
PROJECT_DIR = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.insert(0, PROJECT_DIR)

from utils.graph_utils import mask_adjs, mask_nodes


def attribute_converter(in_attr, attr_flags, in_encoding, out_encoding, num_attr_type,
                        flag_nodes=False, flag_adjs=False, flag_in_ddpm_range=True, flag_out_ddpm_range=True,
                        flag_clamp_int=False):
    """
    Convert node and adj data to different types of encoding
    @param in_attr: [B, N] or or [B, N, C] or [B, N, N] or [B, C, N, N], the range is [0, 1, 2, ..., num_attr_type - 1]
    @param attr_flags: [B, N] or [B, N, N], 1 for valid, 0 for invalid
    @param in_encoding: str, 'int', 'ddpm', 'bits' or 'one_hot'
    @param out_encoding: str, 'int', 'ddpm', 'bits' or 'one_hot'
    @param num_attr_type: int
    @param flag_nodes: bool, whether to check node attributes
    @param flag_adjs: bool, whether to check adjacency attributes
    @param flag_in_ddpm_range: bool, whether the input is in DDPM range [-1, 1] for binary encoding
    @param flag_out_ddpm_range: bool, whether the output is in DDPM range [-1, 1] for binary encoding
    @param flag_clamp_int: bool, whether to clamp the converted integer to a reasonable range
    """

    """input sanity check"""
    attribute_sanity_check(in_attr, attr_flags, in_encoding, flag_nodes=flag_nodes, flag_adjs=flag_adjs)

    """use int encoding as intermediate state"""
    if in_encoding == 'int':
        int_attr = in_attr
    elif in_encoding == 'ddpm':
        int_attr = attribute_ddpm_to_int(in_attr, attr_flags, num_attr_type)
    elif in_encoding == 'bits':
        int_attr = attribute_bits_to_int(in_attr, attr_flags, num_attr_type, flag_in_ddpm_range, flag_clamp_int)
    elif in_encoding == 'one_hot':
        int_attr = attribute_one_hot_to_int(in_attr, attr_flags, num_attr_type, flag_in_ddpm_range)
    else:
        raise ValueError("encoding should be 'int', 'ddpm', 'bits' or 'one_hot'")

    attribute_sanity_check(int_attr, attr_flags, 'int', flag_nodes=flag_nodes, flag_adjs=flag_adjs)

    """convert the int encoding to the desired encoding"""
    if out_encoding == 'int':
        out_attr = int_attr
    elif out_encoding == 'ddpm':
        out_attr = attribute_int_to_ddpm(int_attr, attr_flags, num_attr_type)
    elif out_encoding == 'bits':
        out_attr = attribute_int_to_bits(int_attr, attr_flags, num_attr_type, flag_out_ddpm_range)
    elif out_encoding == 'one_hot':
        out_attr = attribute_int_to_one_hot(int_attr, attr_flags, num_attr_type, flag_out_ddpm_range)
    else:
        raise ValueError("encoding should be 'int', 'ddpm', 'bits' or 'one_hot'")
    # print("final sanity check, in_encoding={}, out_encoding={}".format(in_encoding, out_encoding))
    attribute_sanity_check(out_attr, attr_flags, out_encoding, flag_nodes=flag_nodes, flag_adjs=flag_adjs)

    return out_attr


def get_mask_func(in_attr, flag_nodes=False, flag_adjs=False):
    """
    Get the proper mask function.
    """
    if not flag_adjs and not flag_nodes:
        # infer the node/adj type of in_attr from its shape
        # this only works for [B, N] or [B, N, N] tensors, one scalar for one entry
        if len(in_attr.shape) == 3:
            _mask_func = mask_adjs
        elif len(in_attr.shape) == 2:
            _mask_func = mask_nodes
        else:
            raise ValueError("in_attr shape should be [B, N] or [B, N, N]")
    else:
        assert (flag_adjs + flag_nodes) == 1, "flag_nodes and flag_adjs cannot be both True or False"
        if flag_adjs:
            _mask_func = mask_adjs
        elif flag_nodes:
            _mask_func = mask_nodes
        else:
            raise NotImplementedError
    return _mask_func


def attribute_sanity_check(in_attr: torch.Tensor, attr_flags: torch.Tensor, encoding: str,
                           flag_nodes: bool = False, flag_adjs: bool = False, flag_in_ddpm_range: bool = True):
    """
    Sanity check for attribute code.
    @param in_attr: [B, N] or [B, N, C] or [B, N, N] or [B, C, N, N], the range is [0, 1, 2, ..., num_attr_type - 1]
    @param attr_flags: [B, N] or [B, N, N], 1 for valid, 0 for invalid
    @param encoding: str, 'int', 'ddpm', 'bits' or 'one_hot'
    @param flag_nodes: bool, whether to check node attributes
    @param flag_adjs: bool, whether to check adjacency attributes
    @param flag_in_ddpm_range: bool, whether the input is in DDPM range [-1, 1] for binary digits encoding
    """
    assert (flag_adjs + flag_nodes) == 1, "flag_nodes and flag_adjs cannot be both True or False"

    if len(attr_flags.shape) == 3:
        assert flag_adjs, "attr_flags shape is [B, N, N], flag_adjs should be True"

    if encoding == 'int':
        assert (in_attr == in_attr.long()).all()  # int
    elif encoding == 'ddpm':
        assert (-1 <= in_attr).all() and (in_attr <= 1).all()  # [-1, 1] float
    elif encoding in ['bits', 'one_hot']:
        if flag_in_ddpm_range:
            target_tensor = torch.tensor([-1, 0, 1], device=in_attr.device)
        else:
            target_tensor = torch.tensor([0, 1], device=in_attr.device)
        _unique_attr = in_attr.unique(sorted=True)
        if len(_unique_attr) == 2:
            pdb.set_trace()
            assert (_unique_attr == target_tensor[:2]).all()  # int
        else:
            assert (_unique_attr == target_tensor).all()

    else:
        raise ValueError("encoding should be 'int', 'ddpm', 'bits' or 'one_hot'")


def attribute_ddpm_to_int(in_attr, attr_flags, num_attr_type, flag_quantization=True):
    """
    Convert attribute data from DDPM range [-1, 1] to integer node/edge type.
    @param in_attr: [B, N] or [B, N, N], the range is [-1, ..., 1]
    @param attr_flags: [B, N] or [B, N, N], 1 for valid, 0 for invalid
    @param num_attr_type: int, number of attribute types
    @param flag_quantization: bool, whether to forcefully quantize the attribute to integer type
    """

    def _get_intervals(num_type):
        assert num_type >= 2
        min_ls, max_ls = [], []
        interval_length = 2.0 / (num_type - 1)
        for i in range(num_type):
            center = -1.0 + i * interval_length
            if i == 0:
                assert center == -1.0
                min_ls.append(-float('inf'))
                max_ls.append(center + interval_length * 0.5)
            elif i < num_type - 1:
                min_ls.append(center - 0.5 * interval_length)
                max_ls.append(center + 0.5 * interval_length)
            elif i == num_type - 1:
                assert center == 1.0
                min_ls.append(center - interval_length * 0.5)
                max_ls.append(float('inf'))
        return min_ls, max_ls

    def _assign_integers(in_tensor, num_type):
        min_ls, max_ls = _get_intervals(num_type)
        out_tensor = torch.full_like(in_tensor, -1.0)
        for i in range(num_type):
            this_min, this_max = min_ls[i], max_ls[i]
            flag_min_max = torch.logical_and(in_tensor > this_min, in_tensor <= this_max)
            out_tensor[flag_min_max] = i
        return out_tensor

    # flag_nodes = len(in_attr.shape) == 2
    # flag_adjs = len(in_attr.shape) == 3
    # attribute_sanity_check(in_attr, attr_flags, 'ddpm', flag_adjs=flag_adjs, flag_nodes=flag_nodes)

    if flag_quantization:
        out_attr = _assign_integers(in_attr, num_attr_type)  # [B, N] or [B, N, N]
    else:
        # i = (y+1) * (k-1) / 2.0, with k being the number of types
        # ideally, index i should be an integer in [1, 2, ..., n]
        out_attr = (in_attr + 1) * (num_attr_type - 1) / 2.0  # [B, N, N]

    _mask_func = get_mask_func(in_attr, flag_nodes=False, flag_adjs=False)
    out_attr = _mask_func(out_attr, attr_flags)

    # attribute_sanity_check(out_attr, attr_flags, 'int', flag_adjs=flag_adjs, flag_nodes=flag_nodes)

    return out_attr


def attribute_bits_to_int(in_attr, attr_flags, num_attr_type, flag_in_ddpm_range=True, flag_clamp_int=False):
    """
    Convert attribute data from DDPM range [-1, 1] to integer node/edge type.
    @param in_attr: [B, N, C] or [B, C, N, N], the entries are -1/1 or 0/1
    @param attr_flags: [B, N] or [B, N, N], 1 for valid, 0 for invalid
    @param num_attr_type: int, number of attribute types
    @param flag_in_ddpm_range: bool, whether the input attribute is in DDPM range [-1, 1]
    @param flag_clamp_int: bool, whether to clamp the integer attribute to [0, num_attr_type-1]
    """

    flag_nodes = len(in_attr.shape) == 3
    flag_adjs = len(in_attr.shape) == 4
    # attribute_sanity_check(in_attr, attr_flags, 'bits', flag_adjs=flag_adjs, flag_nodes=flag_nodes,
    #                        flag_in_ddpm_range=flag_in_ddpm_range)
    _mask_func = get_mask_func(in_attr, flag_nodes=flag_nodes, flag_adjs=flag_adjs)
    if flag_in_ddpm_range:
        in_attr = (in_attr + 1.0) / 2.0  # [B, N, C] or [B, C, N, N], entries are 0/1
        in_attr = _mask_func(in_attr, attr_flags)
        # assert (in_attr.unique(sorted=True) == torch.tensor([0, 1], device=in_attr.device)).all()  # [0, 1] int

    in_attr = in_attr.permute(0, 2, 3, 1) if flag_adjs else in_attr  # [B, N, C] or [B, N, N, C]
    out_attr = bin2dec(in_attr, num_bits=in_attr.shape[-1])  # [B, N] or [B, N, N]

    if flag_clamp_int:
        out_attr = torch.clamp(out_attr, min=0, max=num_attr_type - 1)

    out_attr = _mask_func(out_attr, attr_flags)
    # attribute_sanity_check(out_attr, attr_flags, 'int', flag_adjs=flag_adjs, flag_nodes=flag_nodes)

    assert (out_attr <= num_attr_type - 1).all()
    return out_attr


def attribute_one_hot_to_int(in_attr, attr_flags, num_attr_type, flag_in_ddpm_range=True):
    """
    Convert attribute data from DDPM range [-1, 1] to integer node/edge type.
    @param in_attr: [B, N, C] or [B, C, N, N], the entries are -1/1 or 0/1
    @param attr_flags: [B, N] or [B, N, N], 1 for valid, 0 for invalid
    @param num_attr_type: int, number of attribute types
    @param flag_in_ddpm_range: bool, whether the input attribute is in DDPM range [-1, 1]
    """
    flag_nodes = len(in_attr.shape) == 3
    flag_adjs = len(in_attr.shape) == 4
    # attribute_sanity_check(in_attr, attr_flags, 'one_hot', flag_adjs=flag_adjs, flag_nodes=flag_nodes,
    #                        flag_in_ddpm_range=flag_in_ddpm_range)
    _mask_func = get_mask_func(in_attr, flag_nodes=flag_nodes, flag_adjs=flag_adjs)
    if flag_in_ddpm_range:
        in_attr = (in_attr + 1.0) / 2.0  # [B, N, C] or [B, C, N, N], entries are 0/1
        in_attr = _mask_func(in_attr, attr_flags)
        # assert (in_attr.unique(sorted=True) == torch.tensor([0, 1], device=in_attr.device)).all()  # [0, 1] int

    in_attr = in_attr.permute(0, 2, 3, 1) if flag_adjs else in_attr  # [B, N, C] or [B, N, N, C]
    out_attr = in_attr.argmax(dim=-1)  # [B, N] or [B, N, N]

    out_attr = _mask_func(out_attr, attr_flags)
    # attribute_sanity_check(out_attr, attr_flags, 'int', flag_adjs=flag_adjs, flag_nodes=flag_nodes)

    assert (out_attr <= num_attr_type - 1).all()
    return out_attr


def attribute_int_to_ddpm(in_attr, attr_flags, num_attr_type):
    """
    Convert node and adj data to DDPM range [-1, 1]
    @param in_attr: [B, N] or [B, N, N], the range is [0, 1, 2, ..., num_attr_type - 1]
    @param attr_flags: [B, N] or [B, N, N], 1 for valid, 0 for invalid
    @param num_attr_type: int
    """
    assert (0 <= in_attr).all() and (in_attr <= num_attr_type - 1).all()  # k types, in range of [0, 1, 2, ..., k-1]
    _mask_func = get_mask_func(in_attr, flag_nodes=False, flag_adjs=False)

    # y = 2 * i / (k-1) - 1, with i in [0, 1, 2, ..., k-1]
    out_attr = 2 * in_attr / (num_attr_type - 1.0) - 1.0  # [B, N] or [B, N, N]
    out_attr = _mask_func(out_attr, attr_flags)  # [B, N] or [B, N, N]

    return out_attr


def attribute_int_to_bits(in_attr, attr_flags, num_attr_type, flag_ddpm_range=True):
    """
    Convert node and adj data to bits
    @param in_attr: [B, N] or [B, N, N], the range is [0, 1, 2, ..., num_attr_type - 1]
    @param attr_flags: [B, N]
    @param num_attr_type: int
    @param flag_ddpm_range: bool, whether to convert to DDPM range [-1, 1]
    """
    assert (0 <= in_attr).all() and (in_attr <= num_attr_type - 1).all()  # k types, in range of [0, 1, 2, ..., k-1]
    _mask_func = get_mask_func(in_attr, flag_nodes=False, flag_adjs=False)

    num_bits = np.ceil(np.log2(num_attr_type)).astype(int)  # int

    out_attr = dec2bin(in_attr.long(), num_bits=num_bits)  # [B, N, C] <- [B, N] or [B, N, N, C] <- [B, N, N]

    if len(out_attr.shape) == 4:
        out_attr = out_attr.permute(0, 3, 1, 2)  # [B, C, N, N] <- [B, N, N, C], binary entries

    if flag_ddpm_range:
        out_attr = 2 * out_attr - 1  # [B, C, N] or [B, C, N, N], -1/1 entries

    out_attr = _mask_func(out_attr, attr_flags)  # [B, N, C] or [B, C, N, N], -1/1 entries

    return out_attr


def attribute_int_to_one_hot(in_attr, attr_flags, num_attr_type, flag_ddpm_range=True):
    """
    Convert node and adj data to one-hot encoding
    @param in_attr: [B, N] or [B, N, N], the range is [0, 1, 2, ..., num_attr_type - 1]
    @param attr_flags: [B, N]
    @param num_attr_type: int
    @param flag_ddpm_range: bool, whether to convert to DDPM range [-1, 1]
    """
    assert (0 <= in_attr).all() and (in_attr <= num_attr_type - 1).all()  # k types, in range of [0, 1, 2, ..., k-1]
    _mask_func = get_mask_func(in_attr, flag_nodes=False, flag_adjs=False)

    # [B, N, C] or [B, N, N, C]
    out_attr = torch.nn.functional.one_hot(in_attr.long(), num_classes=num_attr_type).float()
    if len(out_attr.shape) == 4:
        out_attr = out_attr.permute(0, 3, 1, 2)  # [B, C, N, N] <- [B, N, N, C]

    if flag_ddpm_range:
        out_attr = 2 * out_attr - 1  # [B, C, N] or [B, C, N, N], -1/1 entries

    out_attr = _mask_func(out_attr, attr_flags)  # [B, N, C] or [B, C, N, N], -1/1 entries

    return out_attr


def dec2bin(dec_tensor, num_bits):
    """
    Convert decimal tensor to binary tensor.
    code reference: https://stackoverflow.com/a/63630138/8683446
    @param: dec_tensor: [B, N]
    @param: num_bits: number of bits to represent the decimal number
    """
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(num_bits - 1, -1, -1).to(dec_tensor.device, dec_tensor.dtype)
    return dec_tensor.clone().unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(bin_tensor, num_bits):
    """
    Convert binary tensor to decimal tensor.
    code reference: https://stackoverflow.com/a/63630138/8683446
    @param bin_tensor: [B, N, bits]
    @param num_bits: number of bits to represent the decimal number
    """

    mask = 2 ** torch.arange(num_bits - 1, -1, -1).to(bin_tensor.device, bin_tensor.dtype)
    return torch.sum(mask * bin_tensor, -1)


def reshape_node_attr_vec_to_mat(node_attr_vec, node_flags_vec, matrix_size):
    """
    Reshape node attribute to matrix form, regardless of the encoding type.
    @param node_attr_vec: [B, N] or [B, N, C]
    @param node_flags_vec: [B, N]
    @param matrix_size: int, the size of the matrix
    @return: node_attr_mat: [B, M, M] or [B, C, M, M], node_flags_mat: [B, M, M]
    """
    _max_num_nodes = node_flags_vec.sum(dim=-1).max().item()  # int
    assert _max_num_nodes <= matrix_size ** 2, f"max_num_nodes={_max_num_nodes} > matrix_size^2={matrix_size ** 2}"

    b, n = node_attr_vec.shape[:2]
    m = matrix_size

    if len(node_attr_vec.shape) == 2:
        # [B, N] -> [B, M, M]
        # node_attr_mat = torch.zeros([b, m, m], dtype=node_attr.dtype, device=node_attr.device)
        node_attr_pad = torch.nn.functional.pad(node_attr_vec, (0, m ** 2 - n), value=0)  # [B, M^2] <- [B, N]
        node_attr_mat = node_attr_pad.view(b, m, m)  # [B, M, M] <- [B, M^2]
        node_flags_pad = torch.nn.functional.pad(node_flags_vec, (0, m ** 2 - n), value=0)  # [B, M^2] <- [B, N]
        node_flags_mat = node_flags_pad.view(b, m, m)  # [B, M, M] <- [B, M^2]

        assert (node_attr_mat[torch.logical_not(node_flags_mat)] == 0.0).all()
    elif len(node_attr_vec.shape) == 3:
        # [B, N, C] -> [B, C, M, M]
        node_attr_pad = torch.nn.functional.pad(node_attr_vec, (0, 0, 0, m ** 2 - n), value=0)  # [B, M^2, C] <- [B, N, C]
        node_attr_mat = node_attr_pad.view(b, m, m, -1).permute(0, 3, 1, 2)  # [B, C, M, M] <- [B, M^2, C]
        node_flags_pad = torch.nn.functional.pad(node_flags_vec, (0, m ** 2 - n), value=0)  # [B, M^2] <- [B, N]
        node_flags_mat = node_flags_pad.view(b, m, m)  # [B, M, M] <- [B, M^2]

        node_flags_mat_ = node_flags_mat.unsqueeze(1).repeat(1, node_attr_mat.shape[1], 1, 1)  # [B, C, M, M]
        assert (node_attr_mat[torch.logical_not(node_flags_mat_)] == 0.0).all()
    else:
        raise ValueError(f"node_attr.shape={node_attr_vec.shape} is not supported.")

    _mask_func = get_mask_func(node_attr_vec, flag_nodes=False, flag_adjs=True)
    node_attr_mat = _mask_func(node_attr_mat, node_flags_mat)  # [B, M, M] or [B, C, M, M]
    return node_attr_mat, node_flags_mat


def reshape_node_attr_mat_to_vec(node_attr_mat, node_flags_mat, vector_size):
    """
    Reshape node attribute to vector form, regardless of the encoding type.
    @param node_attr_mat: [B, M, M] or [B, C, M, M]
    @param node_flags_mat: [B, M, M]
    @param vector_size: int, the size of the matrix
    @return node_attr_vec: [B, N] or [B, N, C], node_flags_vec: [B, N]
    """
    _max_num_nodes = node_flags_mat.sum(dim=[-1, -2]).max().item()  # int
    assert _max_num_nodes <= vector_size, f"max_num_nodes={_max_num_nodes} > matrix_size={vector_size}"

    b = node_attr_mat.size(0)
    m = node_attr_mat.size(-1)
    n = vector_size

    pad_len, slice_len = None, None
    if vector_size >= m ** 2:
        pad_len = vector_size - m ** 2
    else:
        # slice_len = _max_num_nodes
        slice_len = vector_size

    def _pad_or_slice_tensor(_in_tensor):
        """
        Pad or slice the tensor to the desired size.
        """
        if pad_len is not None:
            _out_tensor = torch.nn.functional.pad(_in_tensor, (0, pad_len), value=0)  # [B, N] or [B, C, N]
        else:
            if len(_in_tensor.shape) == 2:
                _out_tensor = _in_tensor[:, :slice_len]  # [B, N]
            else:
                _out_tensor = _in_tensor[:, :, :slice_len]  # [B, C, N]
        return _out_tensor

    if len(node_attr_mat.shape) == 3:
        # [B, M, M] -> [B, N]
        node_attr_flat = node_attr_mat.view(b, -1)  # [B, M^2]
        node_attr_vec = _pad_or_slice_tensor(node_attr_flat)  # [B, N] <- [B, M^2]

        node_flags_flat = node_flags_mat.view(b, -1)  # [B, M^2]
        node_flags_vec = _pad_or_slice_tensor(node_flags_flat)  # [B, N] <- [B, M^2]

        assert (node_attr_vec[torch.logical_not(node_flags_vec)] == 0.0).all()
    elif len(node_attr_mat.shape) == 4:
        # [B, C, M, M] -> [B, N, C]
        node_attr_flat = node_attr_mat.view(b, -1, m * m)  # [B, C, M^2]
        node_attr_vec = _pad_or_slice_tensor(node_attr_flat)  # [B, C, N] <- [B, C, M^2]
        node_attr_vec = node_attr_vec.permute(0, 2, 1)  # [B, N, C] <- [B, C, N]

        node_flags_flat = node_flags_mat.view(b, -1)  # [B, M^2]
        node_flags_vec = _pad_or_slice_tensor(node_flags_flat)  # [B, N] <- [B, M^2]

        assert (node_attr_vec[torch.logical_not(node_flags_vec)] == 0.0).all()
    else:
        raise ValueError(f"node_attr.shape={node_attr_mat.shape} is not supported.")

    _mask_func = get_mask_func(node_attr_mat, flag_nodes=True, flag_adjs=False)
    node_attr_vec = _mask_func(node_attr_vec, node_flags_vec)  # [B, N]
    return node_attr_vec, node_flags_vec


def unit_test():
    batch_size = 256
    num_nodes = 64
    num_attr_type = 51

    for flag_use_adjs in [True, False]:
        if flag_use_adjs:
            raw_attr = torch.randint(low=0, high=num_attr_type - 1, size=(batch_size, num_nodes, num_nodes))  # [B, N, N]
        else:
            raw_attr = torch.randint(low=0, high=num_attr_type - 1, size=(batch_size, num_nodes))  # [B, N]
        _mask_func = get_mask_func(raw_attr, flag_nodes=False, flag_adjs=False)

        attr_flags = torch.ones(raw_attr.shape[:2], dtype=torch.bool)  # [B, N]
        for i in range(batch_size):
            _effective_num_nodes = torch.randint(low=0, high=num_nodes, size=(1,)).item()
            attr_flags[i, _effective_num_nodes:] = False

        raw_attr = _mask_func(raw_attr, attr_flags)  # [B, N, N]
        raw_attr = raw_attr.cuda()
        attr_flags = attr_flags.cuda()
        timer_ls = []
        for in_encoding in ['int', 'ddpm', 'one_hot', 'bits']:
            for out_encoding in ['int', 'ddpm', 'one_hot', 'bits']:
                print("Sanity check: in_encoding: {}, out_encoding: {}".format(in_encoding, out_encoding))
                time_start = time.time()

                # raw to in
                in_attr = attribute_converter(raw_attr, attr_flags, 'int', in_encoding, num_attr_type,
                                              flag_nodes=False, flag_adjs=True,
                                              flag_in_ddpm_range=False, flag_out_ddpm_range=True)

                # in to out
                out_attr = attribute_converter(in_attr, attr_flags, in_encoding, out_encoding, num_attr_type,
                                               flag_nodes=False, flag_adjs=True,
                                               flag_in_ddpm_range=True, flag_out_ddpm_range=True)

                # out back to in
                _in_attr = attribute_converter(out_attr, attr_flags, out_encoding, in_encoding, num_attr_type,
                                               flag_nodes=False, flag_adjs=True,
                                               flag_in_ddpm_range=True, flag_out_ddpm_range=True)

                assert (in_attr == _in_attr).all()
                time_end = time.time() - time_start
                # print("Time elapsed: {:.3f} s".format(time_end))
                timer_ls.append(time_end)

        print("flag_use_adjs {} Average time elapsed: {:.3f} s".format(flag_use_adjs, np.mean(timer_ls)))


if __name__ == "__main__":
    unit_test()

