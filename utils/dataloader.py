import logging
import os
import pdb
import time
import pickle
import json
import numpy as np
import networkx as nx
import torch
from torch import distributed as dist
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader

from utils.attribute_code import attribute_converter, reshape_node_attr_vec_to_mat
from utils.graph_utils import mask_adjs, mask_nodes, pad_adjs
from utils.visual_utils import plot_graphs_list
from utils.mol_utils import load_smiles, canonicalize_smiles, mols_to_nx, smiles_to_mols


def load_data(config, dist_helper, eval_mode=False):
    """
    Setup training/validation/testing dataloader.
    """

    batch_size = config.test.batch_size if eval_mode else config.train.batch_size

    def _build_dataloader(in_dataset):
        if dist_helper.is_ddp:
            sampler = DistributedSampler(in_dataset)
            batch_size_per_gpu = max(1, batch_size // dist.get_world_size())
            data_loader = DataLoader(in_dataset, sampler=sampler, batch_size=batch_size_per_gpu,
                                     pin_memory=True, num_workers=min(6, os.cpu_count()))
        else:
            data_loader = DataLoader(in_dataset, batch_size=batch_size, shuffle=True,
                                     pin_memory=True, num_workers=min(6, os.cpu_count()))
        return data_loader

    if config.flag_mol:
        # molecule data
        train_dataset, test_dataset = load_dataset_mol(config)

        train_smiles, test_smiles = load_smiles(config.dataset.name.upper(), config.dataset.subset)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)
        test_graph_list = mols_to_nx(smiles_to_mols(test_smiles))
    else:
        # general graph data
        train_dataset, test_dataset, train_graph_ls, test_graph_ls = load_dataset_general(config)

    train_dl = _build_dataloader(train_dataset)
    test_dl = _build_dataloader(test_dataset)

    logging.info("Training / testing set size: {:d} / {:d}".format(len(train_dataset), len(test_dataset)))
    logging.info("Training / testing dataloader batch size: {:d} / {:d}".format(
        train_dl.batch_size, test_dl.batch_size))

    # attach additional information to the dataloader
    if config.flag_mol:
        test_dl.train_smiles = train_smiles
        test_dl.test_smiles = test_smiles
        test_dl.test_graph_list = test_graph_list
    else:
        test_dl.train_graph_ls = train_graph_ls
        test_dl.test_graph_ls = test_graph_ls

    return train_dl, test_dl


def load_dataset_mol(config):
    """
    Setup training/validation/testing dataloader for molecule datasets.
    code reference: https://github.com/harryjo97/GDSS/blob/master/utils/data_loader_mol.py
    """
    logging.info("Loading molecule dataset...")
    time_start = time.time()
    filepath = os.path.join('data', '{:s}_kekulized.npz'.format(config.dataset.name))
    data = np.load(filepath)
    results = []
    i = 0
    while True:
        key = f'arr_{i}'
        if key in data.keys():
            results.append(data[key])
            i += 1
        else:
            break

    graphs = list(map(lambda x, a: (x, a), results[0], results[1]))

    with open(os.path.join('dataset', 'valid_idx_{:s}.json'.format(config.dataset.name))) as f:
        test_idx = json.load(f)

    if config.dataset.name == 'qm9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    train_idx = [i for i in range(len(graphs)) if i not in test_idx]

    if config.dataset.subset is not None:
        # DEBUG mode, select a subset of the dataset
        assert isinstance(config.dataset.subset, int)
        set_size = config.dataset.subset
        graphs = graphs[:set_size]
        train_idx = list(range(set_size))
        test_idx = train_idx
        logging.info("Molecule dataset subset selection: the first {:d} data points are used".format(set_size))

    n_max = int(config.dataset.max_node_num)
    assert config.dataset.name.lower() in ['qm9', 'zinc250k']
    adj_ls, x_ls, node_flags_ls = [], [], []

    # TODO: remove the for loop and do everything with tensor slicing
    for i, g in enumerate(graphs):
        node, adj = g  # [N] + [4, N, N], N=4 for QM9, N=38 for ZINC250K

        def transform(adj):
            # GDSS transformation: [4, N, N] matrix input -> [N, N] matrix output
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0).astype(np.float32)
            adj = torch.tensor(adj.argmax(axis=0))  # [4, N, N] (the last place is for virtual edges) -> [N, N]
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            return adj

        adj = transform(adj)  # [N, N]
        assert adj.equal(adj.t()), "adjacency matrix is not symmetric"

        if config.dataset.name == 'qm9':
            node = torch.from_numpy(node).to(torch.float32)  # [N = 9]
            node_flags = node > 0  # [9]
            node -= 6  # actual value start from 0
            node[torch.logical_not(node_flags)] = 0.0  # 0 for padding
        elif config.dataset.name == 'zinc250k':
            zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]  # [10]
            node_flags = torch.from_numpy(node > 0)  # [N = 38]
            for el in range(node.shape[0]):
                idx = zinc250k_atomic_num_list.index(node[el])
                node[el] = idx
            node = torch.from_numpy(node).to(torch.float32)
            node[torch.logical_not(node_flags)] = 0.0
        else:
            raise NotImplementedError
        adj_ls.append(adj)  # [N, N]
        x_ls.append(node)  # [N]
        node_flags_ls.append(node_flags)  # [N]

    # batch process
    node = torch.stack(x_ls, dim=0)  # [B, N]
    adj = torch.stack(adj_ls, dim=0)  # [B, N, N]
    node_flags = torch.stack(node_flags_ls, dim=0)  # [B, N]

    len_node = node.shape[1]
    adj_pad = torch.nn.functional.pad(adj, (0, n_max - len_node, 0, n_max - len_node), "constant", 0.0)  # [B, N, N]
    node_pad = torch.nn.functional.pad(node, (0, n_max - len_node), "constant", 0.0)  # [B, N]
    node_flags_pad = torch.nn.functional.pad(node_flags, (0, n_max - len_node), "constant", 0.0)  # [B, N]

    # at this stage, node and adjs entries are integers ranging in [0, 1, 2, ...]
    # zero-value could mean either padding or null-type, we must keep the node_flags!

    if config.dataset.name == 'qm9':
        num_node_type = 4
        num_adj_type = 4
    elif config.dataset.name == 'zinc250k':
        num_node_type = 9
        num_adj_type = 4
    else:
        raise NotImplementedError

    """encode node and edge attributes"""
    node_encoding = config.train.node_encoding
    edge_encoding = config.train.edge_encoding
    assert node_encoding in ['one_hot', 'ddpm', 'bits']
    assert edge_encoding in ['one_hot', 'ddpm', 'bits']

    if node_encoding == 'one_hot':
        # defer one_hot encoding in the runner for mini-batch processing to save memory
        node_out = node_pad
    else:
        node_out = attribute_converter(node_pad, node_flags_pad, in_encoding='int', out_encoding=node_encoding,
                                       num_attr_type=num_node_type, flag_nodes=True,
                                       flag_in_ddpm_range=False, flag_out_ddpm_range=True)

    if edge_encoding == 'one_hot':
        # defer one_hot encoding in the runner for mini-batch processing to save memory
        adj_out = adj_pad
    else:
        adj_out = attribute_converter(adj_pad, node_flags_pad, in_encoding='int', out_encoding=edge_encoding,
                                      num_attr_type=num_adj_type, flag_adjs=True,
                                      flag_in_ddpm_range=False, flag_out_ddpm_range=True)

    train_dataset = TensorDataset(adj_out[train_idx], node_out[train_idx], node_flags_pad[train_idx])
    test_dataset = TensorDataset(adj_out[test_idx], node_out[test_idx], node_flags_pad[test_idx])

    time_spent = time.time() - time_start
    logging.info("Molecule dataset loaded, time: {:.2f}".format(time_spent))
    return train_dataset, test_dataset


def load_dataset_general(config):
    """
    Setup training/validation/testing dataloader for general graph datasets.
    """

    def _load_nx_object_from_pickle(data_dir='data', file_name=None):
        """Load graph list from pickle file."""
        if dist.is_initialized():
            dist.barrier()
        file_path = os.path.join(data_dir, file_name)

        _pickle_path = file_path + '.pkl'
        assert os.path.exists(_pickle_path), 'File not found: ' + _pickle_path
        graph_list = pickle.load(open(_pickle_path, 'rb'))

        txt_path = file_path + '.txt'
        if os.path.exists(txt_path):
            with open(file_path + '.txt', 'r') as f:
                info = f.read()
            logging.info('load dataset: ' + info)
        return graph_list

    def _nx_graphs_to_dataset(graph_list, ddpm_scale=True):
        """Turn the list of networkx graphs into pytorch tensors and make TensorDataset."""
        adjs_list = []
        for g in graph_list:
            if isinstance(g, nx.Graph):
                pass
            elif isinstance(g, nx.ndarray):
                raise ValueError('Graph must be either networkx.Graph. But got numpy.ndarray.')
            else:
                raise NotImplementedError
            adj = nx.to_numpy_matrix(g)
            padded_adj = pad_adjs(adj, node_number=config.dataset.max_node_num)
            adjs_list.append(padded_adj)

        adjs_np = np.asarray(adjs_list)

        adjs_tensor = torch.tensor(adjs_np, dtype=torch.float32)
        node_feat_tensor = torch.zeros(adjs_tensor.shape[:2], dtype=torch.float32)  # [B, N], dummy node feature
        node_feat_tensor = node_feat_tensor.unsqueeze(-1)  # [B, N, 1]
        node_flags_tensor = adjs_tensor.sum(-1).gt(1e-5).to(dtype=torch.float32)  # [B, N]

        if ddpm_scale:
            adjs_tensor = attribute_converter(in_attr=adjs_tensor, attr_flags=node_flags_tensor,
                                              in_encoding='int', out_encoding='ddpm', num_attr_type=2,
                                              flag_nodes=True, flag_adjs=False,
                                              flag_in_ddpm_range=False, flag_out_ddpm_range=True)

        tensor_ds = TensorDataset(adjs_tensor, node_feat_tensor, node_flags_tensor)
        return tensor_ds

    # pre-created public graph datasets
    train_graph_list = _load_nx_object_from_pickle(data_dir='data', file_name=config.dataset.name + '_train')
    test_graph_list = _load_nx_object_from_pickle(data_dir='data', file_name=config.dataset.name + '_test')

    plot_graphs_list(train_graph_list, title='dataset_preview', max_num=16, save_dir=config.logdir)

    # debug mode, select a subset of the dataset, make the testing data equal to the training data
    if config.dataset.subset is not None:
        assert isinstance(config.dataset.subset, int)
        set_size = config.dataset.subset
        train_graph_list, test_graph_list = train_graph_list[:set_size], train_graph_list[:set_size]

    count_graph_statistics(train_graph_list, 'training set')
    count_graph_statistics(test_graph_list, 'testing set')

    train_ds = _nx_graphs_to_dataset(train_graph_list, ddpm_scale=True)
    test_ds = _nx_graphs_to_dataset(test_graph_list, ddpm_scale=True)
    return train_ds, test_ds, train_graph_list, test_graph_list


def count_graph_statistics(graph_list, section_nm):
    """
    Count the statistics of the graphs stored as networkx objects.
    @param graph_list: a list of networkx objects
    @param section_nm: a string for the section name
    """
    num_node_ls = [item.number_of_nodes() for item in graph_list]
    num_edge_ls = [item.number_of_edges() for item in graph_list]
    max_deg_ls = [max(d for n, d in item.degree) for item in graph_list]
    logging.info("*** Statistics for {:s}: #graphs: {:d} ***".format(
        section_nm, len(graph_list)))
    for nm, data in zip(['node', 'edge', 'max_degree'], [num_node_ls, num_edge_ls, max_deg_ls]):
        logging.info("*** Statistics for {:s}: {:s}, min: {:d}, max: {:d}, mean: {:d}, median: {:d} ***".format(
            section_nm, nm, np.min(data).astype(int), np.max(data).astype(int),
            np.mean(data).astype(int), np.median(data).astype(int)))
