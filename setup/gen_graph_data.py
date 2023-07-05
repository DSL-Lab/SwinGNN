"""
Based on GDSS code, EDP-GNN code, GRAN code and GraphRNN code (modified).
https://github.com/harryjo97/GDSS
https://github.com/lrjconan/GRAN
https://github.com/ermongroup/GraphScoreMatching
https://github.com/JiaxuanYou/graph-generation
"""

import json
import os
import pickle
import random
import sys
import numpy as np
import networkx as nx
from scipy import sparse as sp

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))
os.chdir(os.path.abspath(os.path.join(__file__, '../..')))
from utils.visual_utils import plot_graphs_list


def n_community(num_communities, max_nodes, p_inter=0.05):
    """
    networkx generator for community graph dataset
    """
    assert num_communities > 1

    one_community_size = max_nodes // num_communities
    c_sizes = [one_community_size] * num_communities
    total_nodes = one_community_size * num_communities

    """ 
    here we calculate `p_make_a_bridge` so that `p_inter = \mathbb{E}(Number_of_bridge_edges) / Total_number_of_nodes `

    To make it more clear: 
    let `M = num_communities` and `N = one_community_size`, then

    ```
    p_inter
    = \mathbb{E}(Number_of_bridge_edges) / Total_number_of_nodes
    = (p_make_a_bridge * C_M^2 * N^2) / (MN)  # see the code below for this derivation
    = p_make_a_bridge * (M-1) * N / 2
    ```

    so we have:
    """
    p_make_a_bridge = p_inter * 2 / ((num_communities - 1) * one_community_size)

    print(num_communities, total_nodes, end=' ')
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]

    G = nx.disjoint_union_all(graphs)
    communities = [G.subgraph(c) for c in nx.connected_components(G)]
    add_edge = 0
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):  # loop for C_M^2 times
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:  # loop for N times
                for n2 in nodes2:  # loop for N times
                    if np.random.rand() < p_make_a_bridge:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
                        add_edge += 1
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
                add_edge += 1
    # print('connected comp: ', len(communities),
    #       'add edges: ', add_edge)
    # print(G.number_of_edges())
    return G


NAME_TO_NX_GENERATOR = {
    'community': n_community,
    'lobster': nx.random_lobster
}


class GraphGenerator:
    """
    Graph generator for community and lobster graph datasets.
    """
    def __init__(self, graph_type='community', possible_params_dict=None, corrupt_func=None):
        if possible_params_dict is None:
            possible_params_dict = {}
        assert isinstance(possible_params_dict, dict)
        self.count = {k: 0 for k in possible_params_dict}
        self.possible_params = possible_params_dict
        self.corrupt_func = corrupt_func
        self.nx_generator = NAME_TO_NX_GENERATOR[graph_type]

    def __call__(self):
        params = {}
        for k, v_list in self.possible_params.items():
            params[k] = np.random.choice(v_list)
        graph = self.nx_generator(**params)
        graph = nx.relabel.convert_node_labels_to_integers(graph)
        if self.corrupt_func is not None:
            graph = self.corrupt_func(self.corrupt_func)
        return graph


def gen_graph_comm_grid_lobster(graph_type='grid', possible_params_dict=None, corrupt_func=None, length=1024,
                                save_dir=None, file_name=None, max_node=None, min_node=None):
    """
    Generate graph list for community, grid and lobster datasets.
    """
    assert graph_type in ['community', 'grid', 'lobster']
    params = locals()
    print('{:s} Generating graph data {:s} {:s}'.format('-'*20, graph_type, '-'*20))
    print('gen data: ' + json.dumps(params))
    if file_name is None:
        file_name = graph_type + '_' + str(length)
    file_path = os.path.join(save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    if graph_type == 'grid':
        graph_generator = None
    else:
        graph_generator = GraphGenerator(graph_type=graph_type,
                                         possible_params_dict=possible_params_dict,
                                         corrupt_func=corrupt_func)
    graph_list = []
    i = 0
    max_N = 0

    # control reproducibility
    random.seed(0)
    np.random.seed(0)

    if graph_type == 'grid':
        # grid data
        for i in possible_params_dict['m']:
            for j in possible_params_dict['n']:
                graph_list.append(nx.grid_2d_graph(i, j))
        max_N = max([g.number_of_nodes() for g in graph_list])
    else:
        # community and lobster data
        while i < length:
            graph = graph_generator()
            if max_node is not None and graph.number_of_nodes() > max_node:
                continue
            if min_node is not None and graph.number_of_nodes() < min_node:
                continue
            # print(i, graph.number_of_nodes(), graph.number_of_edges(), end="")
            print("Iteration: {:d}, #nodes: {:d}, #edges: {:d}".format(
                i, graph.number_of_nodes(), graph.number_of_edges()), end="\r")
            max_N = max(max_N, graph.number_of_nodes())
            if graph.number_of_nodes() <= 1:
                continue
            graph_list.append(graph)
            i += 1
        print()

    if save_dir is not None:
        save_with_split(graph_list, file_path)
        with open(file_path + '.txt', 'w') as f:
            f.write(json.dumps(params))
            f.write(f'max node number: {max_N}')

    # print("Max #nodes: {:d}".format(max_N))
    return graph_list


def citeseer_ego(radius=3, node_min=50, node_max=400):
    """
    Generate ego-small graphs from citeseer dataset.
    """

    def _parse_index_file(filename):
        """
        Parse index file for citeseer dataset.
        """
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def _load_citeseer(dataset="citeseer"):
        """
        Load citeseer dataset
        """
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i_nm in range(len(names)):
            load = pickle.load(open("dataset/ind.{}.{}".format(dataset, names[i_nm]), 'rb'), encoding='latin1')
            # print('loaded')
            objects.append(load)
            # print(load)
        # [x, tx, allx]: <class 'list'>: [(140, 1433), (1000, 1433), (1708, 1433)]
        # len(graph) == 2708
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = _parse_index_file("dataset/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)

        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        nx_graph = nx.from_dict_of_lists(graph)
        return features, nx_graph

    _, G = _load_citeseer()
    G = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=radius)
        assert isinstance(G_ego, nx.Graph)
        if G_ego.number_of_nodes() >= node_min and (G_ego.number_of_nodes() <= node_max):
            G_ego.remove_edges_from(list(nx.selfloop_edges(G_ego)))
            graphs.append(G_ego)
    return graphs


def dd_protein(min_num_nodes=20, max_num_nodes=1000, name='ENZYMES', node_attributes=True, graph_labels=True):
    """
    Load DD protein dataset.
    """
    print('Loading graph dataset: ' + str(name))
    G = nx.Graph()
    # load data
    path = 'dataset/' + name + '/'
    data_adj = np.loadtxt(path + name + '_A.txt', delimiter=',').astype(int)
    data_node_att = []
    if node_attributes:
        data_node_att = np.loadtxt(path + name + '_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path + name + '_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path + name + '_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path + name + '_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        subgraph = G.subgraph(nodes)
        if graph_labels:
            subgraph.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if min_num_nodes <= subgraph.number_of_nodes() <= max_num_nodes:
            graphs.append(subgraph)
            if subgraph.number_of_nodes() > max_nodes:
                max_nodes = subgraph.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print('Loaded')
    return graphs


def save_with_split(graph_list, file_path):
    """
    train-dev-test split consistent with GRAN
    first 80% for training, the rest 20% for testing, the first 20% for validation
    taken from https://github.com/lrjconan/GRAN/blob/fc9c04a3f002c55acf892f864c03c6040947bc6b/runner/gran_runner.py#L131
    """

    # save the whole dataset
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(obj=graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)

    # split and save the train, dev, test dataset
    num_train = int(len(graph_list) * 0.8)
    num_dev = int(len(graph_list) * 0.2)
    num_test = len(graph_list) - num_train

    # count statistics
    num_node_ls = [item.number_of_nodes() for item in graph_list]
    num_edge_ls = [item.number_of_edges() for item in graph_list]
    max_deg_ls = [max(d for n, d in item.degree) for item in graph_list]
    for nm, data in zip(['node', 'edge', 'max_degree'], [num_node_ls, num_edge_ls, max_deg_ls]):
        print("*** Statistics: {:s}, min: {:d}, max: {:d}, mean: {:d}, median: {:d} ***".format(
            nm, np.min(data).astype(int), np.max(data).astype(int),
            np.mean(data).astype(int), np.median(data).astype(int)))

    graph_train = graph_list[:num_train]
    graph_dev = graph_list[:num_dev]
    graph_test = graph_list[num_train:]

    print("Dataset split: train size={:d}, val size={:d}, test size={:d}".format(
        len(graph_train), len(graph_dev), len(graph_test)
    ))

    for kw_str in ['train', 'dev', 'test']:
        if kw_str == 'train':
            graph_to_save = graph_train
        elif kw_str == 'dev':
            graph_to_save = graph_dev
        elif kw_str == 'test':
            graph_to_save = graph_test
        else:
            raise NotImplementedError

        with open(file_path + '_' + kw_str + '.pkl', 'wb') as f:
            pickle.dump(obj=graph_to_save, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def save_ego_dd_datasets(graphs, save_name):
    """
    Code reference: https://github.com/ermongroup/GraphScoreMatching
    """
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', save_name)
    print(save_name, len(graphs))
    save_with_split(graphs, file_path)
    with open(file_path + '.txt', 'w') as f:
        f.write(save_name + '\n')
        f.write(str(len(graphs)))


def main():
    """
    Generate all kinds of graph datasets.
    """

    """community-small dataset"""
    # parameters taken from EDP-GNN
    # https://github.com/ermongroup/GraphScoreMatching/blob/0b8206c92860f631240599ad38f00da25a87b8d9/gen_data.py#L7
    file_name = 'community_small'
    res_graph_list = gen_graph_comm_grid_lobster(graph_type='community',
                                                 possible_params_dict={
                                                     'num_communities': [2],
                                                     'max_nodes': np.arange(12, 21).tolist()},
                                                 corrupt_func=None, length=100, save_dir='data', file_name=file_name)
    plot_graphs_list(res_graph_list, title=file_name, save_dir='data')

    """grid dataset"""
    # parameters taken from GRAN
    # https://github.com/lrjconan/GRAN/blob/fc9c04a3f002c55acf892f864c03c6040947bc6b/utils/data_helper.py#L164
    file_name = 'grid'
    res_graph_list = gen_graph_comm_grid_lobster(graph_type='grid',
                                                 possible_params_dict={
                                                     'm': np.arange(10, 20).tolist(),
                                                     'n': np.arange(10, 20).tolist()},
                                                 length=100, save_dir='data', file_name=file_name)
    plot_graphs_list(res_graph_list, title=file_name, save_dir='data')

    """lobster dataset"""
    # parameters taken from GRAN
    # https://github.com/lrjconan/GRAN/blob/fc9c04a3f002c55acf892f864c03c6040947bc6b/utils/data_helper.py#L169
    file_name = 'lobster'
    res_graph_list = gen_graph_comm_grid_lobster(graph_type='lobster',
                                                 possible_params_dict={
                                                     'n': [80],
                                                     'p1': [0.7],
                                                     'p2': [0.7]},
                                                 corrupt_func=None, length=100, save_dir='data', file_name=file_name,
                                                 min_node=10, max_node=100)
    plot_graphs_list(res_graph_list, title=file_name, save_dir='data')

    """ego-small"""
    # parameters taken from EDP-GNN
    # https://github.com/ermongroup/GraphScoreMatching/blob/0b8206c92860f631240599ad38f00da25a87b8d9/process_dataset.py#L147
    dataset_name = 'ego'
    suffix = '_small'
    print('{:s} Generating graph data {:s} {:s}'.format('-'*20, dataset_name+suffix, '-'*20))
    graphs = citeseer_ego(radius=1, node_min=4, node_max=18)[:200]
    save_ego_dd_datasets(graphs, dataset_name + suffix)
    plot_graphs_list(graphs, title=dataset_name+suffix, save_dir='data')
    # print("Max #nodes: {:d}".format(max([g.number_of_nodes() for g in graphs])))

    """DD PROTEIN"""
    # parameters taken from GRAN
    # https://github.com/lrjconan/GRAN/blob/fc9c04a3f002c55acf892f864c03c6040947bc6b/utils/data_helper.py#L191
    dataset_name = 'DD'
    suffix = ''
    print('{:s} Generating graph data {:s} {:s}'.format('-' * 20, dataset_name + suffix, '-' * 20))
    graphs = dd_protein(min_num_nodes=100, max_num_nodes=500, name=dataset_name,
                        node_attributes=False, graph_labels=True)
    save_ego_dd_datasets(graphs, dataset_name.lower() + suffix)
    plot_graphs_list(graphs, title=dataset_name + suffix, save_dir='data')
    # print("Max #nodes: {:d}".format(max([g.number_of_nodes() for g in graphs])))


if __name__ == '__main__':
    main()
