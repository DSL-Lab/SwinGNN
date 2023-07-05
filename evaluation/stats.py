"""
Based on GDSS code, EDP-GNN code, GRAN code and GraphRNN code (modified).
https://github.com/harryjo97/GDSS
https://github.com/lrjconan/GRAN
https://github.com/ermongroup/GraphScoreMatching
https://github.com/JiaxuanYou/graph-generation
"""

import concurrent.futures
import os
import subprocess as sp
from datetime import datetime

from scipy.linalg import eigvalsh
import networkx as nx
import numpy as np
import copy

from evaluation.mmd import pad_array, compute_mmd, gaussian, gaussian_emd, gaussian_tv, compute_nspdk_mmd

PRINT_TIME = True
ORCA_DIR = 'evaluation/orca'  # the relative path to the orca dir


###############################################################################
def degree_worker(nx_graph):
    """
    Helper function for parallel computing of degree distribution.
    """
    return np.array(nx.degree_histogram(nx_graph))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """
    Compute the distance between the degree distributions of two unordered sets of graphs.
    Kernel: Gaussian TV.
    @param graph_ref_list: list of networkx graphs
    @param graph_pred_list: list of networkx graphs
    @param is_parallel: whether to use parallel computing
    @return: the distance between the degree distributions of two unordered sets of graphs
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def clustering_worker(param):
    """
    Helper function for parallel computing of clustering coefficient distribution.
    """
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    """
    Compute the distance between the clustering coefficient distributions of two unordered sets of graphs.
    Kernel: Gaussian TV.
    @param graph_ref_list: list of networkx graphs
    @param graph_pred_list: list of networkx graphs
    @param bins: number of bins for histogram
    @param is_parallel: whether to use parallel computing
    @return: the distance between the clustering coefficient distributions of two unordered sets of graphs
    """
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


###############################################################################

def spectral_worker(nx_graph):
    """
    Helper function for parallel computing of spectral distribution.
    """
    eigs = eigvalsh(nx.normalized_laplacian_matrix(nx_graph).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """
    Compute the distance between the spectral distributions of two unordered sets of graphs.
    Kernel: Gaussian TV.
    @param graph_ref_list: list of networkx graphs
    @param graph_pred_list: list of networkx graphs
    @param is_parallel: whether to use parallel computing
    @return: the distance between the spectral distributions of two unordered sets of graphs
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing spectral mmd: ', elapsed)
    return mmd_dist


###############################################################################


COUNT_START_STR = 'orbit counts: \n'


def edge_list_reindexed(nx_graph):
    """
    Reindex edges of a graph.
    """
    idx = 0
    id2idx = dict()
    for u in nx_graph.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in nx_graph.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    """
    Compute the orbit counts of a graph.
    @param graph: networkx graph

    Note: to run the compiled ORCA executable, you could try the following
    ./orca node 5 test.txt out.txt
    """
    current_time = datetime.now().strftime("%H_%M_%S")
    tmp_file_path = os.path.join(ORCA_DIR, 'tmp_{:s}_{:d}.txt'.format(current_time, os.getpid()))
    if os.path.exists(tmp_file_path):
        current_time = datetime.now().strftime("%H_%M_%S")
        tmp_file_path = os.path.join(ORCA_DIR, 'tmp_{:s}_{:d}_dup.txt'.format(current_time, os.getpid()))
    assert not os.path.exists(tmp_file_path)
    f = open(tmp_file_path, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = sp.check_output([os.path.join(ORCA_DIR, 'orca'), 'node', '4', tmp_file_path, 'std'])
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_file_path)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list, is_parallel=True):
    """
    Compute the distance between the orbit distributions of two unordered sets of graphs.
    Kernel: Gaussian TV.
    @param graph_ref_list: list of networkx graphs
    @param graph_pred_list: list of networkx graphs
    @param is_parallel: whether to use parallel computing (place holder only)
    @return: the distance between the orbit distributions of two unordered sets of graphs
    """
    total_counts_ref = []
    total_counts_pred = []

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except Exception as e:
            print(e)
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian_tv,
                           is_hist=False, sigma=30.0)

    print('-------------------------')
    print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    print('...')
    print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    print('-------------------------')
    return mmd_dist


###############################################################################
def nspdk_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, n_jobs=20)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing nspdk mmd: ', elapsed)
    return mmd_dist


###############################################################################
def adjs_to_graphs(adjs):
    """
    Convert a list of adjacency matrices to a list of graphs.
    @param adjs: list of adjacency matrices in numpy array
    @return: list of networkx graphs
    """
    graph_list = []
    for adj in adjs:
        G = nx.from_numpy_matrix(adj)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


def eval_acc_lobster_graph(graph_list):
    """
    Evaluate the accuracy of a list of graphs in predicting whether a graph is a lobster graph or not.
    @param graph_list: list of networkx graphs
    @return: accuracy scalar
    """
    graph_list = [copy.deepcopy(gg) for gg in graph_list]

    count = 0
    for gg in graph_list:
        if is_lobster_graph(gg):
            count += 1

    return count / float(len(graph_list))


def is_lobster_graph(nx_graph):
    """
    Check if a given graph is a lobster graph or not.
    Removing leaf nodes twice:
    lobster -> caterpillar -> path
    """
    # Check if G is a tree
    if nx.is_tree(nx_graph):
        # Check if G is a path after removing leaves twice
        leaves = [n for n, d in nx_graph.degree() if d == 1]
        nx_graph.remove_nodes_from(leaves)

        leaves = [n for n, d in nx_graph.degree() if d == 1]
        nx_graph.remove_nodes_from(leaves)

        num_nodes = len(nx_graph.nodes())
        num_degree_one = [d for n, d in nx_graph.degree() if d == 1]
        num_degree_two = [d for n, d in nx_graph.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


METHOD_NAME_TO_FUNC = {
    'degree': degree_stats,
    'cluster': clustering_stats,
    'orbit': orbit_stats_all,
    'spectral': spectral_stats,
    'nspdk': nspdk_stats
}


def eval_graph_list(graph_ref_list, grad_pred_list, methods=None):
    """
    Evaluate the graph statistics given networkx graphs of reference and generated graphs.
    @param graph_ref_list: list of networkx graphs
    @param grad_pred_list: list of networkx graphs
    @param methods: list of methods to evaluate
    @return: a dictionary of results
    """
    if methods is None:
        methods = ['degree', 'cluster', 'orbit', 'spectral']
    print("Size of reference graphs: {:d}, size of generated graphs: {:d}".format(
        len(graph_ref_list), len(grad_pred_list)))
    results = {}
    for method in methods:
        results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, grad_pred_list, is_parallel=False)
    results['average'] = np.mean(list(results.values()))
    print(results)
    return results


def eval_torch_batch(ref_batch, pred_batch, methods=None):
    """
    Evaluate the graph statistics given pytorch tensors of reference and generated adjacency matrices.
    @param ref_batch: pytorch tensor of shape (batch_size, num_nodes, num_nodes)
    @param pred_batch: pytorch tensor of shape (batch_size, num_nodes, num_nodes)
    @param methods: list of methods to evaluate
    @return: dictionary of results
    """
    graph_ref_list = adjs_to_graphs(ref_batch.detach().cpu().numpy())
    grad_pred_list = adjs_to_graphs(pred_batch.detach().cpu().numpy())
    results = eval_graph_list(graph_ref_list, grad_pred_list, methods=methods)
    return results
