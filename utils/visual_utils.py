"""
Based on EDP-GNN code (modified).
https://github.com/ermongroup/GraphScoreMatching
"""

import logging
import os
import pdb
import warnings
import networkx as nx
import numpy as np
import torch
import pickle
from PIL import Image, ImageDraw, ImageFont

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

matplotlib.use('Agg')

warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)


options = {
    'node_size': 2,
    'edge_color': 'black',
    'linewidths': 1,
    'width': 0.5
}

CMAP = cm.jet


def plot_graphs_list(graphs, energy=None, node_energy_list=None, title='title', max_num=16, save_dir=None):
    """
    Plot graphs of nx.Graph objects.
    """
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = np.ceil(np.sqrt(max_num)).astype('int')
    figure = plt.figure()

    for i in range(max_num):
        idx = i * (batch_size // max_num)
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)

        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        if energy is not None:
            title_str += f'\n en={energy[idx]:.1e}'

        if node_energy_list is not None:
            node_energy = node_energy_list[idx]
            title_str += f'\n {np.std(node_energy):.1e}'
            nx.draw(G, with_labels=False, node_color=node_energy, cmap=cm.jet, **options)
        else:
            # print(nx.get_node_attributes(G, 'feature'))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)


def save_fig(save_dir=None, title='fig', dpi=300, fig_dir='fig'):
    """
    Figure saving helper.
    """
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(save_dir, fig_dir)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, title),
                    bbox_inches='tight',
                    dpi=dpi,
                    transparent=True)
        plt.close()
    return


def plot_graphs_adj(adjs, energy=None, node_num=None, title='title', max_num=20, save_dir=None):
    """
    Plot graphs of numpy arrays or torch tensors.
    """
    if isinstance(adjs, torch.Tensor):
        adjs = adjs.cpu().numpy()
    with_labels = (adjs.shape[-1] < 10)
    batch_size = adjs.shape[0]
    max_num = min(batch_size, max_num)
    img_c = np.ceil(np.sqrt(max_num)).astype(int)
    figure = plt.figure()
    for i in range(max_num):
        # idx = i * (adjs.shape[0] // max_num)
        idx = i
        adj = adjs[idx, :, :]
        G = nx.from_numpy_matrix(adj)
        assert isinstance(G, nx.Graph)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)

        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        if energy is not None:
            title_str += f'\n en={energy[idx]:.1e}'
        ax.title.set_text(title_str)
        nx.draw(G, with_labels=with_labels, **options)
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)

