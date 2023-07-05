"""
Based on GDSS code and MoFlow code (modified).
https://github.com/harryjo97/GDSS
https://github.com/calvin-zcx/moflow
"""

import traceback
from logging import getLogger

import numpy
import numpy as np
import pandas as pd
import argparse
import time
import os
import pickle
import json
import sys
from rdkit import Chem
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from utils.mol_utils import mols_to_nx, smiles_to_mols


class GGNNPreprocessor(object):
    """
    Code adapted from chainer_chemistry/dataset/preprocessors/common
    """
    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False, kekulize=True):
        super(GGNNPreprocessor, self).__init__()
        self.add_Hs = add_Hs
        self.kekulize = kekulize

        if max_atoms >= 0 and 0 <= out_size < max_atoms:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = construct_discrete_edge_matrix(mol, out_size=self.out_size)
        return atom_array, adj_array

    def prepare_smiles_and_mol(self, mol):
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False,
                                            canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        if self.add_Hs:
            mol = Chem.AddHs(mol)
        if self.kekulize:
            Chem.Kekulize(mol)
        return canonical_smiles, mol

    def get_label(self, mol, label_names=None):
        if label_names is None:
            return []

        label_list = []
        for label_name in label_names:
            if mol.HasProp(label_name):
                label_list.append(mol.GetProp(label_name))
            else:
                label_list.append(None)
        return label_list


def type_check_num_atoms(mol, num_max_atoms=-1):
    """
    Code adapted from chainer_chemistry/dataset/preprocessors/common
    """
    num_atoms = mol.GetNumAtoms()
    if 0 <= num_max_atoms < num_atoms:
        raise MolFeatureExtractionError(
            'Number of atoms in mol {} exceeds num_max_atoms {}'
            .format(num_atoms, num_max_atoms))


class MolFeatureExtractionError(Exception):
    """
    Code adapted from chainer_chemistry/dataset/preprocessors/common
    """
    pass


def construct_atomic_number_array(mol, out_size=-1):
    """
    Code adapted from chainer_chemistry/dataset/preprocessors/common
    """
    atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
    n_atom = len(atom_list)

    if out_size < 0:
        return numpy.array(atom_list, dtype=numpy.int32)
    elif out_size >= n_atom:
        atom_array = numpy.zeros(out_size, dtype=numpy.int32)
        atom_array[:n_atom] = numpy.array(atom_list, dtype=numpy.int32)
        return atom_array
    else:
        raise ValueError('`out_size` (={}) must be negative or '
                         'larger than or equal to the number '
                         'of atoms in the input molecules (={})'
                         '.'.format(out_size, n_atom))


def construct_discrete_edge_matrix(mol, out_size=-1):
    """
    Code adapted from chainer_chemistry/dataset/preprocessors/common
    """
    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise ValueError(
            'out_size {} is smaller than number of atoms in mol {}'
            .format(out_size, N))
    adjs = numpy.zeros((4, size, size), dtype=numpy.float32)

    bond_type_to_channel = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        ch = bond_type_to_channel[bond_type]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjs[ch, i, j] = 1.0
        adjs[ch, j, i] = 1.0
    return adjs


class NumpyTupleDataset(Dataset):
    def __init__(self, datasets, transform=None):
        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[0])  # 133885
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        self._datasets = datasets
        self._length = length
        self.transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, (slice, list, np.ndarray)):
            length = len(batches[0])
            batches = [tuple([batch[i] for batch in batches])
                       for i in range(length)]   # six.moves.range(length)]
        else:
            batches = tuple(batches)

        if self.transform:
            batches = self.transform(batches)
        return batches

    def get_datasets(self):
        return self._datasets

    @classmethod
    def save(cls, filepath, numpy_tuple_dataset):
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        np.savez(filepath, *numpy_tuple_dataset._datasets)
        print('Save {} done.'.format(filepath))

    @classmethod
    def load(cls, filepath, transform=None):
        print('Loading file {}'.format(filepath))
        if not os.path.exists(filepath):
            raise ValueError('Invalid filepath {} for dataset'.format(filepath))
            # return None
        load_data = np.load(filepath)
        result = []
        i = 0
        while True:
            key = 'arr_{}'.format(i)
            if key in load_data.keys():
                result.append(load_data[key])
                i += 1
            else:
                break
        return cls(result, transform)


class DataFrameParser(object):
    """
    Code adapted from chainer_chemistry/dataset/parsers/data_frame_parser.py
    """
    def __init__(self, preprocessor,
                 labels=None,
                 smiles_col='smiles',
                 postprocess_label=None, postprocess_fn=None,
                 logger=None):
        super(DataFrameParser, self).__init__()
        if isinstance(labels, str):
            labels = [labels, ]
        self.labels = labels
        self.smiles_col = smiles_col
        self.postprocess_label = postprocess_label
        self.postprocess_fn = postprocess_fn
        self.logger = logger or getLogger(__name__)
        self.preprocessor = preprocessor

    def parse(self, df, return_smiles=False, target_index=None,
              return_is_successful=False):
        logger = self.logger
        pp = self.preprocessor
        smiles_list = []
        is_successful_list = []

        # counter = 0
        if isinstance(pp, GGNNPreprocessor):
            if target_index is not None:
                df = df.iloc[target_index]

            features = None
            smiles_index = df.columns.get_loc(self.smiles_col)
            if self.labels is None:
                labels_index = []  # dummy list
            else:
                labels_index = [df.columns.get_loc(c) for c in self.labels]

            total_count = df.shape[0]
            fail_count = 0
            success_count = 0
            for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
                smiles = row[smiles_index]
                # currently it assumes list
                labels = [row[i] for i in labels_index]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        fail_count += 1
                        if return_is_successful:
                            is_successful_list.append(False)
                        continue
                    # Note that smiles expression is not unique.
                    # we obtain canonical smiles
                    canonical_smiles, mol = pp.prepare_smiles_and_mol(mol)
                    input_features = pp.get_input_features(mol)

                    # Extract label
                    if self.postprocess_label is not None:
                        labels = self.postprocess_label(labels)

                    if return_smiles:
                        smiles_list.append(canonical_smiles)
                except MolFeatureExtractionError as e:
                    # This is expected error that extracting feature failed,
                    # skip this molecule.
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                except Exception as e:
                    logger.warning('parse(), type: {}, {}'.format(type(e).__name__, e.args))
                    logger.info(traceback.format_exc())
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                # Initialize features: list of list
                if features is None:
                    if isinstance(input_features, tuple):
                        num_features = len(input_features)
                    else:
                        num_features = 1
                    if self.labels is not None:
                        num_features += 1
                    features = [[] for _ in range(num_features)]

                if isinstance(input_features, tuple):
                    for i in range(len(input_features)):
                        features[i].append(input_features[i])
                else:
                    features[0].append(input_features)
                if self.labels is not None:
                    features[len(features) - 1].append(labels)
                success_count += 1
                if return_is_successful:
                    is_successful_list.append(True)
            ret = []

            for feature in features:
                try:
                    feat_array = numpy.asarray(feature)
                except ValueError:
                    # Temporal work around.
                    # See,
                    # https://stackoverflow.com/questions/26885508/why-do-i-get-error-trying-to-cast-np-arraysome-list-valueerror-could-not-broa
                    feat_array = numpy.empty(len(feature), dtype=numpy.ndarray)
                    feat_array[:] = feature[:]
                ret.append(feat_array)
            result = tuple(ret)
            logger.info('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'
                        .format(fail_count, success_count, total_count))
        else:
            raise NotImplementedError

        smileses = numpy.array(smiles_list) if return_smiles else None
        if return_is_successful:
            is_successful = numpy.array(is_successful_list)
        else:
            is_successful = None

        if isinstance(result, (tuple, list)):
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(*result)
            dataset = NumpyTupleDataset(result)
        else:
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(result)
            dataset = NumpyTupleDataset([result])

        return {"dataset": dataset,
                "smiles": smileses,
                "is_successful": is_successful}

    def extract_total_num(self, df):
        return len(df)


def process_for_nspdk(dataset):
    """
    Process the dataset for NSPDK metrics.
    """
    with open(f'dataset/valid_idx_{dataset.lower()}.json') as f:
        test_idx = json.load(f)

    if dataset == 'QM9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
        col = 'SMILES1'
    elif dataset == 'ZINC250k':
        col = 'smiles'
    else:
        raise ValueError(f"[ERROR] Unexpected value data_name={dataset}")

    smiles = pd.read_csv(f'dataset/{dataset.lower()}.csv')[col]
    test_smiles = [smiles.iloc[i] for i in test_idx]
    nx_graphs = mols_to_nx(smiles_to_mols(test_smiles))
    print(f'Converted the {dataset} dataset test molecules into {len(nx_graphs)} graphs')

    with open(f'data/{dataset.lower()}_test_nx.pkl', 'wb') as f:
        pickle.dump(nx_graphs, f)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='ZINC250k', choices=['ZINC250k', 'QM9'])
    args = parser.parse_args()

    start_time = time.time()
    data_name = args.dataset

    if data_name == 'ZINC250k':
        max_atoms = 38
        path = 'dataset/zinc250k.csv'
        smiles_col = 'smiles'
        label_idx = 1
    elif data_name == 'QM9':
        max_atoms = 9
        path = 'dataset/qm9.csv'
        smiles_col = 'SMILES1'
        label_idx = 2
    else:
        raise ValueError(f"[ERROR] Unexpected value data_name={data_name}")

    preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)

    print(f'Preprocessing {data_name} data')
    df = pd.read_csv(path, index_col=0)
    # Caution: Not reasonable but used in chain_chemistry\datasets\zinc.py:
    # 'smiles' column contains '\n', need to remove it.
    # Here we do not remove \n, because it represents atom N with single bond
    labels = df.keys().tolist()[label_idx:]
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col=smiles_col)
    result = parser.parse(df, return_smiles=True)

    dataset = result['dataset']
    smiles = result['smiles']

    NumpyTupleDataset.save(f'data/{data_name.lower()}_kekulized.npz', dataset)

    print(f'Processing {data_name} for nspdk.')
    process_for_nspdk(data_name)
    print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == '__main__':
    main()
