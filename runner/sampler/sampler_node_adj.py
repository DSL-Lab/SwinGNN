import logging

import os
import pdb
import time
import pandas as pd
import numpy as np

from evaluation.mmd import compute_mmd, gaussian_tv, gaussian_emd

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, box_convert

from evaluation.mmd import compute_mmd
from evaluation.stats import eval_torch_batch
from utils.dist_training import get_ddp_save_flag
from utils.graph_utils import add_sym_normal_noise, mask_adjs, mask_nodes
from utils.attribute_code import bin2dec, attribute_converter, reshape_node_attr_mat_to_vec
from utils.mol_utils import construct_mol, correct_mol, valid_mol_can_with_seg, mols_to_smiles
from utils.mol_utils import load_smiles, canonicalize_smiles, mols_to_nx, smiles_to_mols
from evaluation.stats import eval_graph_list
from moses.metrics.metrics import get_all_metrics
from utils.dist_training import gather_tensors

from runner.sampler.sampler_utils import split_test_set


def mol_go_sampling(epoch, model, dist_helper, test_dl, mc_sampler, config,
                    init_noise_strengths=(float('inf')), sanity_check=False, eval_mode=False,
                    sampling_params=None, writer=None):
    """
    Create samples using the sampler and model.
    """

    """Initialization"""
    eval_size = config.test.eval_size
    flag_valid_eval_size = False
    if isinstance(eval_size, int):
        if eval_size > 0:
            flag_valid_eval_size = True

    if eval_mode:
        epoch_or_eval_stamp = 'eval_' + time.strftime('%b-%d-%H-%M-%S')
        shared_plot_dir = os.path.join(config.logdir, "sampling_during_evaluation")
        if flag_valid_eval_size:
            total_samples = eval_size
        else:
            total_samples = len(test_dl.dataset)
        batch_size = config.test.batch_size
    else:
        epoch_or_eval_stamp = 'eval_' + f"epoch_{epoch:05d}"
        shared_plot_dir = os.path.join(config.logdir, "sampling_during_training")
        if flag_valid_eval_size:
            total_samples = eval_size
        else:
            total_samples = config.train.batch_size
        batch_size = config.train.batch_size
    total_samples = min(len(test_dl.dataset), total_samples)  # cap the number of samples to generate for mol data
    os.makedirs(shared_plot_dir, exist_ok=True)
    logging.info("Sampling {:d} samples with batch size {:d}".format(total_samples, batch_size))
    if not isinstance(init_noise_strengths, list) and not isinstance(init_noise_strengths, tuple):
        init_noise_strengths = [init_noise_strengths]

    node_encoding = config.train.node_encoding
    edge_encoding = config.train.edge_encoding
    flag_node_multi_channel = node_encoding != 'ddpm'
    flag_edge_multi_channel = edge_encoding != 'ddpm'

    assert config.dataset.name in ['qm9', 'zinc250k']
    assert config.flag_mol
    if config.dataset.name == 'qm9':
        num_node_type = 4
        num_adj_type = 4
    elif config.dataset.name == 'zinc250k':
        num_node_type = 9
        num_adj_type = 4
    else:
        raise NotImplementedError

    if node_encoding == 'one_hot':
        num_node_chan = num_node_type
    elif node_encoding == 'bits':
        num_node_chan = np.ceil(np.log2(num_node_type)).astype(int)
    elif node_encoding == 'ddpm':
        num_node_chan = 1
    else:
        raise NotImplementedError

    if edge_encoding == 'one_hot':
        num_edge_chan = num_adj_type
    elif edge_encoding == 'bits':
        num_edge_chan = np.ceil(np.log2(num_adj_type)).astype(int)
    elif edge_encoding == 'ddpm':
        num_edge_chan = 1
    else:
        raise NotImplementedError

    # hyperparameter controlling the subset of interim adjs to store in memory
    max_num_interim_adjs = 10

    # Load testing data
    sampler_dl = split_test_set(test_dl, total_samples, batch_size, dist_helper)

    if hasattr(test_dl, 'train_smiles'):
        train_smiles, test_smiles = test_dl.train_smiles, test_dl.test_smiles
        test_graph_list = test_dl.test_graph_list
    else:
        train_smiles, test_smiles = load_smiles(config.dataset.name.upper())
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)
        test_graph_list = mols_to_nx(smiles_to_mols(test_smiles))

    """Draw samples and evaluate"""
    model.eval()
    for noising_coef in init_noise_strengths:
        # loop over initial noise strengths

        """Draw samples from the MCMC sampler"""
        flag_pure_noise = noising_coef == float('inf')

        final_samples_a_ls, final_samples_x_ls = [], []
        final_raw_a_ls, final_raw_x_ls = [], []
        final_samples_a_gt_ls, final_samples_x_gt_ls = [], []
        _sampler_dl_test_adjs_ls, _sampler_dl_test_nodes_ls, _sampler_dl_test_node_flags_ls = [], [], []
        i_generated = 0
        for i_iter, data_tuple in enumerate(sampler_dl):
            test_adjs_gt, test_nodes_gt, test_node_flags = data_tuple
            test_adjs_gt = test_adjs_gt.to(config.dev)  # [B, N, N] or [B, C, N, N]
            test_nodes_gt = test_nodes_gt.to(config.dev)  # [B, N] or [B, N, C]
            test_node_flags = test_node_flags.to(config.dev)  # [B, N]

            # convert node and edge attributes to one-hot encoding if necessary
            if edge_encoding == 'one_hot':
                test_adjs_gt = attribute_converter(test_adjs_gt, test_node_flags, num_attr_type=num_adj_type,
                                                   in_encoding='int', out_encoding='one_hot',
                                                   flag_adjs=True, flag_out_ddpm_range=True)  # [B, C, N, N]
            if node_encoding == 'one_hot':
                test_nodes_gt = attribute_converter(test_nodes_gt, test_node_flags, num_attr_type=num_node_type,
                                                    in_encoding='int', out_encoding='one_hot',
                                                    flag_nodes=True, flag_out_ddpm_range=True)  # [B, N, C]

            # faithfully record whatever returned
            _sampler_dl_test_adjs_ls.append(test_adjs_gt)
            _sampler_dl_test_nodes_ls.append(test_nodes_gt)
            _sampler_dl_test_node_flags_ls.append(test_node_flags)

            # Initialize noisy data
            if flag_pure_noise:
                logging.info("--- Sampling from pure noisy data ---")
                init_adjs_sampler = None
                init_nodes_sampler = None
            else:
                logging.info("--- Sampling from clean data + noise (Gaussian std={:.4f})---".format(noising_coef))
                cur_sigmas = torch.tensor(noising_coef).expand(total_samples).to(config.dev)
                init_adjs_sampler, _ = add_sym_normal_noise(test_adjs_gt, torch.ones_like(cur_sigmas),
                                                            cur_sigmas, test_node_flags)
                init_nodes_sampler = mask_nodes(torch.randn_like(test_nodes_gt), test_node_flags)

            logging.info("Generating [{:d} - {:d}]/ {:d} samples now... ({:d} / {:d} run)".format(
                i_generated, i_generated + test_adjs_gt.size(0), total_samples, i_iter + 1, len(sampler_dl)))
            i_generated += test_adjs_gt.size(0)

            # [B, N, N] + [T, B, N, N] + [B, N] + [T, B, N]
            final_samples_adjs, final_samples_nodes, interim_samples_adjs, interim_samples_nodes = mc_sampler.sample(
                model=model, node_flags=test_node_flags,
                init_adjs=init_adjs_sampler, init_nodes=init_nodes_sampler,
                flag_interim_adjs=True,
                sanity_check_gt_adjs=test_adjs_gt if sanity_check else None,
                sanity_check_gt_nodes=test_nodes_gt if sanity_check else None,
                max_num_interim_adjs=max_num_interim_adjs,
                flag_node_multi_channel=flag_node_multi_channel,
                flag_adj_multi_channel=flag_edge_multi_channel,
                num_node_chan=num_node_chan,
                num_edge_chan=num_edge_chan,
            )

            """quantization based on different encoding methods"""
            def _decode_node(node_samples, node_flags, encoding_method):
                node_samples = node_samples.clamp(-1.0, 1.0)
                if encoding_method in ['bits', 'one_hot']:
                    node_samples = torch.where(node_samples > 0.0, torch.ones_like(node_samples), -torch.ones_like(node_samples))
                    node_samples = mask_nodes(node_samples, node_flags)
                _q_node = attribute_converter(in_attr=node_samples, attr_flags=node_flags.cpu(),
                                              in_encoding=encoding_method, out_encoding='int', num_attr_type=num_node_type,
                                              flag_nodes=True, flag_adjs=False,
                                              flag_in_ddpm_range=True, flag_out_ddpm_range=False,
                                              flag_clamp_int=encoding_method == 'bits')
                return _q_node

            def _decode_adj(adj_samples, node_flags, encoding_method):
                adj_samples = adj_samples.clamp(-1.0, 1.0)
                if encoding_method in ['bits', 'one_hot']:
                    adj_samples = torch.where(adj_samples > 0.0, torch.ones_like(adj_samples), -torch.ones_like(adj_samples))
                    adj_samples = mask_adjs(adj_samples, node_flags)
                _q_adj = attribute_converter(in_attr=adj_samples, attr_flags=node_flags.cpu(),
                                             in_encoding=encoding_method, out_encoding='int', num_attr_type=num_adj_type,
                                             flag_nodes=True, flag_adjs=False,
                                             flag_in_ddpm_range=True, flag_out_ddpm_range=False,
                                             flag_clamp_int=encoding_method == 'bits')
                return _q_adj.contiguous()

            # note: the bits quantization may lead to out-of-boundary results when the generated samples are very poor
            # we clamp the converted integer values to the valid range (max)
            q_node = _decode_node(final_samples_nodes.cpu(), test_node_flags.cpu(), node_encoding)
            q_node_gt = _decode_node(test_nodes_gt.cpu(), test_node_flags.cpu(), node_encoding)
            q_adj = _decode_adj(final_samples_adjs.cpu(), test_node_flags.cpu(), edge_encoding)
            q_adj_gt = _decode_adj(test_adjs_gt.cpu(), test_node_flags.cpu(), edge_encoding)

            final_raw_a_ls.append(final_samples_adjs.cpu())  # [B, N, N], before quantization
            final_raw_x_ls.append(final_samples_nodes.cpu())  # [B, N], before quantization
            final_samples_a_ls.append(q_adj.cpu())  # [B, N, N], quantized!
            final_samples_x_ls.append(q_node.cpu())  # [B, N], quantized!
            final_samples_a_gt_ls.append(q_adj_gt.cpu())  # [B, N, N], quantized!
            final_samples_x_gt_ls.append(q_node_gt.cpu())  # [B, N], quantized!

        # end of sample_dl loop
        final_raw_adjs = torch.cat(final_raw_a_ls, dim=0)                   # [B, N, N]
        final_raw_nodes = torch.cat(final_raw_x_ls, dim=0)                  # [B, N]
        final_samples_adjs = torch.cat(final_samples_a_ls, dim=0)           # [B, N, N]
        final_samples_nodes = torch.cat(final_samples_x_ls, dim=0)          # [B, N]
        final_samples_adjs_gt = torch.cat(final_samples_a_gt_ls, dim=0)     # [B, N, N]
        final_samples_nodes_gt = torch.cat(final_samples_x_gt_ls, dim=0)    # [B, N]

        _sampler_dl_test_adjs = torch.cat(_sampler_dl_test_adjs_ls, dim=0).cpu()                # [B, N, N]
        _sampler_dl_test_nodes = torch.cat(_sampler_dl_test_nodes_ls, dim=0).cpu()              # [B, N]
        _sampler_dl_test_node_flags = torch.cat(_sampler_dl_test_node_flags_ls, dim=0).cpu()    # [B, N]

        if dist_helper.is_ddp:
            final_raw_adjs = gather_tensors(final_raw_adjs, cat_dim=0, device=config.dev).cpu()
            final_raw_nodes = gather_tensors(final_raw_nodes, cat_dim=0, device=config.dev).cpu()
            final_samples_adjs = gather_tensors(final_samples_adjs, cat_dim=0, device=config.dev).cpu()
            final_samples_nodes = gather_tensors(final_samples_nodes, cat_dim=0, device=config.dev).cpu()
            final_samples_adjs_gt = gather_tensors(final_samples_adjs_gt, cat_dim=0, device=config.dev).cpu()
            final_samples_nodes_gt = gather_tensors(final_samples_nodes_gt, cat_dim=0, device=config.dev).cpu()
            _sampler_dl_test_adjs = gather_tensors(_sampler_dl_test_adjs, cat_dim=0, device=config.dev).cpu()
            _sampler_dl_test_nodes = gather_tensors(_sampler_dl_test_nodes, cat_dim=0, device=config.dev).cpu()
            _sampler_dl_test_node_flags = gather_tensors(_sampler_dl_test_node_flags, cat_dim=0, device=config.dev).cpu()

        """Compute MMD and visualize the final sample"""
        if get_ddp_save_flag():
            # Init
            plot_subdir = "{:s}_exp_{:s}_{:s}".format("pure_noise" if flag_pure_noise else "denoising",
                                                      epoch_or_eval_stamp,
                                                      'sanity_check' if sanity_check else 'model_inference')

            path_plot_subdir = os.path.join(shared_plot_dir, plot_subdir)
            os.makedirs(path_plot_subdir, exist_ok=True)
            save_path_smiles = os.path.join(path_plot_subdir, 'gen_smiles.txt')
            path_final_samples_array = os.path.join(path_plot_subdir, 'final_samples_array.npz')

            # Note we must use exactly what is returned in the sampler_dl.
            # Otherwise, the node flags would be problematic and the final output would be wrong.
            test_adjs_gt = _sampler_dl_test_adjs.cpu()
            test_nodes_gt = _sampler_dl_test_nodes.cpu()
            test_node_flags = _sampler_dl_test_node_flags.cpu()

            # reconstruct molecules using ops on integer matrix, following GDSS
            final_samples_adjs = final_samples_adjs - 1
            final_samples_adjs[final_samples_adjs == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

            # [B, 4, N, N]
            adj_one_hot = F.one_hot(final_samples_adjs.long(), num_classes=num_adj_type).permute(0, 3, 1, 2)

            # use one-hot encoding for node type representation
            x_one_hot = F.one_hot(final_samples_nodes.long(), num_classes=num_node_type)  # [B, N, 4]
            x_one_hot = mask_nodes(x_one_hot, test_node_flags)
            x_one_hot = torch.concat([x_one_hot, 1 - x_one_hot.sum(dim=-1, keepdim=True)], dim=-1).numpy()  # 32, 9, 4 -> 32, 9, 5

            if config.dataset.name == 'qm9':
                atomic_num_list = [6, 7, 8, 9, 0]
            elif config.dataset.name == 'zinc250k':
                atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            else:
                raise NotImplementedError

            gen_mols, num_no_correct = [], 0
            for x, a in zip(x_one_hot, adj_one_hot):
                # [N, 5] + [4, N, N]
                mol = construct_mol(x, a, atomic_num_list)
                c_mol, no_correct = correct_mol(mol)
                if no_correct:
                    num_no_correct += 1
                vc_mol = valid_mol_can_with_seg(c_mol, largest_connected_comp=True)
                gen_mols.append(vc_mol)
            gen_mols = [mol for mol in gen_mols if mol is not None]  # remove None molecules

            gen_smiles = mols_to_smiles(gen_mols)
            gen_smiles = [smi for smi in gen_smiles if len(smi)]  # remove empty smiles

            # save results
            with open(save_path_smiles, 'a') as f:
                for smiles in gen_smiles:
                    f.write(f'{smiles}\n')

            # -------- Evaluation --------
            scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=config.dev, n_jobs=8,
                                     test=test_smiles, train=train_smiles)

            scores_nspdk = eval_graph_list(test_graph_list, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']

            logging.info(f'Number of molecules: {len(gen_mols)}')
            logging.info(f'validity w/o correction: {num_no_correct / len(gen_mols)}')
            for metric in ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test', 'Novelty']:
                logging.info(f'{metric}: {scores[metric]}')
            logging.info(f'NSPDK MMD: {scores_nspdk}')
            logging.info('=' * 100)

            # save and evaluate final samples
            np.savez_compressed(path_final_samples_array,
                                samples_a=final_samples_adjs.cpu().numpy(),
                                samples_x=final_samples_nodes.cpu().numpy(),
                                raw_a=final_raw_adjs.cpu().numpy(),
                                raw_x=final_raw_nodes.cpu().numpy(),
                                node_flags=test_node_flags.cpu().bool().numpy(),
                                gt_a=final_samples_adjs_gt.cpu().numpy(),
                                gt_x=final_samples_nodes_gt.cpu().numpy(),
                                )

            # save to CSV
            if sampling_params is not None:
                result_dict = {}
                result_dict.update(scores)
                result_dict['nspdk'] = scores_nspdk

                result_dict['save_path_array'] = path_final_samples_array
                result_dict['save_path_smiles'] = save_path_smiles
                result_dict['gen_data_size'] = len(gen_mols)
                result_dict['test_data_size'] = len(test_smiles)
                result_dict['valid_wo_cor'] = float(num_no_correct / len(gen_mols))

                sampling_params.update(result_dict)

                df = pd.DataFrame.from_dict(data=sampling_params, orient='index').transpose()
                cols = ['model_nm', 'weight_kw', 'gen_data_size', 'test_data_size',
                        'valid', 'valid_wo_cor', 'unique@{:d}'.format(len(gen_mols)),  'Novelty', 'FCD/Test', 'nspdk',
                        'SNN/Test', 'Frag/Test', 'Scaf/Test', 'FCD/TestSF', 'SNN/TestSF', 'Frag/TestSF', 'Scaf/TestSF',
                        'IntDiv', 'IntDiv2', 'Filters', 'logP', 'SA', 'QED', 'weight',
                        'save_path_array', 'save_path_smiles', 'model_path']
                df = df[cols]
                csv_path = os.path.join(config.logdir, 'eval_results.csv')
                df.to_csv(csv_path, header=not os.path.exists(csv_path), index=False, mode='a')

            # save to tensorboard
            if writer is not None:
                _selected_metrics = ['valid', 'valid_wo_cor', 'unique@{:d}'.format(len(gen_mols)),
                                     'Novelty', 'FCD/Test', 'nspdk']
                _tag = 'gen_epoch/ema_kw_{}'.format(sampling_params['weight_kw'])

                for metrics in _selected_metrics:
                    if 'unique@' in metrics:
                        metrics_in_tag = 'unique'
                    elif 'FCD/Test' in metrics:
                        metrics_in_tag = 'FCD'
                    else:
                        metrics_in_tag = metrics
                    writer.add_scalar(f'{_tag}/{metrics_in_tag}', sampling_params[metrics], epoch)

    # clean up
    del test_adjs_gt, test_node_flags, sampler_dl
