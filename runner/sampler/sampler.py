import logging

import os
import pdb
import pickle
import time
import pandas as pd
import numpy as np
import torch

import networkx as nx

from evaluation.stats import adjs_to_graphs
from runner.sampler.sampler_utils import visualize_interim_adjs, visualize_evaluate_final_adjs, split_test_set
from utils.dist_training import get_ddp_save_flag, gather_tensors
from utils.graph_utils import add_sym_normal_noise
from utils.attribute_code import attribute_converter


def go_sampling(epoch, model, dist_helper, test_dl, mc_sampler, config, writer=None,
                init_noise_strengths=(float('inf')), sanity_check=False, eval_mode=False,
                sampling_params=None):
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

    os.makedirs(shared_plot_dir, exist_ok=True)
    logging.info("Sampling {:d} samples with batch size {:d}".format(total_samples, batch_size))
    if not isinstance(init_noise_strengths, list) and not isinstance(init_noise_strengths, tuple):
        init_noise_strengths = [init_noise_strengths]

    # hyperparameter controlling the subset of interim adjs to store in memory
    # number of interim adjs to visualize is controlled in the visualize_interim_adjs function
    max_num_interim_adjs = 10

    # step_per_interim_adjs indicates how many sampling steps have past during two selected interim adjs
    # this is used to recover the exact original sampling step
    if mc_sampler.objective == 'diffusion':
        step_per_interim_adjs = max(mc_sampler.max_steps / max_num_interim_adjs, 1)
    elif mc_sampler.objective == 'edm':
        step_per_interim_adjs = max(mc_sampler.num_steps / max_num_interim_adjs, 1)
    else:
        step_per_interim_adjs = 1

    # Load testing data
    sampler_dl = split_test_set(test_dl, total_samples, batch_size, dist_helper)

    """Draw samples and evaluate"""
    model.eval()
    for noising_coef in init_noise_strengths:
        # loop over initial noise strengths

        """Draw samples from the MCMC sampler"""
        flag_pure_noise = noising_coef == float('inf')

        init_adjs_ls, final_samples_ls, interim_samples_ls = [], [], []
        _sampler_dl_test_adjs_ls, _sampler_dl_test_node_flags_ls = [], []
        i_generated = 0
        for i_iter, data_tuple in enumerate(sampler_dl):
            if len(data_tuple) == 2:
                test_adjs_gt, test_x_gt = data_tuple
                test_node_flags = test_adjs_gt.abs().sum(-1).gt(1e-5).to(dtype=torch.float32)  # [B, N]
            elif len(data_tuple) == 3:
                test_adjs_gt, test_x_gt, test_node_flags = data_tuple
            else:
                raise NotImplementedError
            test_adjs_gt = test_adjs_gt.to(config.dev)  # [B, N, N]
            test_x_gt = test_x_gt.to(config.dev)  # [B, N, 1], all zeros by default
            test_node_flags = test_node_flags.to(config.dev)  # [B, N]

            # faithfully record whatever returned
            _sampler_dl_test_adjs_ls.append(test_adjs_gt)
            _sampler_dl_test_node_flags_ls.append(test_node_flags)

            # Initialize noisy data
            if flag_pure_noise:
                logging.info("--- Sampling from pure noisy data ---")
                init_adjs_sampler = None
            else:
                logging.info("--- Sampling from clean data + noise (Gaussian std={:.4f})---".format(noising_coef))
                cur_sigmas = torch.tensor(noising_coef).expand(total_samples).to(config.dev)
                init_adjs_sampler, _ = add_sym_normal_noise(test_adjs_gt, torch.ones_like(cur_sigmas),
                                                            cur_sigmas, test_node_flags)

            logging.info("Generating [{:d} - {:d}]/ {:d} samples now... ({:d} / {:d} run)".format(
                i_generated, i_generated + test_adjs_gt.size(0), total_samples, i_iter + 1, len(sampler_dl)))
            i_generated += test_adjs_gt.size(0)

            # [B, N, N] + [T, B, N, N]
            final_samples, interim_samples = mc_sampler.sample(model=model, node_flags=test_node_flags,
                                                               init_adjs=init_adjs_sampler,
                                                               flag_interim_adjs=True,
                                                               sanity_check_gt_adjs=test_adjs_gt if sanity_check else None,
                                                               max_num_interim_adjs=max_num_interim_adjs
                                                               )

            # [B, N, N], retrieve actual init adjs
            if mc_sampler.objective == "diffusion":
                # note: interim samples contain x_t and x_0 predictions, a list of [T, B, N, N] + [T, B, N, N]
                init_adjs = interim_samples[0][0] if flag_pure_noise else init_adjs_sampler.cpu()
            elif mc_sampler.objective == "edm":
                # note: interim samples only have x_t predictions
                init_adjs = interim_samples[0] if flag_pure_noise else init_adjs_sampler.cpu()
            else:
                raise NotImplementedError

            final_samples = final_samples.clamp(min=-1.0, max=1.0)
            final_samples = attribute_converter(in_attr=final_samples, attr_flags=test_node_flags.cpu(),
                                                in_encoding='ddpm', out_encoding='int', num_attr_type=2,
                                                flag_nodes=True, flag_adjs=False,
                                                flag_in_ddpm_range=True, flag_out_ddpm_range=False)

            init_adjs_ls.append(init_adjs)                          # [B, N, N]
            final_samples_ls.append(final_samples)                  # [B, N, N]
            interim_samples_ls.append(interim_samples)              # [T, B, N, N]

        # end of sample_dl loop
        init_adjs = torch.cat(init_adjs_ls, dim=0)                  # [B, N, N]
        final_samples = torch.cat(final_samples_ls, dim=0)          # [B, N, N]
        _sampler_dl_test_adjs = torch.cat(_sampler_dl_test_adjs_ls, dim=0).cpu()  # [B, N, N]
        _sampler_dl_test_node_flags = torch.cat(_sampler_dl_test_node_flags_ls, dim=0).cpu()  # [B, N]
        if isinstance(interim_samples, torch.Tensor):
            # only x_t predictions
            interim_samples = torch.cat(interim_samples_ls, dim=1)  # [T, B, N, N]
        else:
            # x_t predictions + x_0 predictions
            len_tuple = len(interim_samples)
            interim_samples = []
            for i_tuple in range(len_tuple):
                # [T, B, N, N]
                interim_samples_i_tuple = torch.cat([elem[i_tuple] for elem in interim_samples_ls], dim=1)
                interim_samples.append(interim_samples_i_tuple)
            interim_samples = tuple(interim_samples)

        if dist_helper.is_ddp:
            init_adjs = gather_tensors(init_adjs, cat_dim=0, device=config.dev).cpu()
            final_samples = gather_tensors(final_samples, cat_dim=0, device=config.dev).cpu()
            _sampler_dl_test_adjs = gather_tensors(_sampler_dl_test_adjs, cat_dim=0, device=config.dev).cpu()
            _sampler_dl_test_node_flags = gather_tensors(_sampler_dl_test_node_flags, cat_dim=0, device=config.dev).cpu()
            if isinstance(interim_samples, torch.Tensor):
                interim_samples = gather_tensors(interim_samples, cat_dim=1, device=config.dev).cpu()
            else:
                # tuple concatenation
                interim_samples_ls = []
                for i in range(len(interim_samples)):
                    interim_samples_ls.append(gather_tensors(interim_samples[i], cat_dim=1, device=config.dev).cpu())
                interim_samples = tuple(interim_samples_ls)

        """Compute MMD and visualize the final sample"""
        if get_ddp_save_flag():
            # Init
            plot_subdir = "{:s}_exp_{:s}_{:s}".format("pure_noise" if flag_pure_noise else "denoising",
                                                      epoch_or_eval_stamp,
                                                      'sanity_check' if sanity_check else 'model_inference')
            if sanity_check:
                plot_subtitle = "sanity_check"
            else:
                plot_subtitle = "pure_noise" if flag_pure_noise else "fixed_sigma_{:.4f}".format(noising_coef)

            if sampling_params is not None:
                fig_keyword = sampling_params['model_nm'] + '_weight_{:s}'.format(sampling_params['weight_kw'])
                plot_subdir = fig_keyword + '_' + plot_subdir
                plot_subtitle = fig_keyword + '_' + plot_subtitle
            fig_title = '{:s}_{:s}.png'.format(epoch_or_eval_stamp, plot_subtitle)
            path_plot_subdir = os.path.join(shared_plot_dir, plot_subdir)
            path_final_samples_array = os.path.join(path_plot_subdir, 'final_samples_array.npz')
            path_final_samples_graph = os.path.join(path_plot_subdir, 'final_samples_graph.pt')
            os.makedirs(path_plot_subdir, exist_ok=True)

            # Note we must use exactly what is returned in the sampler_dl.
            # Otherwise, the node flags would be problematic and the final output would be wrong.
            test_adjs_gt = _sampler_dl_test_adjs.cpu()
            test_node_flags = _sampler_dl_test_node_flags.cpu()

            # save final samples
            np.savez_compressed(path_final_samples_array, samples=final_samples.cpu().numpy())

            final_samples_graph = adjs_to_graphs(final_samples.cpu().numpy())  # nx objects
            pickle.dump(final_samples_graph, open(path_final_samples_graph, 'wb'))

            # evaluate final samples against the testing set
            test_adjs_gt = attribute_converter(in_attr=test_adjs_gt, attr_flags=test_node_flags.cpu(),
                                               in_encoding='ddpm', out_encoding='int', num_attr_type=2,
                                               flag_nodes=True, flag_adjs=False,
                                               flag_in_ddpm_range=True, flag_out_ddpm_range=False)

            result_dict = visualize_evaluate_final_adjs(init_adjs, final_samples, test_adjs_gt, test_node_flags, mc_sampler, fig_title, config.logdir)
            logging.info('MMD at {:s} with config {:s}: {}'.format(epoch_or_eval_stamp, plot_subtitle, result_dict))

            # save to tensorboard
            if writer is not None:
                for mmd_key in result_dict:
                    writer.add_scalar('MMD/{}'.format(mmd_key), result_dict[mmd_key],
                                      global_step=epoch)

            # save to CSV
            if sampling_params is not None:
                result_dict['save_path'] = path_final_samples_array
                result_dict['gen_data_size'] = len(final_samples)
                result_dict['test_data_size'] = len(test_adjs_gt)
                sampling_params.update(result_dict)

                df = pd.DataFrame.from_dict(data=sampling_params, orient='index').transpose()
                cols = ['model_nm', 'weight_kw', 'gen_data_size', 'test_data_size',
                        'degree', 'cluster', 'orbit', 'spectral', 'average',
                        'save_path', 'model_path']
                if 'perm_' in config.dataset.name:
                    cols.insert(4, 'non_novelty')
                df = df[cols]
                csv_path = os.path.join(config.logdir, 'eval_results.csv')
                df.to_csv(csv_path, header=not os.path.exists(csv_path), index=False, mode='a')

            # visualize interim samples, for efficiency, we only save one sample's interim results
            # all interim results -> store subset by max_num_interim_adjs -> visualize subset (inside func. below)
            visualize_interim_adjs(interim_samples, test_adjs_gt, test_node_flags, mc_sampler.objective,
                                   step_per_interim_adjs, plot_save_dir=os.path.join(shared_plot_dir, plot_subdir))

    # clean up
    del init_adjs, interim_samples, test_adjs_gt, test_node_flags, sampler_dl
