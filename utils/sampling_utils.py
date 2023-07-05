import logging
import pdb

import torch

from utils.visual_utils import plot_graphs_adj
from runner.mcmc_sampler.diffusion import DiffusionSampler
from runner.mcmc_sampler.score_matching import ScoreMatchingSampler
from runner.mcmc_sampler.edm import EDMSampler, NodeAdjEDMSampler


def get_mc_sampler(config):
    """
    Configure MCMC sampler.
    """
    # Setup sampler
    flag_clip_samples = config.mcmc.sample_clip.min is not None and config.mcmc.sample_clip.max is not None
    if config.mcmc.name == 'score':
        mc_sampler = ScoreMatchingSampler(eps_step_size=config.mcmc.eps_step_size,
                                          eps_noise=config.mcmc.eps_noise,
                                          steps_per_sigma=config.mcmc.steps_per_sigma,
                                          sigma_num_slices=config.mcmc.sigmas.num_slices,
                                          sigma_min=config.mcmc.sigmas.min,
                                          sigma_max=config.mcmc.sigmas.max,
                                          sigma_preset=None,
                                          clip_samples=flag_clip_samples,
                                          clip_samples_min=config.mcmc.sample_clip.min,
                                          clip_samples_max=config.mcmc.sample_clip.max,
                                          clip_samples_scope=config.mcmc.sample_clip.scope,
                                          dev=config.dev)
    elif config.mcmc.name == 'diffusion':
        mc_sampler = DiffusionSampler(max_steps=config.mcmc.betas.max_steps,
                                      beta_min=config.mcmc.betas.min,
                                      beta_max=config.mcmc.betas.max,
                                      schedule=config.mcmc.betas.schedule,
                                      pred_target=config.mcmc.pred_target,
                                      clip_samples=flag_clip_samples,
                                      clip_samples_min=config.mcmc.sample_clip.min,
                                      clip_samples_max=config.mcmc.sample_clip.max,
                                      clip_samples_scope=config.mcmc.sample_clip.scope,
                                      dev=config.dev)
    elif config.mcmc.name == 'edm':
        if config.flag_mol:
            mc_sampler = NodeAdjEDMSampler(num_steps=config.mcmc.num_steps,
                                           clip_samples=flag_clip_samples,
                                           clip_samples_min=config.mcmc.sample_clip.min,
                                           clip_samples_max=config.mcmc.sample_clip.max,
                                           clip_samples_scope=config.mcmc.sample_clip.scope,
                                           dev=config.dev,
                                           objective='edm',
                                           self_condition=config.train.self_cond,
                                           symmetric_noise=True)
        else:
            mc_sampler = EDMSampler(num_steps=config.mcmc.num_steps,
                                    clip_samples=flag_clip_samples,
                                    clip_samples_min=config.mcmc.sample_clip.min,
                                    clip_samples_max=config.mcmc.sample_clip.max,
                                    clip_samples_scope=config.mcmc.sample_clip.scope,
                                    dev=config.dev,
                                    objective=config.mcmc.name,
                                    self_condition=config.train.self_cond)
    else:
        raise ValueError('Unknown mcmc method')

    # Print out sampler information
    if config.mcmc.name == "score":
        _sigma_list = mc_sampler.const_sigma_t.cpu().numpy().tolist()
        logging.info("Score estimation objective. \n"
                     "Sigma list: {}\n"
                     "Sampling steps per sigma: {:d}".format(_sigma_list, mc_sampler.steps_per_sigma))
    elif config.mcmc.name == "diffusion":
        logging.info("Diffusion denoising objective. \n"
                     "Prediction target: {:s}. Betas: min: {:.4f}, max: {:.4f}, #steps: {:d}, schedule: {:s}".format(
                        mc_sampler.pred_target, mc_sampler.beta_min, mc_sampler.beta_max,
                        mc_sampler.max_steps, mc_sampler.schedule))
    elif "edm" in config.mcmc.name:
        logging.info('EDM-variant objective. \n'
                     'Model: {:s}. Num of steps: {:d}'.format(config.mcmc.name, config.mcmc.num_steps))
    else:
        raise NotImplementedError

    logging.info('Self-conditioning: {}'.format(config.train.self_cond))

    return mc_sampler


def load_model(ckp_data, model, weight_keyword):
    """
    Load network weight.
    """
    assert weight_keyword in ckp_data
    cur_keys = sorted(list(model.state_dict().keys()))
    ckp_keys = sorted(list(ckp_data[weight_keyword].keys()))
    if set(cur_keys) == set(cur_keys) & set(ckp_keys):
        model.load_state_dict(ckp_data[weight_keyword], strict=True)
    else:
        to_load_state_dict = {}
        for cur_key, ckp_key in zip(cur_keys, ckp_keys):
            if cur_key == ckp_key:
                pass
            # note: .module prefix is added during the DP training
            elif cur_key.startswith('module.') and not ckp_key.startswith('module.'):
                assert cur_key == 'module.' + ckp_key
            elif ckp_key.startswith('module.') and not cur_key.startswith('module.'):
                assert 'module.' + cur_key == ckp_key
            else:
                pdb.set_trace()
                raise NotImplementedError
            to_load_state_dict[cur_key] = ckp_data[weight_keyword][ckp_key]
        assert set(cur_keys) == set(list(to_load_state_dict.keys()))
        model.load_state_dict(to_load_state_dict, strict=True)
        del to_load_state_dict
        torch.cuda.empty_cache()
    return model


def eval_sample_batch(sample_b, test_adj_b, init_adjs, save_dir, title="", threshold=0.5):
    """
    Evaluate the graph data in torch tensor.
    """
    delta = sample_b - test_adj_b
    init_delta = init_adjs - test_adj_b
    round_init_adjs = torch.where(init_adjs < threshold, torch.zeros_like(init_adjs), torch.ones_like(init_adjs))
    round_init_delta = round_init_adjs - test_adj_b
    logging.info(f"sample delta_norm_mean: {delta.norm(dim=[1, 2]).mean().item():.3e} "
                 f"| init delta_norm_mean: {init_delta.norm(dim=[1, 2]).mean().item():.3e}"
                 f"| round init delta_norm_mean: {round_init_delta.norm(dim=[1, 2]).mean().item():.3e}")

    plot_graphs_adj(sample_b,
                    node_num=test_adj_b.sum(-1).gt(1e-5).sum(-1).cpu().numpy(),
                    title=title,
                    save_dir=save_dir)
