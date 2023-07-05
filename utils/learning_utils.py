import logging
import os
import pdb

import numpy as np
import torch
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer

from ema_pytorch import EMA
from model.unet.unet_edm import DhariwalUNet, SongUNet
from model.precond.precond import Precond, NodeAdjPrecond
from model.swin_gnn.swin_gnn import SwinGNN, NodeAdjSwinGNN
from model.self_cond.self_cond_wrapper import SelfCondWrapper

from loss.rainbow_loss import RainbowLoss, NodeAdjRainbowLoss
from runner.objectives.diffusion import DiffusionObjectiveGenerator
from runner.objectives.score_matching import ScoreMatchingObjectiveGenerator
from runner.objectives.edm import EDMObjectiveGenerator, NodeAdjEDMObjectiveGenerator
from utils.dist_training import get_ddp_save_flag
from utils.sampling_utils import load_model


def get_training_objective_generator(config):
    """
    Get training objective generator.
    """
    if config.mcmc.name == "score":
        train_obj_gen = ScoreMatchingObjectiveGenerator(sigma_num_slices=config.mcmc.sigmas.num_slices,
                                                        sigma_min=config.mcmc.sigmas.min,
                                                        sigma_max=config.mcmc.sigmas.max,
                                                        sigma_preset=None,
                                                        dev=config.dev)
    elif config.mcmc.name == "diffusion":
        train_obj_gen = DiffusionObjectiveGenerator(max_steps=config.mcmc.betas.max_steps,
                                                    beta_min=config.mcmc.betas.min,
                                                    beta_max=config.mcmc.betas.max,
                                                    schedule=config.mcmc.betas.schedule,
                                                    pred_target=config.mcmc.pred_target,
                                                    other_params=config.mcmc,
                                                    dev=config.dev)
    elif config.mcmc.name == "edm":
        if config.flag_mol:
            train_obj_gen = NodeAdjEDMObjectiveGenerator(precond=config.mcmc.precond,
                                                         sigma_dist=config.mcmc.sigma_dist,
                                                         other_params=config.mcmc,
                                                         dev=config.dev,
                                                         symmetric_noise=True)
        else:
            train_obj_gen = EDMObjectiveGenerator(precond=config.mcmc.precond,
                                                  sigma_dist=config.mcmc.sigma_dist,
                                                  other_params=config.mcmc,
                                                  dev=config.dev)
    else:
        raise NotImplementedError
    return train_obj_gen


def get_network(config, dist_helper):
    """
    Configure the neural network.
    """
    model_config = config.model
    feature_nums = model_config.feature_dims if 'feature_dims' in model_config else [0]

    plot_save_dir = os.path.join(config.logdir, 'training_plot')
    if get_ddp_save_flag():
        os.makedirs(plot_save_dir, exist_ok=True)
    if config.model.name == 'unet':
        # ADM Unet architecture
        attn_resolutions = [32, 16, 8] if not hasattr(model_config, 'attn_resolutions') else model_config.attn_resolutions
        attn_resolutions = [] if attn_resolutions is None else attn_resolutions
        num_blocks = 3 if not hasattr(model_config, 'num_blocks') else model_config.num_blocks
        denoising_model = DhariwalUNet(
                img_resolution=config.dataset.max_node_num,     # Image resolution at input/output.
                in_channels=1,                                  # Number of color channels at input.
                out_channels=1,                                 # Number of color channels at output.
                model_channels=feature_nums[-1],                # Base multiplier for the number of channels.
                channel_mult=model_config.feature_multipliers,  # Per-resolution multipliers for the number of channels.
                self_condition=config.train.self_cond,           # Self-conditioning
                attn_resolutions=attn_resolutions,
                num_blocks=num_blocks,
                ).to(config.dev)

        # vanilla DDPM Unet
        # denoising_model = Unet(
        #     dim=feature_nums[-1],
        #     dim_mults=tuple(model_config.feature_multipliers),
        #     channels=1,
        # ).to(config.dev)

        # DDPM++ Unet architecture
        # denoising_model = SongUNet(
        #         img_resolution=config.dataset.max_node_num,     # Image resolution at input/output.
        #         in_channels=1,                                  # Number of color channels at input.
        #         out_channels=1,                                 # Number of color channels at output.
        #         model_channels=feature_nums[-1],                # Base multiplier for the number of channels.
        #         channel_mult=model_config.feature_multipliers,  # Per-resolution multipliers for the number of channels.
        #         ).to(config.dev)
    elif config.model.name == 'swin_gnn':
        if config.flag_mol:
            # with node and edge attributes
            in_chans, out_chans_adj, out_chans_node = get_model_input_output_channels(config)

            denoising_model = NodeAdjSwinGNN(
                img_size=config.dataset.max_node_num,
                in_chans=in_chans,
                # patch_size=4,
                # embed_dim=96,
                # depths=[2, 2, 6, 2],
                patch_size=model_config.patch_size,
                embed_dim=feature_nums[-1],
                depths=model_config.depths,
                num_heads=[3, 6, 12, 24],
                window_size=model_config.window_size,
                mlp_ratio=4.,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                self_condition=config.train.self_cond,
                symmetric_noise=True,
                out_chans_adj=out_chans_adj,
                out_chans_node=out_chans_node
            ).to(config.dev)
        else:
            # without node and edge attributes
            denoising_model = SwinGNN(
                img_size=config.dataset.max_node_num,
                in_chans=1,
                # patch_size=4,
                # embed_dim=96,
                # depths=[2, 2, 6, 2],
                patch_size=model_config.patch_size,
                embed_dim=feature_nums[-1],
                depths=model_config.depths,
                num_heads=[3, 6, 12, 24],
                window_size=model_config.window_size,
                mlp_ratio=4.,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                self_condition=config.train.self_cond
                ).to(config.dev)
    else:
        raise ValueError(f'Unknown model name {config.model.name}')

    # EDM preconditioning module adaptation
    if config.mcmc.name == 'edm':
        if config.flag_mol:
            denoising_model = NodeAdjPrecond(precond=config.mcmc.precond,
                                             model=denoising_model,
                                             self_condition=config.train.self_cond,
                                             symmetric_noise=True)
        else:
            denoising_model = Precond(precond=config.mcmc.precond,
                                      model=denoising_model,
                                      self_condition=config.train.self_cond)

    # non-EDM self-conditioning nn.Module wrapper
    # EDM doesn't need this as its precond layer is already an nn.Module layer
    if config.mcmc.name != 'edm' and config.train.self_cond:
        denoising_model = SelfCondWrapper(model=denoising_model, self_condition=config.train.self_cond)

    # DEBUG: plot model intermediate states
    denoising_model.plot_save_dir = plot_save_dir

    # count model parameters
    logging.info('model: ' + str(denoising_model))
    param_string, total_params, total_trainable_params = count_model_params(denoising_model)
    logging.info(f"Parameters: \n{param_string}")
    logging.info(f"Parameters Count: {total_params:,}, Trainable: {total_trainable_params:,}")

    # load checkpoint to resume training
    if config.train.resume is not None:
        logging.info("Resuming training from checkpoint: {:s}".format(config.train.resume))
        ckp_data = torch.load(config.train.resume)
        denoising_model = load_model(ckp_data, denoising_model, 'model')

    # adapt to distributed training
    if dist_helper.is_distributed:
        denoising_model = dist_helper.dist_adapt_model(denoising_model)
    else:
        logging.info("Distributed OFF. Single-GPU training.")

    return denoising_model


def get_model_input_output_channels(config):
    def _get_in_out_chan_num(encoding, is_node_attr):
        # for [i, j] entry, we concat node i and node j types and edge [i, j] type
        if config.flag_mol:
            if config.dataset.name == 'qm9':
                raw_num_node_type, raw_num_adj_type = 4, 4
            elif config.dataset.name == 'zinc250k':
                raw_num_node_type, raw_num_adj_type = 9, 4
            else:
                raise NotImplementedError

            if encoding == 'one_hot':
                num_node_type, num_adj_type = raw_num_node_type, raw_num_adj_type
            elif encoding == 'bits':
                num_node_type, num_adj_type = np.ceil(np.log2(raw_num_node_type)).astype(int), np.ceil(np.log2(raw_num_adj_type)).astype(int)
            elif encoding == 'ddpm':
                num_node_type, num_adj_type = 1, 1
            else:
                raise NotImplementedError
            _in_chans = num_node_type * 2 if is_node_attr else num_adj_type
            _out_chans = num_node_type if is_node_attr else num_adj_type
        else:
            raise NotImplementedError
        return _in_chans, _out_chans

    in_chans_node, out_chans_node = _get_in_out_chan_num(config.train.node_encoding, is_node_attr=True)
    in_chans_adj, out_chans_adj = _get_in_out_chan_num(config.train.edge_encoding, is_node_attr=False)

    in_chans = in_chans_node + in_chans_adj  # inputs are concatenated, outputs are separated

    return in_chans, out_chans_adj, out_chans_node


def count_model_params(model):
    """
    Go through the model parameters
    """
    param_strings = []
    max_string_len = 126
    for name, param in model.named_parameters():
        if param.requires_grad:
            line = '.' * max(0, max_string_len - len(name) - len(str(param.size())))
            param_strings.append(f"{name} {line} {param.size()}")
    param_string = '\n'.join(param_strings)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_string, total_params, total_trainable_params


def get_optimizer(model, config, dist_helper):
    """
    Configure the optimizer.
    """
    if dist_helper.is_ddp:
        optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                            optimizer_class=torch.optim.Adam,
                                            lr=config.train.lr_init,
                                            betas=(0.9, 0.999), eps=1e-8,
                                            weight_decay=config.train.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=config.train.lr_init,
                               betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_dacey)
    return optimizer, scheduler


def get_ema_helper(config, model):
    """
    Setup exponential moving average training helper.
    """
    flag_ema = False
    ema_coef = config.train.ema_coef
    if isinstance(ema_coef, list):
        flag_ema = True
    if isinstance(ema_coef, float):
        flag_ema = config.train.ema_coef < 1
    if flag_ema:
        ema_coef = [ema_coef] if isinstance(ema_coef, float) else ema_coef
        assert isinstance(ema_coef, list)
        ema_helper = []
        for coef in sorted(ema_coef):
            ema = EMA(model=model, beta=coef, update_every=1, update_after_step=0, inv_gamma=1, power=1)
            ema_helper.append(ema)
        logging.info("Exponential moving average is ON. Coefficient: {}".format(ema_coef))
    else:
        ema_helper = None
        logging.info("Exponential moving average is OFF.")
    return ema_helper


def get_rainbow_loss(config):
    """
    Construct all-in-one training loss wrapper.
    """

    if config.flag_mol:
        loss_func = NodeAdjRainbowLoss(edge_loss_weight=config.train.edge_loss_weight,
                                       node_loss_weight=config.train.node_loss_weight,
                                       flag_reweight=config.train.reweight_entry,
                                       objective=config.mcmc.name)
    else:
        loss_func = RainbowLoss(regression_loss_weight=1.0,  # default
                                flag_reweight=config.train.reweight_entry,
                                objective=config.mcmc.name)

    logging.info("Loss weight: denoising regression loss: {:.2f}".format(1.0))

    logging.info("Loss reweight based on zero/one entries: {}.".format(config.train.reweight_entry))
    return loss_func
