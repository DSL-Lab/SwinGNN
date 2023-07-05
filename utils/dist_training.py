import logging
import os
import pdb

import torch
from torch import distributed as dist, nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedHelper(object):
    def __init__(self, flag_dp, flag_ddp, ddp_gpu_ids, init_method):
        self.flag_dp = flag_dp
        self.flag_ddp = flag_ddp
        self.ddp_gpu_ids = ddp_gpu_ids
        self.init_method = init_method

        if (self.flag_dp or self.flag_ddp) and ddp_gpu_ids is None:
            assert torch.cuda.device_count() > 1, "Number of GPU must be more than one to use distributed learning!"
        assert not all((flag_dp, flag_ddp)), \
            "Flag DP ({:}) and flag DDP ({:}) cannot be both true!".format(flag_dp, flag_ddp)

        self.gpu_name = 'dummy'
        self.init_ddp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_ddp(self):
        """
        Initialize DDP distributed training if necessary.
        Note: we have to initialize DDP mode before initialize the logging file, otherwise the multiple DDP
        processes' loggings will interfere with each other.
        """
        print("Number of available GPU to use: {}".format(torch.cuda.device_count()))
        if self.flag_ddp:
            self.init_ddp_backend()
            self.gpu_name = torch.cuda.get_device_name()
            print("Setup DDP for process {:d} using GPUs {} (ID) with NCCL backend. GPU for this process: {:s}".format(
                os.getpid(), self.ddp_gpu_ids, self.gpu_name))
        elif self.flag_dp:
            gpu_specs = [torch.cuda.get_device_name(i_gpu) for i_gpu in range(torch.cuda.device_count())]
            self.gpu_name = ','.join(gpu_specs)
            print("Setup DP using {:d} GPUs, specs: {:s}.".format(torch.cuda.device_count(), self.gpu_name))
        else:
            self.gpu_name = torch.cuda.get_device_name()
            print("Single GPU mode, specs: {:s}.".format(self.gpu_name))

    def init_ddp_backend(self):
        """
        Start DDP engine using NCCL backend.
        """
        ddp_status, env_dict = self.get_ddp_status()
        local_rank = env_dict['LOCAL_RANK']

        if self.ddp_gpu_ids is not None:
            assert isinstance(self.ddp_gpu_ids, list)
            num_gpus = len(self.ddp_gpu_ids)
            gpu_id = int(self.ddp_gpu_ids[local_rank % num_gpus])
            torch.cuda.set_device(gpu_id)  # set single gpu device per process
        else:
            torch.cuda.set_device(local_rank)  # set single gpu device per process
        dist.init_process_group(backend="nccl", init_method=self.init_method, rank=env_dict['WORLD_RANK'], world_size=env_dict['WORLD_SIZE'])

    def dist_adapt_model(self, model):
        """
        Setup distributed learning for network.
        """
        logging.info("Adapt the model for distributed training...")
        if self.flag_ddp:
            # DDP
            model = DDP(model.cuda(), device_ids=[torch.cuda.current_device()])  # single CUDA device per process
            # model = DDP(model.cuda(), device_ids=[torch.cuda.current_device()], ind_unused_parameters = True)
            logging.info("Distributed ON. Mode: DDP. Backend: {:s}, Rank: {:d} / World size: {:d}. "
                         "Current device: {}, spec: {}".format(
                          dist.get_backend(), dist.get_rank(), dist.get_world_size(),
                          torch.cuda.current_device(), self.gpu_name))
        elif self.flag_dp:
            # DP
            model = nn.DataParallel(model)
            model.to(torch.device("cuda"))  # multiple devices per process, controlled by CUDA_VISIBLE_DEVICES
            logging.info("Distributed ON. Mode: DP. Number of available GPU to use: {}, specs: {}".format(
                          torch.cuda.device_count(), self.gpu_name))
        else:
            # single GPU
            logging.info("Distributed OFF. Single-GPU training, specs: {}.".format(self.gpu_name))

        return model

    def ddp_sync(self):
        if self.flag_ddp and dist.is_initialized():
            dist.barrier()
        else:
            pass

    def clean_up(self):
        self.ddp_sync()
        if self.flag_ddp and dist.is_initialized():
            dist.destroy_process_group()
        else:
            pass

    @staticmethod
    def get_ddp_status():
        """
        Get DDP-related env. parameters.
        """
        if 'LOCAL_RANK' in os.environ:
            # Environment variables set by torch.distributed.launch or torchrun
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            world_rank = int(os.environ['RANK'])
        elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
            # Environment variables set by mpirun
            local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        else:
            raise NotImplementedError

        env_dict = {
            'MASTER_ADDR': os.environ['MASTER_ADDR'],
            'MASTER_PORT': os.environ['MASTER_PORT'],
            'LOCAL_RANK': local_rank,
            'WORLD_SIZE': world_size,
            'WORLD_RANK': world_rank,
        }
        ddp_status = "Process PID: {}. DDP setup: {} ".format(os.getpid(), env_dict)
        return ddp_status, env_dict

    @property
    def is_ddp(self):
        """
        DDP flag.
        """
        return self.flag_ddp

    @property
    def is_dp(self):
        """
        DP flag.
        """
        return self.flag_dp

    @property
    def is_distributed(self):
        """
        Distributed learning flag.
        """
        return self.flag_dp or self.flag_ddp


# Independent function helpers
def get_ddp_save_flag():
    """
    Return saving flag for DDP mode, only rank 0 process makes the output.
    """
    flag_save = True
    if dist.is_initialized():
        if dist.get_rank() != 0:
            flag_save = False
    return flag_save


def dist_save_model(data_to_save, to_save_path):
    """
    Wrapper to save based on DDP status (for main process only).
    """
    if get_ddp_save_flag():
        torch.save(data_to_save, to_save_path)


def gather_tensors(in_tensor, cat_dim, device):
    """
    Gather tensors from all GPU processes.
    :param in_tensor:   input tensor, which is distributed across GPUs
    :param cat_dim:     dimension to concatenate
    :param device:      device to gather tensors, usually GPU to enable NVCC backend
    :return:
    """
    if hasattr(dist, 'all_gather_into_tensor'):
        # new API available after v1.13.0
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor
        _shape_in_tensor = list(in_tensor.shape)
        _shape_out_tensor = _shape_in_tensor.copy()
        _shape_out_tensor[cat_dim] *= dist.get_world_size()  # enlarge tensor size along the concat dimension

        out_tensor = torch.zeros(_shape_out_tensor, dtype=in_tensor.dtype, device=device)
        dist.all_gather_into_tensor(out_tensor, in_tensor.to(device).contiguous())  # list of tensors from X GPUs
    elif hasattr(dist, 'all_gather'):
        # stable API
        in_tensor = in_tensor.to(device).contiguous()  # turn into CUDA tensor
        out_tensor = [torch.zeros_like(in_tensor) for _ in range(dist.get_world_size())]  # must be a list of tensors
        dist.all_gather(out_tensor, in_tensor)  # list of tensors from X GPUs
        out_tensor = torch.cat(out_tensor, dim=cat_dim)  # [*, X, *] gathered from X GPUs
    else:
        raise NotImplementedError
    return out_tensor
