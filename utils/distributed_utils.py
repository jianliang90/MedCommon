import os

import torch
import torch.distributed as dist

class DistributedUtils:
    def __init__(self):
        pass
    
    @staticmethod
    def init_distributed_mode(args):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            print('Not using distributed mode')
            args.distributed = False
            return

        args.distributed = True

        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        dist.barrier()

    
    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    @staticmethod
    def is_dist_avail_and_initialized():
        """检查是否支持分布式环境"""
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    @staticmethod
    def get_world_size():
        if not DistributedUtils.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    @staticmethod
    def get_rank():
        if not DistributedUtils.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    @staticmethod
    def is_main_process():
        return get_rank() == 0

    @staticmethod
    def reduce_value(value, average=True):
        world_size = get_world_size()
        if world_size < 2:  # 单GPU的情况
            return value

        with torch.no_grad():
            dist.all_reduce(value)
            if average:
                value /= world_size

            return value


