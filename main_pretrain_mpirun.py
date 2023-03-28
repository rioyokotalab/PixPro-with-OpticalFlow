import os

import torch
import torch.distributed as dist

from contrast.option import parse_option

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

from main_pretrain import main_prog


def dist_setup():
    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv("MASTER_PORT", default="8888")
    method = "tcp://{}:{}".format(master_addr, master_port)
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    local_size = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
    node_rank = int(os.getenv("OMPI_COMM_WORLD_NODE_RANK", "0"))
    host_port_str = f"host: {master_addr}, port: {master_port}"
    print(
        "rank: {}, world_size: {}, local_rank: {}, local_size: {}, node_rank: {}, {}"
        .format(rank, world_size, local_rank, local_size, node_rank, host_port_str))
    dist.init_process_group("nccl", init_method=method, rank=rank,
                            world_size=world_size)
    print("Rank: {}, Size: {}, Host: {} Port: {}".format(dist.get_rank(),
                                                         dist.get_world_size(),
                                                         master_addr, master_port))
    return local_rank


if __name__ == '__main__':
    opt = parse_option(stage='pre-train')

    if opt.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"

    local_rank = dist_setup()
    opt.local_rank = local_rank
    torch.cuda.set_device(opt.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')

    main_prog(opt)
