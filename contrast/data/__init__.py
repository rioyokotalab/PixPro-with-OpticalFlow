import os

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .transform import get_transform
from .dataset import ImageFolder


def get_loader(aug_type, args, two_crop=False, prefix='train', return_coord=False):
    transform = get_transform(aug_type, args.crop, args.image_size, args.crop_ratio)

    # use flow file
    fwd_path, bwd_path = "", ""
    if args.use_flow_file:
        flow_root = args.flow_root
        if flow_root is None or flow_root == "":
            flow_root = os.path.dirname(args.data_dir)
            flow_root = os.path.join(flow_root, "flow", "pth")
        flow_root = os.path.join(flow_root, prefix)
        args.flow_root = flow_root

        fwd_name, bwd_name = args.fwd_name, args.bwd_name
        if fwd_name is None or fwd_name == "":
            fwd_name = "forward"
        if bwd_name is None or bwd_name == "":
            bwd_name = "backward"
        args.fwd_name, args.bwd_name = fwd_name, bwd_name

        fwd_path = os.path.join(flow_root, fwd_name)
        bwd_path = os.path.join(flow_root, bwd_name)
    flow_file_root_list = [fwd_path, bwd_path]

    # dataset
    if args.zip:
        if args.dataset == 'ImageNet' or args.dataset == "bdd100k":
            train_ann_file = prefix + "_map.txt"
            train_prefix = prefix + ".zip@/"
        else:
            raise NotImplementedError('Dataset {} is not supported. We only support ImageNet now'.format(args.dataset))

        train_dataset = ImageFolder(
            args.data_dir,
            train_ann_file,
            train_prefix,
            transform,
            two_crop=two_crop,
            cache_mode=args.cache_mode,
            dataset=args.dataset,
            return_coord=return_coord,
            n_frames=args.n_frames,
            flow_file_root_list=flow_file_root_list,
            use_flow_frames=args.use_flow_frames,
            debug=args.debug)
    else:
        train_folder = os.path.join(args.data_dir, prefix)
        train_dataset = ImageFolder(
            train_folder,
            transform=transform,
            two_crop=two_crop,
            dataset=args.dataset,
            return_coord=return_coord,
            n_frames=args.n_frames,
            flow_file_root_list=flow_file_root_list,
            use_flow_frames=args.use_flow_frames,
            debug=args.debug)

    # sampler
    indices = np.arange(dist.get_rank(), len(train_dataset), dist.get_world_size())
    if args.zip and args.cache_mode == 'part':
        sampler = SubsetRandomSampler(indices)
    else:
        sampler = DistributedSampler(train_dataset)

    # dataloader
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True)
