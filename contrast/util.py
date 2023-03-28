import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .flow import upflow8


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class MyHelpFormatter(argparse.MetavarTypeHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


# optical flow utils
@torch.no_grad()
def calc_optical_flow(imgs, flow_model, up=False, verbose=False):
    num_img = len(imgs)
    assert num_img >= 2
    if verbose:
        orig_im1, orig_im2 = imgs[0].clone(), imgs[-1].clone()
    i = 1 if up else 0
    flow_model.eval()
    flow_fwds = torch.stack([
        flow_model(img0, img1, upsample=False, test_mode=True)[i]
        for img0, img1 in zip(imgs[:-1], imgs[1:])
    ])
    flow_bwds = torch.stack([
        flow_model(img0, img1, upsample=False, test_mode=True)[i]
        for img0, img1 in zip(imgs[1:][::-1], imgs[:-1][::-1])
    ])
    flow_fwds = flow_fwds.cuda()
    flow_bwds = flow_bwds.cuda()

    if verbose:
        rank = dist.get_rank()
        print(f"rank: {rank} orig_im1: {orig_im1.dtype} orig_im2: {orig_im2.dtype}")
        print(f"rank: {rank} orig_im1: {orig_im1.shape}", orig_im1.tolist())
        print(f"rank: {rank} orig_im2: {orig_im2.shape}", orig_im2.tolist())
        print(f"rank: {rank} flow_fwds: {flow_fwds.shape}", flow_fwds.tolist())
        print(f"rank: {rank} flow_bwds: {flow_bwds.shape}", flow_bwds.tolist())
    return flow_fwds, flow_bwds


def all_concat_flow(flow_fwds, flow_bwds, is_norm=False, use_flow_frames=True):
    if not use_flow_frames:
        flow_fwd = concat_flow(flow_fwds, is_norm)
        flow_bwd = concat_flow(flow_bwds, is_norm)
        return flow_fwd, flow_bwd

    num_flow = flow_bwds.shape[0]
    tmp_fwd_list, tmp_bwd_list = [], []
    for i in range(num_flow):
        flow_frame_num = i + 1
        l_num_flow = num_flow - flow_frame_num + 1
        for fwd_s_idx in range(l_num_flow):
            fwd_next_idx = fwd_s_idx + flow_frame_num
            bwd_next_idx = num_flow - fwd_s_idx
            bwd_s_idx = bwd_next_idx - flow_frame_num
            tmp_fwd = concat_flow(flow_fwds[fwd_s_idx:fwd_next_idx], is_norm)
            tmp_bwd = concat_flow(flow_bwds[bwd_s_idx:bwd_next_idx], is_norm)
            tmp_fwd_list.append(tmp_fwd)
            tmp_bwd_list.append(tmp_bwd)
    flow_fwd = torch.stack(tmp_fwd_list)
    flow_bwd = torch.stack(tmp_bwd_list)
    return flow_fwd, flow_bwd


def mem_reduce_calc_optical_flow(orig_imgs, flow_model, args):
    orig_im1 = orig_imgs[0]
    num_img = len(orig_imgs)
    bs = orig_im1.shape[0]
    is_use_flow_frames = args.use_flow_frames and num_img > 2
    # to reduce memory usage
    flow_fwds, flow_bwds = [], []
    flow_bs = 8
    if hasattr(args, "flow_bs") and args.flow_bs is not None:
        flow_bs = args.flow_bs
    s_index = bs % flow_bs
    if s_index != 0:
        l_orig_imgs = [im[0:s_index] for im in orig_imgs]
        flow_fwd, flow_bwd = calc_optical_flow(l_orig_imgs, flow_model,
                                               up=args.flow_up, verbose=args.verbose)
        flow_fwd, flow_bwd = all_concat_flow(flow_fwd, flow_bwd,
                                             is_norm=args.flow_cat_norm,
                                             use_flow_frames=is_use_flow_frames)
        flow_fwds.append(flow_fwd)
        flow_bwds.append(flow_bwd)
    for i in range(s_index, bs, flow_bs):
        if i + flow_bs > bs:
            break
        l_orig_imgs = [im[i:i+flow_bs] for im in orig_imgs]
        flow_fwd, flow_bwd = calc_optical_flow(l_orig_imgs, flow_model,
                                               up=args.flow_up, verbose=args.verbose)
        flow_fwd, flow_bwd = all_concat_flow(flow_fwd, flow_bwd,
                                             is_norm=args.flow_cat_norm,
                                             use_flow_frames=is_use_flow_frames)
        flow_fwds.append(flow_fwd)
        flow_bwds.append(flow_bwd)
    ndim = flow_fwds[0].ndim
    assert ndim == 4 or ndim == 5
    cat_dim = 1 if ndim == 5 else 0
    flow_fwd = torch.cat(flow_fwds, dim=cat_dim)
    flow_bwd = torch.cat(flow_bwds, dim=cat_dim)
    # flow_fwd, flow_bwd = calc_optical_flow(orig_im1, orig_im2, flow_model)
    flow_fwd = flow_fwd.cuda()
    flow_bwd = flow_bwd.cuda()
    if ndim == 4:
        flow_fwd = flow_fwd.unsqueeze(0)
        flow_bwd = flow_bwd.unsqueeze(0)
    return flow_fwd, flow_bwd


@torch.no_grad()
def apply_optical_flow(data, flow_model, args):
    orig_imgs_tmp = data[6]
    size, num_img = orig_imgs_tmp[0][0], orig_imgs_tmp[1][0].item()
    is_mask_flow = args.alpha1 is not None and args.alpha2 is not None
    is_use_flow_frames = args.use_flow_frames and num_img > 2
    if args.use_flow_file:
        _, flow_fwds, flow_bwds = data[5]
        # transpose nb, num, 2, h, w -> num, nb, 2, h, w
        flow_fwds = flow_fwds.permute(1, 0, 2, 3, 4)
        flow_bwds = flow_bwds.permute(1, 0, 2, 3, 4)
        if args.flow_up:
            num, nb, c, h, w = flow_fwds.shape
            flow_fwds = upflow8(flow_fwds.reshape(-1, c, h, w))
            flow_bwds = upflow8(flow_bwds.reshape(-1, c, h, w))
            _, new_c, new_h, new_w = flow_fwds.shape
            flow_fwds = flow_fwds.reshape(num, nb, new_c, new_h, new_w)
            flow_bwds = flow_bwds.reshape(num, nb, new_c, new_h, new_w)
        flow_fwd, flow_bwd = all_concat_flow(flow_fwds, flow_bwds,
                                             is_norm=args.flow_cat_norm,
                                             use_flow_frames=is_use_flow_frames)
        ndim = flow_fwd.ndim
        flow_fwd = flow_fwd.cuda()
        flow_bwd = flow_bwd.cuda()
        if ndim == 4:
            flow_fwd = flow_fwd.unsqueeze(0)
            flow_bwd = flow_bwd.unsqueeze(0)
    else:
        orig_imgs = orig_imgs_tmp[2:]
        # to reduce memory usage
        flow_fwd, flow_bwd = mem_reduce_calc_optical_flow(orig_imgs, flow_model, args)

    mask_fwd, mask_bwd = None, None
    if is_mask_flow:
        mask_fwd, mask_bwd,  = [], []
        if args.debug:
            flow_cycle_fwd, flow_cycle_bwd = [], []
        for l_flow_fwd, l_flow_bwd in zip(flow_fwd, flow_bwd):
            _, _, l_mask_fwd = forward_backward_consistency(l_flow_fwd, l_flow_bwd, alpha_1=args.alpha1, alpha_2=args.alpha2, is_norm=args.flow_cat_norm)
            _, _, l_mask_bwd = forward_backward_consistency(l_flow_bwd, l_flow_fwd, alpha_1=args.alpha1, alpha_2=args.alpha2, is_norm=args.flow_cat_norm)
            l_mask_fwd, l_flow_cycle_fwd = l_mask_fwd
            l_mask_bwd, l_flow_cycle_bwd = l_mask_bwd
            mask_fwd.append(l_mask_fwd)
            mask_bwd.append(l_mask_bwd)
            if args.debug:
                flow_cycle_fwd.append(l_flow_cycle_fwd)
                flow_cycle_bwd.append(l_flow_cycle_bwd)
        mask_fwd = torch.stack(mask_fwd)
        mask_bwd = torch.stack(mask_bwd)
        if args.debug:
            flow_cycle_fwd = torch.stack(flow_cycle_fwd)
            flow_cycle_bwd = torch.stack(flow_cycle_bwd)
            mask_fwd = [mask_fwd, flow_cycle_fwd]
            mask_bwd = [mask_bwd, flow_cycle_bwd]

    if args.flow_cat_norm:
        flow_fwd = torch.stack([denormalize_flow(f) for f in flow_fwd])
        flow_bwd = torch.stack([denormalize_flow(f) for f in flow_bwd])

    if not is_use_flow_frames:
        flow_fwd, flow_bwd = flow_fwd[-1], flow_bwd[-1]
        if mask_fwd is None or mask_bwd is None:
            if args.debug:
                mask_fwd, mask_bwd = [None, None], [None, None]
        else:
            if isinstance(mask_fwd, list):
                mask_fwd = [m[-1] for m in mask_fwd]
                mask_bwd = [m[-1] for m in mask_bwd]
            else:
                mask_fwd = mask_fwd[-1]
                mask_bwd = mask_bwd[-1]

    flow_fwd = [flow_fwd, size, mask_fwd]
    flow_bwd = [flow_bwd, size, mask_bwd]
    return flow_fwd, flow_bwd


# implement: https://arxiv.org/pdf/1711.07837.pdf
@torch.no_grad()
def forward_backward_consistency(flow_fwd, flow_bwd, coords0=None, alpha_1=0.01, alpha_2=0.5, is_norm=False):
    if alpha_1 is None or alpha_2 is None:
        return flow_fwd.clone(), flow_bwd.clone(), [None, None]
    flow_fwd = flow_fwd.detach()
    flow_bwd = flow_bwd.detach()
    if is_norm:
        flow_fwd_norm = flow_fwd.clone()
        flow_bwd_norm = flow_bwd.clone()
        flow_fwd = denormalize_flow(flow_fwd_norm)
        flow_bwd = denormalize_flow(flow_bwd_norm)
    else:
        flow_fwd_norm = normalize_flow(flow_fwd)
        flow_bwd_norm = normalize_flow(flow_bwd)

    if coords0 is None:
        nb, _, ht, wd = flow_fwd.shape
        coords0 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        coords0 = torch.stack(coords0[::-1], dim=0).float().repeat(nb, 1, 1, 1).to(flow_fwd.device)
        coords0_norm = normalize_coord(coords0)

    # coords1 = coords0 + flow_fwd
    # coords1_norm = normalize_coord(coords1)
    coords1_norm = coords0_norm + flow_fwd_norm
    mask = (torch.abs(coords1_norm[:, 0]) < 1) & (torch.abs(coords1_norm[:, 1]) < 1)

    flow_bwd_interpolate_norm = F.grid_sample(flow_bwd_norm, coords1_norm.permute(0, 2, 3, 1), align_corners=True)
    flow_cycle_norm = flow_fwd_norm + flow_bwd_interpolate_norm
    flow_cycle_tmp = flow_cycle_norm.clone()
    flow_bwd_interpolate_tmp = flow_bwd_interpolate_norm.clone()
    flow_fwd_tmp = flow_fwd_norm.clone()
    # flow_bwd_interpolate = F.grid_sample(flow_bwd, coords1_norm.permute(0, 2, 3, 1), align_corners=True)
    # flow_cycle = flow_fwd + flow_bwd_interpolate
    # flow_cycle_tmp = flow_cycle.clone()
    # flow_bwd_interpolate_tmp = flow_bwd_interpolate.clone()
    # flow_fwd_tmp = flow_fwd.clone()

    h, w = flow_fwd.shape[-2:]
    h, w = torch.tensor(h), torch.tensor(w)
    alpha_2 = alpha_2 / (torch.sqrt(h**2 + w**2).item())

    flow_cycle_abs_norm = (flow_cycle_tmp**2).sum(1)
    eps = alpha_1 * ((flow_fwd_tmp**2).sum(1) + (flow_bwd_interpolate_tmp**2).sum(1)) + alpha_2

    mask = mask & ((flow_cycle_abs_norm - eps) <= 0)
    return coords0_norm, coords1_norm, [mask, flow_cycle_tmp]


@torch.no_grad()
def concat_flow(flows, is_norm=False):
    num, nb, _, ht, wd = flows.shape
    if num == 1:
        flows_copy = flows.clone()
        out_flow = flows_copy[0]
        if is_norm:
            return normalize_flow(out_flow)
        return out_flow
    coords0 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords0 = torch.stack(coords0[::-1], dim=0).float().repeat(nb, 1, 1, 1)
    coords0 = coords0.to(flows.device)
    coords0_norm = normalize_coord(coords0)
    coords1 = coords0.clone()
    coords1_norm = coords0_norm.clone()
    for flow in flows:
        if is_norm:
            flow_norm = normalize_flow(flow)
            flow_interpolate_norm = F.grid_sample(flow_norm, coords1_norm.permute(0, 2, 3, 1), align_corners=True)
            coords1_norm = coords1_norm + flow_interpolate_norm
        else:
            coords1_norm_tmp = normalize_coord(coords1)
            flow_interpolate = F.grid_sample(flow, coords1_norm_tmp.permute(0, 2, 3, 1), align_corners=True)
            coords1 = coords1 + flow_interpolate

    if is_norm:
        out_flow = coords1_norm - coords0_norm
    else:
        out_flow = coords1 - coords0

    return out_flow


@torch.no_grad()
def normalize_coord(coords):
    _, _, ht, wd = coords.shape
    coords_norm = coords.clone()
    coords_norm[:, 0] = 2 * coords_norm[:, 0] / (wd - 1) - 1
    coords_norm[:, 1] = 2 * coords_norm[:, 1] / (ht - 1) - 1
    return coords_norm


@torch.no_grad()
def normalize_flow(flow):
    _, _, ht, wd = flow.shape
    flow_norm = flow.clone()
    flow_norm[:, 0] = 2 * flow_norm[:, 0] / (wd - 1)
    flow_norm[:, 1] = 2 * flow_norm[:, 1] / (ht - 1)
    return flow_norm


@torch.no_grad()
def denormalize_flow(flow_norm):
    _, _, ht, wd = flow_norm.shape
    flow = flow_norm.clone()
    flow[:, 0] = (flow[:, 0] * (wd - 1)) / 2
    flow[:, 1] = (flow[:, 1] * (ht - 1)) / 2
    return flow


@torch.no_grad()
def calc_mask_ratio(mask):
    if mask is None:
        return None
    mask_rev = torch.logical_not(mask)
    r = mask_rev.float().mean(-1).mean(-1)
    return r
