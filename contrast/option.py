import argparse
import os

from contrast import resnet
from contrast.util import MyHelpFormatter

model_names = sorted(name for name in resnet.__all__ if name.islower() and callable(resnet.__dict__[name]))


def parse_option(stage='pre-train'):
    parser = argparse.ArgumentParser(f'contrast {stage} stage', formatter_class=MyHelpFormatter)

    # dataset
    parser.add_argument('--data-dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--crop', type=float, default=0.2 if stage == 'pre-train' else 0.08, help='minimum crop')
    parser.add_argument('--crop-ratio', nargs=2, type=float, default=[3. / 4., 4. / 3.], help='ratio of crop')
    parser.add_argument('--aug', type=str, default='NULL',
                        choices=['NULL', 'InstDisc', 'MoCov2', 'SimCLR', 'RandAug', 'BYOL', 'val'],
                        help='which augmentation to use.')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='cache mode: no for no cache, full for cache all data, part for only cache part of data')
    parser.add_argument('--dataset', type=str, default='ImageNet', choices=['ImageNet', "bdd100k"], help='dataset type')
    parser.add_argument('--ann-file', type=str, default='', help='annotation file')
    parser.add_argument('--image-size', nargs=2, type=int, default=[224, 224], help='image crop size')
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers per GPU to use')
    parser.add_argument('--n-frames', type=int, default=1, help='num of frames to load if dataset is video')

    # RAFT
    parser.add_argument('--use_flow', action='store_true')
    parser.add_argument('--flow_model', default="", help="raft model path")
    parser.add_argument('--flow_up', action='store_true')
    parser.add_argument('--alpha1', type=float, default=None, help='alpha1 for cycle consistency')
    parser.add_argument('--alpha2', type=float, default=None, help='alpha2 for cycle consistency')
    parser.add_argument('--flow_cat_norm', action='store_true', help='normalize flow after calc flow')
    parser.add_argument('--flow_bs', type=int, default=None, help='flow num for calc optical flow / 1gpu')
    parser.add_argument('--use_flow_frames', action='store_true', help='use flow frames which is less than n_frame flow')
    parser.add_argument('--use_flow_file', action='store_true', help='get flow value by loading file')
    parser.add_argument('--flow_root', type=str, default='', help='dataset for flow director')
    parser.add_argument('--fwd_name', type=str, default='', help='flow fwd path name')
    parser.add_argument('--bwd_name', type=str, default='', help='flow bwd path name')

    if stage == 'linear':
        parser.add_argument('--total-batch-size', type=int, default=256, help='total train batch size for all GPU')
    else:
        parser.add_argument('--batch-size', type=int, default=64, help='batch_size for single gpu')

    # model
    parser.add_argument('--arch', type=str, default='resnet50', choices=model_names, help="backbone architecture")
    if stage == 'pre-train':
        parser.add_argument('--model', type=str, required=True, help='which model to use')
        parser.add_argument('--feature-dim', type=int, default=256, help='feature dimension')
        parser.add_argument('--head-type', type=str, default='mlp_head', help='choose head type')

    # optimization
    if stage == 'pre-train':
        parser.add_argument('--base-learning-rate', '--base-lr', type=float, default=0.03,
                            help='base learning when batch size = 256. final lr is determined by linear scale')
    else:
        parser.add_argument('--learning-rate', type=float, default=30, help='learning rate')

    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars'], default='sgd',
                        help='for optimizer choice.')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4 if stage == 'pre-train' else 0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--debug-epochs', type=int, default=None, help='debug number of training epochs')
    parser.add_argument('--verbose', action='store_true', help='debug print for raft')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--no_benchmark', action='store_true', help='torch.backends.cudnn.benchmark set to False')

    # misc
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')
    parser.add_argument('--auto-resume', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--print-freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    if stage == 'linear':
        parser.add_argument('--pretrained-model', type=str, required=True, help="pretrained model path")
        parser.add_argument('-e', '--eval', action='store_true', help='only evaluate')
    else:
        parser.add_argument('--pretrained-model', type=str, default="", help="pretrained model path")

    # PixPro arguments
    if stage == 'pre-train':
        parser.add_argument('--pixpro-p', type=float, default=1.)
        parser.add_argument('--pixpro-momentum', type=float, default=0.99, help='momentume parameter used in MoCo and InstDisc')
        parser.add_argument('--pixpro-pos-ratio', type=float, default=0.7, help='maximum distance ratio of positive patches')
        parser.add_argument('--pixpro-neg-ratio', type=float, default=1.0, help='minimum distance ratio of negative patches')
        parser.add_argument('--pixpro-neg-loss-weight', type=float, default=1.0, help='negative loss weight')
        parser.add_argument('--pixpro-ins-loss-weight', type=float, default=0., help='loss weight for instance branch')
        parser.add_argument('--pixpro-clamp-value', type=float, default=0.)
        parser.add_argument('--pixpro-transform-layer', type=int, default=0)

    args = parser.parse_args()

    if args.flow_model != "":
        base_name = os.path.basename(args.flow_model)
        if "small" in base_name:
            args.small = True
        args.mixed_precision = False

    if args.image_size[0] == args.image_size[1]:
        args.image_size = args.image_size[0]

    if args.debug_epochs is None:
        args.debug_epochs = args.epochs + 1

    return args
