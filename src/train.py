from __future__ import annotations

import argparse

import torch

from features import FEATURE_TYPES
from model_registry import model_choices
from utils import seed_everything


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train and evaluate WFCAT and WF baselines')
    parser.add_argument('--model', choices=model_choices(), default='wfcat', help='model backbone')
    parser.add_argument('--feature-type', choices=FEATURE_TYPES, default='iat', help='input feature type')
    parser.add_argument('--iat-bins', type=int, default=9, help='number of IAT histogram bins per direction')
    parser.add_argument('--time-window', type=float, default=0.044,
                        help='time window size for IAT histogram features, in seconds')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='train or test')
    parser.add_argument('--defense', choices=['other', 'trafficsliver'], default='other', help='dataset parser variant')

    parser.add_argument('--data-path', type=str, required=True, help='directory containing .cell traces')
    parser.add_argument('--model-path', type=str, default='./checkpoints', help='output checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path for test mode')
    parser.add_argument('--pretrained', type=str, default=None, help='optional pretrained checkpoint')
    parser.add_argument('--exist-ok', action='store_true', default=False, help='reuse checkpoint directory if it exists')
    parser.add_argument('--nosave', action='store_true', default=False, help='do not save model checkpoints')

    parser.add_argument('--suffix', type=str, default='.cell', help='trace file suffix')
    parser.add_argument('--mon-classes', default=100, type=int, help='number of monitored classes')
    parser.add_argument('--mon-inst', default=100, type=int, help='number of monitored instances per class')
    parser.add_argument('--mon-inst-train', default=-1, type=int,
                        help='monitored instances per class for training, -1 uses all')
    parser.add_argument('--unmon-inst', default=10000, type=int, help='number of unmonitored instances')
    parser.add_argument('--unmon-inst-train', default=-1, type=int,
                        help='unmonitored training instances, -1 uses all')
    parser.add_argument('--open-world', default=False, action='store_true', help='run open-world evaluation')
    parser.add_argument('--seq-length', default=1800, type=int, help='feature sequence length')
    parser.add_argument('--page-per-class', default=1, type=int, help='pages per monitored site class')

    parser.add_argument('--num-kernels', type=int, default=4, help='number of multi-scale kernels for WFCAT')
    parser.add_argument('--n-tam', default=2, type=int, help='number of TAM channels for tam_plus')

    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='optimizer weight decay')
    parser.add_argument('--label-smoothing', '--ls', type=float, default=0.0, help='cross-entropy label smoothing')
    parser.add_argument('-j', '--workers', default=10, type=int, help='data loader workers')

    parser.add_argument('--use-gpu', type=_str2bool, default=True, help='use CUDA if available')
    parser.add_argument('--cpu', action='store_false', dest='use_gpu', help='force CPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id used when CUDA_VISIBLE_DEVICES is unset')
    parser.add_argument('--use-multi-gpu', action='store_true', default=False, help='use DataParallel')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='visible GPU ids for multi-GPU mode')
    parser.add_argument('--not-amp', action='store_true', default=False, help='disable mixed precision training')

    parser.add_argument('--bandwidth-split', choices=['none', 'fast', 'slow'], default='none',
                        help='train on high-bandwidth or low-bandwidth traces and test on the rest')
    parser.add_argument('--bandwidth-ratio', type=float, default=0.9, help='training ratio for bandwidth split')
    parser.add_argument('--one-fold', action='store_true', default=False, help='run only the first stratified fold')
    parser.add_argument('--test-every-epoch', action='store_true', default=False, help='evaluate test set each epoch')
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed logs')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    return parser.parse_args()


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {'true', '1', 'yes', 'y'}


def main():
    args = parse_arguments()
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    seed_everything(args.seed)
    if args.verbose:
        print(args)
    from trainer import WFExperiment
    WFExperiment(args).run(one_fold_only=args.one_fold)


if __name__ == '__main__':
    main()
