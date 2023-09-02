# Copyright (c) OpenMMLab. All rights reserved.
import sys
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.utils.prune as prune

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import argparse

from mmcv import Config
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from flops_counter import get_model_complexity_info
# from mmcv.cnn import get_model_complexity_info

from mmaction.models import build_model

# from tome.patch.mmseg import apply_patch as apply_tome

default_shapes = (3, 32, 384, 384)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--load-pretrain', action='store_true')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=None,
        help='input image size')
    parser.add_argument('--tome_r', type=float, default=0.)
    args = parser.parse_args()
    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def main():

    args = parse_args()

    if args.shape is None:
        args.shape = default_shapes

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    elif len(args.shape) == 4:
        # n, c, h, w = args.shape
        input_shape = tuple(args.shape)
    elif len(args.shape) == 5:
        # n, c, t, h, w = args.shape
        input_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # if not args.load_pretrain:
    #     cfg.model.pretrained = None

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # apply_tome(model.backbone)
    # model.backbone.r = args.tome_r
    if args.load_pretrain:
        model.init_weights()
        
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # parameters_to_prune = []
    # for i in range(24):
    #     parameters_to_prune.append((model.backbone.layers[i].ffn.layers[0][0], 'weight'))
    #     parameters_to_prune.append((model.backbone.layers[i].ffn.layers[1], 'weight'))
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=0.3,
    # )
    model = model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    num_warmup = 50
    pure_inf_time = 0
    total_iters = 200
    batch_size = args.batch_size
    for i in range(total_iters):
        sample = torch.ones(()).new_empty(
            (batch_size, *input_shape),
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )
        sample = torch.rand_like(sample)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            model.backbone(sample)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) * batch_size / pure_inf_time
                print('Done image [{:3}/ {}], '.format(i+1, total_iters) + 
                      'fps: {:.2f} img / s'.format(fps))

        if (i + 1) == total_iters:
            fps = (total_iters - num_warmup) * batch_size / pure_inf_time
            print('Overall fps: {:.2f} img / s'.format(fps))
            break

    with torch.no_grad():
        flops, params = get_model_complexity_info(model.backbone, input_shape, print_per_layer_stat=True)
        # flops, params = get_model_complexity_info(model, (1, *input_shape), print_per_layer_stat=True)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
