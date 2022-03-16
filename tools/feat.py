#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data import build_reid_test_loader, build_reid_train_loader
import torch.distributed as dist
import torch
import numpy as np

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def tensor_all_gather(x):
    world_size = dist.get_world_size()
    xlist = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(xlist, x)
    return torch.cat(xlist, dim=0)

def varsize_tensor_all_gather(tensor: torch.Tensor):
    tensor = tensor.contiguous()

    cuda_device = f'cuda:{dist.get_rank()}'
    size_tens = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=cuda_device)

    size_tens = tensor_all_gather(size_tens).cpu()

    max_size = size_tens.max()

    padded = torch.empty(max_size, *tensor.shape[1:],
                         dtype=tensor.dtype,
                         device=cuda_device)
    padded[:tensor.shape[0]] = tensor

    ag = tensor_all_gather(padded)

    slices = []
    for i, sz in enumerate(size_tens):
        start_idx = i * max_size
        end_idx = start_idx + sz.item()

        if end_idx > start_idx:
            slices.append(ag[start_idx:end_idx])

    ret = torch.cat(slices, dim=0)

    return ret.to(tensor)


def main(args):
    cfg = setup(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    # model = DefaultTrainer.build_model(cfg)
    # Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    # res = DefaultTrainer.test(cfg, model)
    # print("cfg: ", cfg)
    pred = DefaultPredictor(cfg)
    reid_dl, num_query = build_reid_test_loader(cfg, dataset_name='PCL')
    # "['images', 'targets', 'camids', 'img_paths']"
    world_size = dist.get_world_size()

    print(f"gpu: {dist.get_rank()} #batches: {len(reid_dl)}")
    all_feats = []
    for i, data in enumerate(reid_dl):
        feat = pred(data['images'])
        feat = torch.nn.functional.normalize(feat, dim=1)
        if world_size > 1:
            combine_feat = varsize_tensor_all_gather(feat)
        else:
            combine_feat = feat
        all_feats.append(combine_feat)

    all_feats = torch.cat(all_feats, dim=0)
    print(all_feats.shape)
    all_feats = all_feats.cpu().numpy()
    np.save(f"{args.feat_file}.npy", all_feats)



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
