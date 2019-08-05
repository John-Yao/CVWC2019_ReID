import sys
import pdb
import argparse

import torch
from torch import nn

sys.path.append('.')
# from solver.lr_scheduler import WarmupMultiStepLR
# from solver.build import make_optimizer

from apis.inference import Extractor
from config import cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    extractor = Extractor(cfg,use_cuda=True)
    pdb.set_trace()