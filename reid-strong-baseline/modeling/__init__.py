# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .baseline2 import Baseline2
from .baseline3 import Baseline3
from .baseline4 import Baseline4
from .pcb import PCB
def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    # model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    if cfg.FRAMEWORK == 'Baseline4' or cfg.FRAMEWORK=='PCB':
        model = eval(cfg.FRAMEWORK)(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,cfg)
        # model = eval(cfg.FRAMEWORK)(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,cosine_loss_type=cfg.MODEL.COSINE_LOSS_TYPE)
    else:
        model = eval(cfg.FRAMEWORK)(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model
