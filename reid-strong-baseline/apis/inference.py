
import os
import sys
import cv2
import warnings
import argparse
import time
import numpy as np 

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.backends import cudnn

from PIL import Image
# sys.path.append('.')
# from config import cfg
from data import make_data_loader
from data.datasets import init_dataset, ImageDataset
from data.datasets.dataset_loader import read_image
from data.transforms import build_transforms
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger

def init_extractor(cfg, num_classes=8368):
    # if cfg.MODEL.DEVICE == "cuda":
    #     os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True
    if num_classes is None:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
        num_classes = dataset.num_train_pids
    
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    return model
class Extractor(object):
    def __init__(self, cfg, use_cuda=True,device=None):
        self.cfg = cfg
        self.net = init_extractor(cfg)
        if device==None:
            self.device = "cuda" if use_cuda else "cpu"
        else:
            self.device = device
        self.net.to(self.device)
        self.norm = build_transforms(cfg, is_train=False)
    # PIL.Image W,H,C RGB
    def __call__(self, img):
        assert isinstance(img, str) or isinstance(img,Image.Image), "type error"
        if isinstance(img, str):
            img = read_image(img) # 
        img = self.norm(img).unsqueeze(0)
        with torch.no_grad():
            img = img.to(self.device)
            feature = self.net(img)
        if self.cfg.TEST.FEAT_NORM == 'yes':
            feature = torch.nn.functional.normalize(feature, dim=1, p=2)
        return feature
    # PIL.Image W,H,C RGB
    def apply(self, imgs):
        norm_imgs = []
  
        for i in range(len(imgs)):
            norm_imgs.append(self.norm(imgs[i]).unsqueeze(0)) #0.79ms/roi
        norm_imgs = torch.cat(norm_imgs,0)
    
        with torch.no_grad():
            norm_imgs = norm_imgs.to(self.device)   #56ms/roi
            features = self.net(norm_imgs)

        if self.cfg.TEST.FEAT_NORM == 'yes':
            features = torch.nn.functional.normalize(features, dim=1, p=2)
        
        return features
    def apply_batch(self, norm_imgs):
        with torch.no_grad():
            norm_imgs = norm_imgs.to(self.device)
            features = self.net(norm_imgs)
        if self.cfg.TEST.FEAT_NORM == 'yes':
            features = torch.nn.functional.normalize(features, dim=1, p=2)
        return features
    # https://blog.csdn.net/enjoy_now/article/details/73941979
class ParrallelExtractor(object):
    def __init__(self, cfg,device_ids):
        self.cfg = cfg
        self.net = init_extractor(cfg)
       
        self.net = nn.DataParallel(self.net.cuda(),device_ids=device_ids)
        self.norm = build_transforms(cfg, is_train=False)
    # PIL.Image W,H,C RGB
    def apply_batch(self, norm_imgs):
        with torch.no_grad():
            features = self.net(norm_imgs)
        if self.cfg.TEST.FEAT_NORM == 'yes':
            features = torch.nn.functional.normalize(features, dim=1, p=2)
        return features