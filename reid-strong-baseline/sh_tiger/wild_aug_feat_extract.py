# encoding: utf-8
import argparse
import os
import sys
import pdb
import glob
import json
import numpy as np
import pandas as pd
from os import mkdir
from PIL import Image
from tqdm import tqdm

import torch
from torch.backends import cudnn
from torch.utils.data import Dataset,DataLoader,SequentialSampler
import torchvision.transforms as T

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger

from apis.inference import Extractor,ParrallelExtractor
class ImageDataset(Dataset):
    """RoIs Person ReID Dataset"""

    def __init__(self, img_fnames,transform=None):
        self.img_fnames = img_fnames
        self.transform = transform
    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, index):
        img = self.img_fnames[index]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

def main():
    # load confiig
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--imgs_path", default="", help="path to images file", type=str)
    parser.add_argument("--det_fname", default="", help="path to det file", type=str)
    parser.add_argument("--save_feats_fname", default="", help="shortname of image", type=str)
    parser.add_argument("--fresh_feats", default=False, help="whether to refresh feat", action='store_true')
    parser.add_argument("--aug_ms", default=False, help="whether to aug", action='store_true')
    parser.add_argument("--aug_flip", default=False, help="whether to aug", action='store_true')
    parser.add_argument("--aug_centercrop", default=False, help="whether to aug", action='store_true')
    
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
 

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("feature extract", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    print("Running with config:\n{}".format(cfg))
    # ------------------- extractor feature
    if args.fresh_feats:
        extractor = Extractor(cfg)

        with open(args.det_fname,'r',encoding='utf-8') as fid:
            det_boxes = json.load(fid)

        img_fnames = [args.imgs_path+_x['save_filename'] for _x in det_boxes]
        rescales = [1.0]
        flip_num = 1
        crop_scales = [1.0]
        if args.aug_ms:
            rescales = [0.7,1.0,1.4]
        if args.aug_flip:
            flip_num = 2
        if args.aug_centercrop:
            crop_scales = [1.0,0.7]
        aug_features = []
        for i in range(flip_num):
            for crop_scale in crop_scales:
                for rescale in rescales:
                    # build transform
                    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
                    h,w = cfg.INPUT.SIZE_TEST
                    if i == 0:
                        transform = T.Compose([
        
                            T.Resize((int(h*rescale),int(w*rescale)),interpolation=Image.LANCZOS),  
                            T.CenterCrop((int(h*crop_scale),int(w*crop_scale))) ,                   
                            T.ToTensor(),
                            normalize_transform
                        ])
                    else:
                        transform = T.Compose([
                            T.Resize((int(h*rescale),int(w*rescale)),interpolation=Image.LANCZOS), #
                            T.CenterCrop((int(h*crop_scale),int(w*crop_scale))) ,
                            T.RandomVerticalFlip(1.1),# T.RandomHorizontalFlip(1.1),
                            T.ToTensor(),
                            normalize_transform
                        ])
                    logger.info(transform)
                    image_set = ImageDataset(img_fnames,transform)
                    image_loader = DataLoader(image_set,
                                        sampler = SequentialSampler(image_set),batch_size= cfg.TEST.IMS_PER_BATCH,
                                        num_workers = 4
                                        )
                    features = []
                    with tqdm(total=len(image_loader)) as pbar:
                        for idx,batchs in enumerate(image_loader):
                            features.append(extractor.apply_batch(batchs.cuda()).cpu().numpy())
                            pbar.update(1)
                    features = np.vstack(features) #N,F
                    aug_features.append(features)

        features = np.hstack(aug_features)
        np.save(os.path.join(output_dir,args.save_feats_fname.replace('.npy','_cat.npy')),features)

        features = aug_features[0]
        for i in range(1,len(aug_features)):
            features += aug_features[i]
        features /= len(aug_features)

        np.save(os.path.join(output_dir,args.save_feats_fname.replace('.npy','_mean.npy')),features)

if __name__ == '__main__':
    main()
