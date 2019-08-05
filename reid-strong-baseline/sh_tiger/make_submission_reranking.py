# encoding: utf-8
import argparse
import os
import sys
import pdb
import glob
import json
import time
import numpy as np
import pandas as pd
from os import mkdir
from PIL import Image
from tqdm import tqdm

import torch
from torch.backends import cudnn
from torch.utils.data import Dataset,DataLoader,SequentialSampler

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger

from apis.inference import Extractor,ParrallelExtractor
from apis.retrieval import retrieval_reranking_gpu
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
    parser = argparse.ArgumentParser(description="Image Retrival Baseline Inference")
    parser.add_argument("--query_fname", default="", help="path to query file", type=str)
    parser.add_argument("--query_feats_fname", default="", help="fname of feature", type=str)
    parser.add_argument("--save_json_fname", default="", help="shortname of json", type=str)

    # parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)

    args = parser.parse_args()
   
    if 1:
        # ------------------- compute dist and get top n
        with open(args.query_fname,'r',encoding='utf-8') as fid:
            query_boxes = json.load(fid)

        query_features = np.load(args.query_feats_fname)
        query_ids = [_x['det_id'] for _x in query_boxes]
        query_image_ids = [int(_x['save_filename'].split('_')[0]) for _x in query_boxes]
        

        # to fix the accelarate ratio
        # ref https://cloud.tencent.com/developer/ask/136331
        query_features = query_features.astype(np.float64)

        features = torch.from_numpy(query_features).cuda()
        results = retrieval_reranking_gpu(features, features, len(query_ids),k1=20, k2=6, lambda_value=0.25)

        # # combine result
        outputs = []
        with tqdm(total=len(results)) as pbar:
            for result in results:
                query_id,query_results = result['query_id'],result['top_n']
                
                match_inds = []
                for query_result in query_results:
                    gallery_id = query_result[0]
                    if query_image_ids[gallery_id] != query_id:
                        match_inds.append(int(gallery_id)) #
                outputs.append({
                    "query_id":int(query_id),
                    "ans_ids":match_inds
                })

                pbar.update(1)
        boxes = []
        for box in query_boxes:
            pos = box['bbox']
            pos = [float('%.1f'%_x) for _x in pos]
            boxes.append({
                'bbox_id':box['det_id'],
                'image_id':box['image_id'],
                'pos':pos
            })
        results = {
            'bboxs':boxes,
            'reid_result':outputs
        }
        with open(args.save_json_fname,'w') as fid:
            json.dump(results,fid)
    
if __name__ == '__main__':
    main()


