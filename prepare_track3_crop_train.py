import os
import sys
import pdb
import json
import cv2
import math
import glob
import subprocess
from tqdm import tqdm
from PIL import Image
from multiprocessing import Process,Queue

import numpy as np
import pandas as pd 

if __name__ == "__main__":
    # the directory of track3's detection
    det_result_dir = 'detections/work_dirs/'
    
    # the root directory of track3's trainset
    data_dir = '/data/nif/tiger/'
    img_dir = data_dir+'reid/train/'
    det_fname = det_result_dir+'htc_reidtrain_e10_2200_1100.pkl.bbox.json'
    save_img_dir = det_result_dir+'/crop_images/htc_reidtrain_e10_2200_1100/train/'

    raw_image_fnames = glob.glob(img_dir+'*.jpg')
    raw_image_fnames = [_x.split('/')[-1] for _x in raw_image_fnames]
    os.makedirs(save_img_dir,exist_ok=True)
  
    with open(det_fname,'r',encoding='utf-8') as fid:
        det_boxes = json.load(fid)
    print('len(det_boxes)',len(det_boxes))
    results = {}
    # load results
    for det_id,det in enumerate(det_boxes):
        file_name = '%06d'%det['image_id']+'.jpg'
        det['det_id'] = det_id
        det['save_filename'] = file_name
        ID = file_name
        if ID not in results.keys():
            results[ID] = {'ID':ID,\
                'scores':[], \
                'category_ids':[],\
                'bboxes':[],\
                'save_filenames':[]
            }
        results[ID]['scores'].append(det['score'])
        results[ID]['category_ids'].append(det['category_id'])
        results[ID]['bboxes'].append(det['bbox'])
        results[ID]['save_filenames'].append(det['save_filename'])
    # filter:
    for ID,result in results.items():
        category_ids = np.array(result['category_ids'])
        scores = np.array(result['scores'])
        bboxes = np.array(result['bboxes'])
        areas = bboxes[:,2]*bboxes[:,3]
        
        keep_result = {'ID':ID,\
                'scores':[], \
                'category_ids':[],\
                'bboxes':[],\
                'save_filenames':[]
            }
        # keep the max area
        keep_ind = np.argsort(areas)[::-1][:1]
        # 
        for idx in keep_ind:
            keep_result['scores'].append(result['scores'][idx])
            keep_result['category_ids'].append(result['category_ids'][idx])
            keep_result['bboxes'].append(result['bboxes'][idx])
            keep_result['save_filenames'].append(result['save_filenames'][idx])

        results[ID] = keep_result
  
    save_fnames = []
    with tqdm(total=len(results)) as pbar:
        for ID,result in results.items():
            I = cv2.imread(img_dir+ID)
            if I is None:
                pdb.set_trace()
 
            for idx,det in enumerate(result['bboxes']):
                x,y,w,h = det
                x,y,w,h = int(x),int(y),int(w),int(h)
                crop_img = I[y:y+h,x:x+w,:]
           
                cv2.imwrite(save_img_dir+result['save_filenames'][idx],crop_img)
                save_fnames.append(result['save_filenames'][idx])
            pbar.update(1)
    print(len(save_fnames))
    for fname in raw_image_fnames:
        if fname not in save_fnames:
            I = cv2.imread(img_dir+fname)
            cv2.imwrite(save_img_dir+fname,I)

