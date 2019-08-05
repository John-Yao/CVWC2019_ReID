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
    # the root directory of track4's testset
    data_dir = '/data/nif/tiger/'
    imgs_dir = data_dir+'detection/test/images/'
    # the directory of track4's detection
    det_result_dir = 'detections/work_dirs/'

    det_fname = det_result_dir+'/enew2hard.json'
    save_json_fname  = det_result_dir+'/enew2hard.rois'
    save_imgs_dir = det_result_dir+'/crop_images/enew2hard/'

    os.makedirs(save_imgs_dir,exist_ok=True)
  
    with open(det_fname,'r',encoding='utf-8') as fid:
        det_boxes = json.load(fid)

    results = {}
    # load results
    for det_id,det in enumerate(det_boxes):
        file_name = '%04d'%det['image_id']+'.jpg'
        det['det_id'] = det_id
        det['save_filename'] = file_name.replace('.jpg','_%08d.jpg'%det_id)
        ID = file_name
        if ID not in results.keys():
            results[ID] = {
                            'ID':ID,'dtboxes':[] \
                            }
        
        results[ID]['dtboxes'].append(det)
    # crop image
    with tqdm(total=len(results)) as pbar:
        for ID,result in results.items():
            I = cv2.imread(imgs_dir+ID)
            if I is None:
                pdb.set_trace()
            for det in result['dtboxes']:
                x,y,w,h = det['bbox']
                x,y,w,h = int(x),int(y),int(w),int(h)
                crop_img = I[y:y+h,x:x+w,:]
                cv2.imwrite(save_imgs_dir+det['save_filename'],crop_img)
            pbar.update(1)
    print('len(det_boxes)',len(det_boxes))
    # save infomation for crop images
    with open(save_json_fname,'w',encoding='utf-8') as fid:
        json.dump(det_boxes,fid)


