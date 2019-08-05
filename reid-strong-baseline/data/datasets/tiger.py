#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/14 15:00
# @Author  : YYH
# @File    : Tiger.py

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class Tiger(BaseImageDataset):
    """
    Tiger

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/Tiger.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    # dataset_dir = 'Tiger'

    def __init__(self,root='/home/haoluo/data', verbose=True, **kwargs):
        super(Tiger, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'train')
        # self.list_train_path = osp.join(self.dataset_dir, 'list_train.csv')
        self.list_train_path = osp.join(self.dataset_dir, 'list_alltrain.csv')
        self.list_val_path = osp.join(self.dataset_dir, 'list_valid.csv')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.csv')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.csv')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path,camid=1)
        #val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path)
        query = self._process_dir(self.test_dir, self.list_query_path,camid=2)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path,camid=3)
        if verbose:
            print("=> Tiger loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path,camid=1):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        for img_idx, img_info in enumerate(lines):
            pid,img_path = img_info.split(',') # id ,
            
            pid = int(pid)  # no need to relabel
            img_path = img_path.strip()
            # camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)

        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset
class Tiger2(Tiger):
    def __init__(self,root='/home/haoluo/data', verbose=True, **kwargs):
        super(Tiger2, self).__init__(root=root,verbose=False)
   
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'train')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.csv')
        # self.list_train_path = osp.join(self.dataset_dir, 'list_alltrain.csv')
        self.list_val_path = osp.join(self.dataset_dir, 'list_valid.csv')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.csv')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.csv')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path,camid=1)
        #val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path)
        query = self._process_dir(self.test_dir, self.list_query_path,camid=2)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path,camid=3)
        if verbose:
            print("=> Tiger2 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)