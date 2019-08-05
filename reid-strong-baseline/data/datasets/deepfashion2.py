#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/14 15:00
# @Author  : YYH
# @File    : deepfashion2.py

import glob
import re

import os.path as osp
import pandas as pd
from .bases import BaseImageDataset


class DeepFashion2(BaseImageDataset):
    """
    DeepFashion2

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/DeepFashion2.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    # dataset_dir = 'DeepFashion2'

    def __init__(self,root='/home/haoluo/data', verbose=True, **kwargs):
        super(DeepFashion2, self).__init__()
        self.dataset_dir = root
        self.train_dir = self.dataset_dir
        self.test_dir = self.dataset_dir
        self.list_train_path = osp.join(self.dataset_dir, 'csvs/list_train.csv.min8')
        self.list_query_path = osp.join(self.dataset_dir, 'csvs/list_query.csv.min8')
        self.list_gallery_path = osp.join(self.dataset_dir, 'csvs/list_gallery.csv.min8')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path,camid=1)
        query = self._process_dir(self.test_dir, self.list_query_path,camid=2)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path,camid=3)
        if verbose:
            print("=> DeepFashion2 loaded")
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

        df = pd.read_csv(list_path)
        dataset = []
        pid_container = set()
        for row in df.itertuples():
            pid = getattr(row, 'id')
            img_path = getattr(row,'filename')
            
            pid = int(pid)  # no need to relabel
            img_path = img_path.strip()

            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)

        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset


class DeepFashion2_1(BaseImageDataset):
    """
    DeepFashion2_1

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/DeepFashion2_1.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    # dataset_dir = 'DeepFashion2_1'

    def __init__(self,root='/home/haoluo/data', verbose=True, **kwargs):
        super(DeepFashion2_1, self).__init__()
        self.dataset_dir = root
        self.train_dir = self.dataset_dir
        self.test_dir = self.dataset_dir
        self.list_train_path = osp.join(self.dataset_dir, 'mask_csvs/list_train.csv.min8')
        self.list_query_path = osp.join(self.dataset_dir, 'mask_csvs/list_query.csv.min8')
        self.list_gallery_path = osp.join(self.dataset_dir, 'mask_csvs/list_gallery.csv.min8')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path,camid=1)
        query = self._process_dir(self.test_dir, self.list_query_path,camid=2)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path,camid=3)
        if verbose:
            print("=> DeepFashion2_1 loaded")
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

        df = pd.read_csv(list_path)
        dataset = []
        pid_container = set()
        for row in df.itertuples():
            pid = getattr(row, 'id')
            img_path = getattr(row,'filename')
            
            pid = int(pid)  # no need to relabel
            img_path = img_path.strip()

            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)

        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset