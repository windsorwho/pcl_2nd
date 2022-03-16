# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PCL(ImageDataset):
    dataset_dir = 'pcl'

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        if 0:
            required_files = [
                self.data_dir,
                self.train_dir,
                self.query_dir,
                self.gallery_dir,
            ]

        train = self.process_dir(self.train_dir)

        query = [ x for x in train ]
        gallery = []

        super(PCL, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, is_train=True):
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')

        # data = []
        # for img_path in img_paths:
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     if pid == -1:
        #         continue  # junk images are just ignored
        #     assert 0 <= pid <= 1501  # pid == 0 means background
        #     assert 1 <= camid <= 6
        #     camid -= 1  # index starts from 0
        #     if is_train:
        #         pid = self.dataset_name + "_" + str(pid)
        #         camid = self.dataset_name + "_" + str(camid)
        #     data.append((img_path, pid, camid))

        label_file = osp.join(self.data_dir, 'train_list.txt')
        files = open(label_file, 'r').read().splitlines()
        files = [x.split() for x in files]

        data = [(osp.join(self.data_dir, x), int(y), 0) for x, y in files]
        print("data examples: ", data[:2])

        return data
