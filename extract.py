import os
import os.path as osp
import glob

import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import sys
sys.path.insert(0, 'project')

from model import make_model
from config import cfg
import numpy as np

torch.backends.cudnn.benchmark = True

def setup(config_file):
    """
    Create configs and perform basic setups.
    """
    cfg.merge_from_file(config_file)
    # cfg.freeze()
    # default_setup(cfg, args)
    return cfg

class CommDataset(Dataset):
    def __init__(self,  transform=None, img_dir='image/'):

        self.transform = transform
        self.img_dir = img_dir
        imgs = [ x for x in os.listdir(self.img_dir) if x.endswith('.png') ]
        print("#imgs: ", len(imgs))


        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        in_ = self.imgs[index]
        img = Image.open(osp.join(self.img_dir, in_))
        # print(type(img), img.size)
        img = self.transform(img)
        return img, in_
        # {
        #     "images": img,
        #     "img_paths": in_,
        # }

def train_collate_fn(batch):
    imgs, img_paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, img_paths



def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


# def extract_feature(im_path: str) -> np.ndarray:
#     im = Image.open(im_path)
#     fea = np.asarray(im)[::4, ::4, 2].reshape(-1).astype('<f4') / 255
#     return fea


def extract():
    DEBUG = False

    img_dir = 'image'
    fea_dir = 'feature'
    if DEBUG:
        img_dir = 'project/' + img_dir
        fea_dir = 'project/' + fea_dir

    cfg_file = "project/configs/pcl/resnet101.yml"

    cfg = setup(cfg_file)

    # transform = T.Compose(
    #         [T.Resize(size=[256, 128], interpolation=3), T.ToTensor()]
    # )
    transform = T.Compose([
            T.Resize(size=[256, 128], interpolation=Image.BILINEAR), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        )

    os.makedirs(fea_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(img_dir, '*.*'))
    assert(len(img_paths) != 0)
    cfg.TEST.WEIGHT = 'project/logs/r101_0317/resnet101_120.pth'
    print("cfg weights: ", cfg.TEST.WEIGHT)

    num_classes = 15000
    camera_num = 1
    view_num = 1
    model = make_model(cfg, num_class=num_classes, camera_num=1, view_num=1)
    model.load_param(cfg.TEST.WEIGHT)


    reid_dataset = CommDataset(transform=transform, img_dir=img_dir)
    data_loader = DataLoader(reid_dataset,
            batch_size=32,
            num_workers=2,
            pin_memory=True,
            shuffle=False)
    device="cuda:0"

    model.to(device)

    for imgs, in_names in data_loader:
        # basename = get_file_basename(im_path)
        # fea = extract_feature(im_path)
        # write_feature_file(fea, os.path.join(fea_dir, basename + '.dat'))
        with torch.no_grad():
            imgs = imgs.to(device)
            feat = model(imgs, is_training=False)
            feat = feat.cpu().numpy()
            print(np.linalg.norm(feat, axis=1))
            feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)

            for in_, fea in zip(in_names, feat):
                # np.fromfile("a.bin", dtype=np.float)

                # a.tofile("a.bin")
                assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
                fea.astype('<f4').tofile(f"{fea_dir}/{in_[:-4]}.dat")

    print('Extraction Done')


if __name__ == '__main__':
    extract()
