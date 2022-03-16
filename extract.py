import os
import os.path as osp
import glob

import numpy as np
from PIL import Image


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import sys
sys.path.insert(0, 'project')

from fastreid.config import get_cfg
from fastreid.engine import default_setup
from fastreid.modeling.meta_arch import build_model
import torch
from fastreid.utils.checkpoint import Checkpointer
torch.backends.cudnn.benchmark = True



def setup(config_file):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.freeze()
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


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.
    This predictor takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
    Examples:
    .. code-block:: python
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg.defrost()
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.model = build_model(self.cfg)
        self.model.eval()
        print("weights: ", cfg.MODEL.WEIGHTS)
        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

    def __call__(self, image):
        """
        Args:
            image (torch.tensor): an image tensor of shape (B, C, H, W).
        Returns:
            predictions (torch.tensor): the output features of the model
        """
        inputs = {"images": image.to(self.model.device)}
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            predictions = self.model(inputs)
        return predictions.cpu()


def extract():
    DEBUG = False

    img_dir = 'image'
    fea_dir = 'feature'
    if DEBUG:
        img_dir = 'project/' + img_dir
        fea_dir = 'project/' + fea_dir

    cfg_file = "project/configs/pcl/bagtricks_R50-ibn.yml"

    cfg = setup(cfg_file)

    transform = T.Compose(
            [T.Resize(size=[256, 128], interpolation=3), T.ToTensor()]
    )

    os.makedirs(fea_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(img_dir, '*.*'))
    assert(len(img_paths) != 0)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = build_model(cfg)
    cfg.MODEL.WEIGHTS='project/logs/pcl/bagtricks_R50-ibn/model_final.pth'
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    # TODO: load the checkpoint

    model.eval()
    reid_dataset = CommDataset(transform=transform, img_dir=img_dir)
    data_loader = DataLoader(reid_dataset,
            batch_size=32,
            num_workers=2,
            pin_memory=True,
            shuffle=False)

    device="cuda:0"

    for imgs, in_names in data_loader:
        # basename = get_file_basename(im_path)
        # fea = extract_feature(im_path)
        # write_feature_file(fea, os.path.join(fea_dir, basename + '.dat'))
        with torch.no_grad():
            inputs = {"images": imgs.to(device)}
            feat = model(inputs)
            feat = feat.cpu().numpy()
            feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)

            for in_, fea in zip(in_names, feat):
                # np.fromfile("a.bin", dtype=np.float)
                # a.tofile("a.bin")
                fea.tofile(f"{fea_dir}/{in_[:-4]}.dat")


    print('Extraction Done')


if __name__ == '__main__':
    extract()
