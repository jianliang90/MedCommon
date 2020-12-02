import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(ROOT)

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(ROOT, 'segmentation/external_lib/MedicalZooPytorch'))

from segmentation.external_lib.MedicalZooPytorch.lib.medzoo.Unet3D import UNet3D
from segmentation.external_lib.MedicalZooPytorch.lib.losses3D.dice import DiceLoss
from segmentation.datasets.common_seg_datasets import CommonSegDS
from segmentation.runner.train_seg import SegmentationTrainer
from utils.datasets_utils import DatasetsUtils
from segmentation.models.unet3d_auto_resample import ResampledUnet3D

import torch
import torch.nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.misc_utils import AverageMeter
import time
from tqdm import tqdm

import argparse
import SimpleITK as sitk

import fire


seg_root = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg_resampled_unified'


class Options():
    def __init__(self):
        self.lr = 1e-3
        self.epochs = 100
        self.lr_fix = 50
        self.display = 2
        self.model_dir = './output/seg/model'


def main():
    opts = Options()

    train_data_root = seg_root
    train_config_file = os.path.join(seg_root, 'config/train.txt')

    val_data_root = seg_root
    val_config_file = os.path.join(seg_root, 'config/val.txt')

    crop_size = [128, 128, 128]

    train_ds = CommonSegDS(train_data_root, train_config_file, crop_size)
    train_dataloader = DataLoader(train_ds, batch_size=6, pin_memory=True, num_workers=2, drop_last=True)

    val_ds = CommonSegDS(val_data_root, val_config_file, crop_size)
    val_dataloader = DataLoader(val_ds, batch_size=3, pin_memory=True, num_workers=2, drop_last=True)

    # crop_size = [64, 64, 64]
    num_classes = 8

    model = ResampledUnet3D(1, num_classes)

    trainer = SegmentationTrainer()
    criterion = DiceLoss(num_classes).cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=opts.lr, betas=(0.9, 0.999))
    best_loss = 1
    for epoch in range(opts.epochs):
        loss_train, _ = trainer.train_one_epoch(train_dataloader, torch.nn.DataParallel(model).cuda(), criterion, optimizer, epoch, opts.display)
        loss, _ = trainer.train_one_epoch(val_dataloader, torch.nn.DataParallel(model).cuda(), criterion, optimizer, epoch, opts.display, 'val')
        if loss < best_loss:
            best_loss = loss
            print('current best val loss is:\t{}'.format(best_loss))
            os.makedirs(opts.model_dir, exist_ok=True)
            saved_model_path = os.path.join(opts.model_dir, 'common_seg_train_{:.3f}_val_{:.3f}'.format(loss_train, loss))
            torch.save(model.cpu().state_dict(), saved_model_path)
            print('====> save model:\t{}'.format(saved_model_path))


if __name__ == '__main__':
    # DatasetsUtils.split_ds(os.path.join(seg_root, 'images'), os.path.join(seg_root, 'config'))
    main()