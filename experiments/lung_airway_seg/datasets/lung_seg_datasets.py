import os
import numpy as np
import pandas as pd

import SimpleITK as sitk
from tqdm import tqdm
from glob import glob

import torch
import torch.nn

from torch.utils.data import Dataset, DataLoader

import sys

MEDCOMMON_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(MEDCOMMON_ROOT)
from utils.data_io_utils import DataIO

class AirwayCoarseSeg_DS(Dataset):
    def __init__(self, image_root, mask_root, config_file, crop_size):
        self.image_root = image_root
        self.mask_root = mask_root
        self.config_file = config_file
        self.crop_size = crop_size

        series_uids = []
        self.images_list = []
        self.masks_list = []

        df = pd.read_csv(config_file)
        series_uids = df['SeriesUID'].tolist()

        for series_uid in series_uids:
            image_file = os.path.join(self.image_root, '{}'.format(series_uid))
            mask_file = os.path.join(self.mask_root, '{}'.format(series_uid))
            if not os.path.isfile(image_file):
                continue
            if not os.path.isfile(mask_file):
                continue
            self.images_list.append(image_file)
            self.masks_list.append(mask_file)

    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, item):
        image_file = self.images_list[item]
        mask_file = self.masks_list[item]

        image_arr = DataIO.load_nii_image(image_file)['image']
        mask_arr = DataIO.load_nii_image(mask_file)['image']

        image_tensor = torch.from_numpy(image_arr).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_arr).long()

        return image_tensor, mask_tensor, image_file, mask_file


def test_AirwayCoarseSeg_DS():
    from tqdm import tqdm

    image_root = '/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/image_refine/ori_128_128_128'
    mask_root = '/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/mask_refine/ori_128_128_128'

    train_config_file = '/fileser/zhangfan/DataSet/airway_segment_data/csv/train_filename.csv'
    val_config_file = '/fileser/zhangfan/DataSet/airway_segment_data/csv/val_filename.csv'

    crop_size = [128, 128, 128]

    ds = AirwayCoarseSeg_DS(image_root, mask_root, train_config_file, crop_size)

    dataloader = DataLoader(ds, batch_size=2, pin_memory=True, num_workers=1, drop_last=True)

    for index, (images, masks, _, _) in tqdm(enumerate(dataloader)):
        print('images shape:\t', images.shape)



if __name__ == '__main__':
    test_AirwayCoarseSeg_DS()
        