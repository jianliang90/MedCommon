import os
import sys

from glob import glob
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch


class GAN_SLICE_DS(Dataset):
    def __init__(self, data_root, config_data_file, phase, crop_size, scale_size):
        self.data_root = data_root
        self.config_data_file = config_data_file
        self.phase = phase
        self.crop_size = crop_size
        self.scale_size = scale_size

        self.pos_list = []
        self.neg_list = []
        self.series_uids = []

        with open(config_data_file) as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                series_uid = line
                self.series_uids.append(series_uid)

        print('data group size:\t{}'.format(len(self.series_uids)))

        for series_uid in self.series_uids:
            series_path = os.path.join(self.data_root, series_uid)
            if not os.path.isdir(series_path):
                print('{} is not a path!')
            src_pos_files = glob(os.path.join(series_path, '*src*pos*.npy'))
            dst_pos_files = [i.replace('src', 'dst') for i in src_pos_files]
            src_neg_files = glob(os.path.join(series_path, '*src*neg*.npy'))
            dst_neg_files = [i.replace('src', 'dst') for i in src_neg_files]

            pos_files = list(zip(src_pos_files, dst_pos_files))
            neg_files = list(zip(src_neg_files, dst_neg_files))

            self.pos_list += pos_files
            self.neg_list += neg_files

        print('positive samples size:\t{}'.format(len(self.pos_list)))
        print('negative samples size:\t{}'.format(len(self.neg_list)))


    def __len__(self):
        return len(self.pos_list)


    def __getitem__(self, item):
        if self.phase == 'train':
            if np.random.rand() <0.85:
                src_file = self.pos_list[item][0]
                dst_file = self.pos_list[item][1]
            else:
                rand_idx = np.random.randint(0, len(self.neg_list))
                src_file = self.neg_list[rand_idx][0]
                dst_file = self.neg_list[rand_idx][1]

            src_data = np.load(src_file)
            dst_data = np.load(dst_file)

            src_tensor = torch.from_numpy(src_data).float()
            src_tensor = torch.unsqueeze(src_tensor, axis=0)

            dst_tensor = torch.from_numpy(dst_data).float()
            dst_tensor = torch.unsqueeze(dst_tensor, axis=0)

            return src_tensor, dst_tensor, src_file, dst_file


def test_GAN_SLICE_DS():
    data_root = '/data/medical/hospital/huadong/copd/copd_gan/data_412/images/slice'
    config_data_file = '/data/medical/hospital/huadong/copd/copd_gan/data_412/annotation/config/slice/train.txt'
    phase = 'train'
    crop_size = 512
    scale_size = 512

    ds = GAN_SLICE_DS(data_root, config_data_file, phase, crop_size, scale_size)

    data_loader = DataLoader(ds, shuffle=True, batch_size=2, pin_memory=False)

    for index, (src_img, dst_img, src_file, dst_file) in enumerate(data_loader):
        print(src_file)
        if index > 2000:
            break


if __name__ == '__main__':
    test_GAN_SLICE_DS()