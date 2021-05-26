import os
import shutil
import sys

MEDCOMMON_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir)
sys.path.append(MEDCOMMON_ROOT)
sys.path.append(os.path.join(MEDCOMMON_ROOT, 'external_lib'))

from utils.data_io_utils import DataIO
from utils.mask_bounding_utils import MaskBoundingUtils
from utils.detection_utils import DETECTION_UTILS
import SimpleITK as sitk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm

import json

def extract_boundary_info(mask_root, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    info_dict = {}
    for filename in tqdm(os.listdir(mask_root)):
        mask_file = os.path.join(mask_root, filename)
        boundary_info = MaskBoundingUtils.extract_mask_file_bounding(mask_file)
        image = sitk.ReadImage(mask_file)
        image_shape = image.GetSize()
        info = list(boundary_info) + list(image_shape)
        info_dict[filename] = [int(i) for i in info]
    with open(out_file, 'w') as f:
        f.write(json.dumps(info_dict))
    print('====> extract_boundary_info finished!')
    


class PositionDetectionDS(Dataset):
    def __init__(self, root, image_shape=[128,128,128], boundary_info_file=None) -> None:
        super().__init__()
        self.root = root
        self.image_root = os.path.join(self.root, 'images')
        self.mask_root = os.path.join(self.root, 'masks')
        self.image_files = []
        self.targets = []
        boundary_infos = None
        if boundary_info_file:
            with open(boundary_info_file) as f:
                boundary_infos = json.loads(f.read())
        for filename in tqdm(os.listdir(self.image_root)):
            image_file = os.path.join(self.image_root, filename)
            mask_file = os.path.join(self.mask_root, filename)
            if not os.path.exists(image_file):
                continue
            if not os.path.exists(mask_file):
                continue
            if boundary_info_file:
                z_min, y_min, x_min, z_max, y_max, x_max = boundary_infos[filename][:6]
                in_shape = boundary_infos[filename][6:]
            else:
                z_min, y_min, x_min, z_max, y_max, x_max = MaskBoundingUtils.extract_mask_file_bounding(mask_file)
                in_image = sitk.ReadImage(image_file)
                in_shape = in_image.GetSize()
            self.image_files.append(image_file)

            x_min, y_min, z_min = DETECTION_UTILS.point_coordinate_resampled(in_shape, image_shape, [x_min, y_min, z_min])
            x_max, y_max, z_max = DETECTION_UTILS.point_coordinate_resampled(in_shape, image_shape, [x_max, y_max, z_max])
            # 归一化
            x_min /= image_shape[0]
            x_max /= image_shape[0]
            y_min /= image_shape[1]
            y_max /= image_shape[1]
            z_min /= image_shape[2]
            z_max /= image_shape[2]
            self.targets.append(np.array([[z_min, y_min, x_min, z_max, y_max, x_max]]))
            # if self.image_files.__len__() > 2:
            #     break
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = sitk.ReadImage(image_file)
        arr = sitk.GetArrayFromImage(image)
        image_tensor = torch.from_numpy(arr).float()
        image_tensor = image_tensor.unsqueeze(0)
        target = self.targets[index]
        return image_tensor, target, image_file

def test_PositionDetectionDS():
    root = '/data/medical/brain/cerebral_parenchyma/exp/cta'
    boundary_info_file='/data/medical/brain/cerebral_parenchyma/exp/cta/config/mask_boundary_info.json'
    ds = PositionDetectionDS(root, boundary_info_file=boundary_info_file)

    dataloader = DataLoader(ds, batch_size=1)

    for index, (images, targets, _) in enumerate(dataloader):
        print(images.shape)
        print(targets)


if __name__ == '__main__':
    # extract_boundary_info(mask_root='/data/medical/brain/cerebral_parenchyma/exp/cta/masks', out_file='/data/medical/brain/cerebral_parenchyma/exp/cta/config/mask_boundary_info.json')
    # extract_boundary_info(mask_root='/data/medical/cardiac/seg/coronary/coronary_ori/masks', out_file='/data/medical/cardiac/seg/coronary/coronary_ori/config/mask_boundary_info.json')
    test_PositionDetectionDS()

