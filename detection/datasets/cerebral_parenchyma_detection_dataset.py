import os
import shutil
import sys

MEDCOMMON_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir)
sys.path.append(MEDCOMMON_ROOT)
sys.path.append(os.path.join(MEDCOMMON_ROOT, 'external_lib'))

from utils.data_io_utils import DataIO
from utils.mask_bounding_utils import MaskBoundingUtils
import SimpleITK as sitk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

# 1. 将数据集统一格式, CTA脑实质数据
def convert_cta_data():
    from tqdm import tqdm
    import shutil
    images_root = '/data/medical/brain/cerebral_parenchyma/cta/image/dicom'
    mask_root = '/data/medical/brain/cerebral_parenchyma/cta/mask/Ori_nii'
    out_root = '/data/medical/brain/cerebral_parenchyma/exp/cta'
    out_images_root = os.path.join(out_root, 'images')
    out_mask_root = os.path.join(out_root, 'masks')
    os.makedirs(out_images_root, exist_ok=True)
    os.makedirs(out_mask_root, exist_ok=True)
    for suid in tqdm(os.listdir(images_root)):
        series_path = os.path.join(images_root, suid)
        out_series_path = os.path.join(out_images_root, '{}.nii.gz'.format(suid))
        sitk_image = DataIO.load_dicom_series(series_path)['sitk_image']
        sitk.WriteImage(sitk_image, out_series_path)
        in_mask_file = os.path.join(mask_root, '{}.nii.gz'.format(suid))
        if not os.path.exists(in_mask_file):
            print('mask file {} not exist!'.format(in_mask_file))
            continue
        out_mask_file = os.path.join(out_mask_root, '{}.nii.gz'.format(suid))
        shutil.copyfile(in_mask_file, out_mask_file)

# 2. 将数据集统一格式, CTA脑实质数据
def convert_ncct_data():
    from glob import glob
    from tqdm import tqdm
    in_root = '/data/medical/brain/cerebral_parenchyma/ncct/brain-NCCT-with-mask'
    out_root = '/data/medical/brain/cerebral_parenchyma/exp/ncct'
    out_images_root = os.path.join(out_root, 'images')
    out_mask_root = os.path.join(out_root, 'masks')
    os.makedirs(out_images_root, exist_ok=True)
    os.makedirs(out_mask_root, exist_ok=True)
    files = glob(os.path.join(in_root, '*_CT.nii.gz'))
    for image_file in tqdm(files):
        image_filename = os.path.basename(image_file)
        mask_filename = image_filename.replace('_CT', '_brain_mask')
        mask_file = os.path.join(in_root, mask_filename)
        out_image_file = os.path.join(out_images_root, image_filename.replace('_CT', ''))
        out_mask_file = os.path.join(out_mask_root, image_filename.replace('_CT', ''))
        if not os.path.exists(mask_file):
            print('mask file {} not exist!'.format(mask_file))
            shutil.copyfile()
        shutil.copyfile(image_file, out_image_file)
        shutil.copyfile(mask_file, out_mask_file)

class CerebralParenchymaDetectionDS(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.image_root = os.path.join(self.root, 'images')
        self.mask_root = os.path.join(self.root, 'images')
        self.image_files = []
        self.targets = []
        for filename in os.listdir(self.image_root):
            image_file = os.path.join(self.image_root, filename)
            mask_file = os.path.join(self.mask_root, filename)
            if not os.path.exists(image_file):
                continue
            if not os.path.exists(mask_file):
                continue
            z_min, y_min, x_min, z_max, y_max, x_max = MaskBoundingUtils.extract_mask_file_bounding(mask_file)
            self.targets.append(np.array([[z_min, y_min, x_min, z_max, y_max, x_max]]))
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


if __name__ == '__main__':
    # convert_cta_data()
    # convert_ncct_data()