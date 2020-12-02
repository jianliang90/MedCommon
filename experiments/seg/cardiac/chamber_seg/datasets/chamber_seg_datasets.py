import os
import sys
import shutil

from glob import glob

import numpy as np

'''
src folder format: 


tree -L 2
.
├── 1.2.392.200036.9116.2.2054276706.1558914751.12.1071700005.1
│   ├── img.nii.gz
│   └── mask.nii.gz
├── 1.2.392.200036.9116.2.2054276706.1578891695.29.1215600017.1
│   ├── img.nii.gz
│   └── mask.nii.gz
├── 1.2.392.200036.9116.2.2054276706.1579657735.12.1267500005.1
│   ├── img.nii.gz
│   └── mask.nii.gz
├── 1.2.392.200036.9116.2.2054276706.1579657986.13.1267800006.1
│   ├── img.nii.gz
│   └── mask.nii.gz
├── 1.2.392.200036.9116.2.2054276706.1580188912.9.1276300003.1
│   ├── img.nii.gz
│   └── mask.nii.gz
├── 1.2.392.200036.9116.2.2054276706.1580276125.6.1281700002.1
│   ├── img.nii.gz
│   └── mask.nii.gz



dst folder format:


'''

def unified_chamber_seg_folder(src_root, dst_root):
    '''
    unified_chamber_seg_folder('/fileser/zhangwd/data/cardiac/chamber/seg/nii_file', '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg')
    '''
    series_uids = os.listdir(src_root)
    dst_images_root = os.path.join(dst_root, 'images')
    dst_masks_root = os.path.join(dst_root, 'masks')
    os.makedirs(dst_images_root, exist_ok=True)
    os.makedirs(dst_masks_root, exist_ok=True)
    for series_uid in series_uids:
        series_path = os.path.join(src_root, series_uid)
        src_image_file = os.path.join(series_path, 'img.nii.gz')
        src_mask_file = os.path.join(series_path, 'mask.nii.gz')
        dst_image_file = os.path.join(dst_images_root, '{}.nii.gz'.format(series_uid))
        dst_mask_file = os.path.join(dst_masks_root, '{}.nii.gz'.format(series_uid))
        shutil.copyfile(src_image_file, dst_image_file)
        shutil.copyfile(src_mask_file, dst_mask_file)
        print('====> copy from {} to {}'.format(src_image_file, dst_image_file))
        print('====> copy from {} to {}'.format(src_mask_file, dst_mask_file))


if __name__ == '__main__':
    unified_chamber_seg_folder('/fileser/zhangwd/data/cardiac/chamber/seg/nii_file', '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg')