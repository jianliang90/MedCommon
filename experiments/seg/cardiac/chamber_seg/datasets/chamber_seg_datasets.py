import os
import sys
import shutil

from glob import glob

import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir))
print('solution root:\t', root)
sys.path.append(root)

from utils.data_io_utils import DataIO
import SimpleITK as sitk

from tqdm import tqdm


'''
将数据统一处理成分割网络所需要的的标准文件夹形式

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

tree -L 1
.
├── config
├── images
└── masks


'''

def unified_chamber_seg_folder(src_root, dst_root):
    '''
    unified_chamber_seg_folder('/data/medical/cardiac/chamber/seg/nii_file', '/data/medical/cardiac/chamber/seg/chamber_seg')
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

'''
统计数据的维度
'''
def analyze_data(src_root, out_file=None):
    '''
    src_root = '/data/medical/cardiac/chamber/seg/chamber_seg/images'

    tree -L 1
    .
    ├── 1.2.392.200036.9116.2.2054276706.1558914751.12.1071700005.1.nii.gz
    ├── 1.2.392.200036.9116.2.2054276706.1578891695.29.1215600017.1.nii.gz
    ├── 1.2.392.200036.9116.2.2054276706.1579657735.12.1267500005.1.nii.gz

    '''
    max_w = 0
    max_h = 0
    max_d = 0
    logs = []
    for suid in tqdm(os.listdir(src_root)):
        series_path = os.path.join(src_root, suid)
        try:
            image = sitk.ReadImage(series_path)
            w,h,d = image.GetSize()
            log = '{}\tw:{}\th:{}\td:{}'.format(suid, w, h, d)
            logs.append(log)
            # print(log)
            if max_w < w:
                max_w = w
            if max_h < h:
                max_h = h
            if max_d < d:
                max_d = d
        except:
            pass
    log = 'max dimensions:\t\t\tw:{}\th:{}\td:{}'.format(max_w, max_h, max_d)
    logs.append(log)
    if not out_file:
        out_dir = './out_result'
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, '{}_analyze_data.txt'.format('{}'.format(__file__).split('.')[0]))
    with open(out_file, 'w') as f:
        f.write('\n'.join(logs))


if __name__ == '__main__':
    # 1. 将数据的文件夹形式统一处理成分割网络所需要的的标准文件夹形式
    # unified_chamber_seg_folder('/data/medical/cardiac/chamber/seg/nii_file', '/data/medical/cardiac/chamber/seg/chamber_seg')
    
    # 2. 对数据的维度进行分析，结果存于`./out_result/chamber_seg_datasets_analyze_data.txt`
    analyze_data('/data/medical/cardiac/chamber/seg/chamber_seg/images')
    