import os
import sys
from glob import glob
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir))

from utils.download_utils import download_dcms_with_website, download_mha_with_csv, get_series_uids, rename_mask_files

# 心包分割根路径
pericardium_seg_root = '/fileser/zhangwd/data/cardiac/seg/pericardium'
pericardium_seg_root_images = os.path.join(pericardium_seg_root, 'images')
pericardium_seg_root_masks = os.path.join(pericardium_seg_root, 'masks')
pericardium_seg_root_renamed_masks = os.path.join(pericardium_seg_root, 'renamed_masks')


def make_dirs_root():
    os.makedirs(pericardium_seg_root, exist_ok=True)
    os.makedirs(pericardium_seg_root_images, exist_ok=True)
    os.makedirs(pericardium_seg_root_masks, exist_ok=True)


# 1. 下载dicom数据
def download_images(out_path, config_file):
    '''
    invoke cmd: python download_ds.py download_images '../../data/changzheng/airway/airway_20201030/images' '../../data/changzheng/airway/annotation/table/文件内网地址信息-导出结果_长征COPD气道分割.csv'
    debug cmd: download_images(pericardium_seg_root_images, os.path.join(pericardium_seg_root_images, 'annotation/ct/文件内网地址信息-导出结果-心包分割.csv'))
    '''
    download_dcms_with_website(out_path, config_file)


# 2. 下载心包标注数据
def download_masks(out_path, config_file):
    '''
    download_masks(pericardium_seg_root_masks, os.path.join(pericardium_seg_root, 'annotation/ct/image_anno_TASK_3836.csv'))
    download_masks(pericardium_seg_root_masks, os.path.join(pericardium_seg_root, 'annotation/ct/image_anno_TASK_3986.csv'))
    '''
    download_mha_with_csv(out_path, config_file)

# 3. rename mask
def rename_mask_files_local(indir, outdir, anno_dir):
    '''
    debug cmd: rename_mask_files_local(pericardium_seg_root_masks, pericardium_seg_root_renamed_masks, os.path.join(pericardium_seg_root, 'annotation/ct'))
    '''
    anno_files = glob(os.path.join(anno_dir, 'image_anno_TASK_*.csv'))
    print('====> files processed:\t', anno_files)
    for anno_file in anno_files:
        rename_mask_files(indir, outdir, anno_file)


if __name__ == '__main__':
    make_dirs_root()

    # 下载cta心包dicom数据
    # download_images(pericardium_seg_root_images, os.path.join(pericardium_seg_root_images, 'annotation/ct/文件内网地址信息-导出结果-心包分割.csv'))

    # 下载cta心包分割标注数据
    # download_masks(pericardium_seg_root_masks, os.path.join(pericardium_seg_root, 'annotation/ct/image_anno_TASK_3836.csv'))
    # download_masks(pericardium_seg_root_masks, os.path.join(pericardium_seg_root, 'annotation/ct/image_anno_TASK_3986.csv'))

    # 将下载的mask文件重命名
    rename_mask_files_local(pericardium_seg_root_masks, pericardium_seg_root_renamed_masks, os.path.join(pericardium_seg_root, 'annotation/ct'))