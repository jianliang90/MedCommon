import os
import sys
from tqdm import tqdm

import torch

MEDCOMMON_ROOT_DIR = '/home/zhangwd/code/work/MedCommon'
sys.path.append(os.path.join(MEDCOMMON_ROOT_DIR, 'experiments/lung_airway_seg/lung_airway_coarse_seg/lung_airway_coarse_seg_20201116'))

from train import inference

def inference_singletask(infiles, outdir, model_pth, is_series=False):
    os.makedirs(outdir, exist_ok=True)
    for infile in tqdm(infiles):
        basename = os.path.basename(infile).replace('.nii.gz', '')
        out_sub_dir = os.path.join(outdir, basename)
        os.makedirs(out_sub_dir, exist_ok=True)
        inference(infile, model_pth, out_sub_dir, is_series)


def inference_changzheng_airway(root_dir, out_dir, model_pth='./airway_coarse_seg_train_0.020_val_0.052'):
    '''
    root_dir: ./data/changzheng/airway/airway_20201030/images
    out_dir: ./data/changzheng/airway/airway_20201030/coarse_lung_masks
    debug cmd: inference_changzheng_airway('./data/changzheng/airway/airway_20201030/images', './data/changzheng/airway/airway_20201030/coarse_lung_masks')
    '''
    suids = os.listdir(root_dir)
    series_paths = [os.path.join(root_dir, i) for i in suids]
    inference_singletask(series_paths, out_dir, model_pth, True)


if __name__ == '__main__':
    inference_changzheng_airway('./data/changzheng/airway/airway_20201030/images', './data/changzheng/airway/airway_20201030/coarse_lung_masks')


