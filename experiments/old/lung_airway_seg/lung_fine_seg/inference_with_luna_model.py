import os
import sys
from tqdm import tqdm

import torch

MEDCOMMON_ROOT_DIR = '/home/zhangwd/code/work/MedCommon'
sys.path.append(os.path.join(MEDCOMMON_ROOT_DIR, 'experiments/lung_airway_seg/lung_fine_seg'))

from train_luna import net_config, parse_args, init_model, inference_onecase

def inference_singletask(infiles, outdir, is_series=False):
    os.makedirs(outdir, exist_ok=True)
    args = parse_args()
    network, model = init_model(args, net_config)
    for infile in tqdm(infiles):
        basename = os.path.basename(infile).replace('.nii.gz', '')
        outfile = os.path.join(outdir, '{}_mask.nii.gz'.format(basename))
        inference_onecase(network, args, infile, outfile, is_series)


def inference_changzheng_airway(root_dir, out_dir):
    '''
    root_dir: ./data/changzheng/airway/airway_20201030/images
    out_dir: ./data/changzheng/airway/airway_20201030/coarse_lung_masks
    debug cmd: inference_changzheng_airway('./data/changzheng/airway/airway_20201030/images', './data/changzheng/airway/airway_20201030/coarse_lung_masks')
    '''
    suids = os.listdir(root_dir)
    series_paths = [os.path.join(root_dir, i) for i in suids]
    inference_singletask(series_paths, out_dir, True)


if __name__ == '__main__':
    inference_changzheng_airway('./data/changzheng/airway/airway_20201030/images', './data/changzheng/airway/airway_20201030/coarse_lung_masks')


