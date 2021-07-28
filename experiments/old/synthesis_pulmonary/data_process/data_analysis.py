import os
import sys

from glob import glob
from tqdm import tqdm

import fire

# 分析数据
def analyze_synthesis_pulmonary_data(root_dir):
    '''
    root_dir: ../data/synthesis_pulmonary/raw_data

    debug cmd: analyze_synthesis_pulmonary_data('../data/synthesis_pulmonary/raw_data')

    .
    ├── raw_data
    │   ├── APCTP_001-008
    │   │   ├── 001
    │   │   │   ├── 001_air.nrrd
    │   │   │   ├── 001_art.nrrd
    │   │   │   ├── 001_image.nrrd
    │   │   │   ├── 001_lung.nrrd
    │   │   │   ├── 001_parenchyma.nrrd
    │   │   │   ├── 001_refstd.nrrd
    │   │   │   └── 001_vein.nrrd
    │   │   ├── 002
    │   │   │   ├── 002_air.nrrd
    │   │   │   ├── 002_art.nrrd
    │   │   │   ├── 002_image.nrrd
    │   │   │   ├── 002_lung.nrrd
    │   │   │   ├── 002_parenchyma.nrrd
    │   │   │   ├── 002_refstd.nrrd
    │   │   │   └── 002_vein.nrrd

    '''
    tissue_dict = {}
    patient_path = []
    for sub_folder in os.listdir(root_dir):
        if not sub_folder.startswith('APCTP'):
            continue
        sub_folder = os.path.join(root_dir, sub_folder)
        if not os.path.isdir(sub_folder):
            continue
        for pid in os.listdir(sub_folder):
            # 检查pid，pid为数字，否则跳过
            try:
                int(pid)
            except:
                continue
            pid_path = os.path.join(sub_folder, pid)
            if not os.path.isdir(pid_path):
                continue
            tissue_files = glob(os.path.join(pid_path, '*.nrrd'))
            for tissue_file in tissue_files:
                tissue_name = os.path.basename(tissue_file).split('.')[0].split('_')[1]
                if tissue_name in tissue_dict:
                    tissue_dict[tissue_name] += [pid]
                else:
                    tissue_dict[tissue_name] = [pid]

    print('====> tissues :\t{}'.format(list(tissue_dict)))
    
    for key,val in tissue_dict.items():
        print('\t{}\t{}\t{}'.format(key, len(val), val))


if __name__ == '__main__':
    analyze_synthesis_pulmonary_data('../data/synthesis_pulmonary/raw_data')
