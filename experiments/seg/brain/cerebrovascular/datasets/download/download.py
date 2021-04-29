import os
import sys
import pandas as pd
import numpy as np

from glob import glob

import fire

root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir
    )
)

print(root)

sys.path.append(root)

DATA_ROOT = '/data/medical/brain/Cerebrovascular/segmenation'

from utils.download_utils import DownloadUtils


# 1. export image series uids
# DownloadUtils.get_series_uids_batch(os.path.join(DATA_ROOT, 'annotation/anno_table'), os.path.join(DATA_ROOT, 'annotation/series_table/series_uids.txt'))

# 2. download image series
# download_pth = os.path.join(DATA_ROOT, 'images')
# config_file = os.path.join(DATA_ROOT, 'annotation/series_table/文件内网地址信息-导出结果.csv')
# DownloadUtils.download_dcms_with_website_multiprocess(download_pth, config_file)

# 3. download mask
# anno_root = os.path.join(DATA_ROOT, 'annotation/anno_table')
# out_path = os.path.join(DATA_ROOT, 'masks')
# DownloadUtils.download_masks_batch(anno_root, out_path)

# 4. rename the mask to match the corresponding image
indir = os.path.join(DATA_ROOT, 'masks')
outdir = os.path.join(DATA_ROOT, 'masks')
anno_root = os.path.join(DATA_ROOT, 'annotation/anno_table')
DownloadUtils.rename_mask_files_batch(indir, outdir, anno_root)