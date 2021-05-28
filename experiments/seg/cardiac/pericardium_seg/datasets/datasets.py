import os
import sys
from glob import glob
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir))

from utils.datasets_utils import DatasetsUtils
from segmentation.datasets.common_seg_datasets import CommonSegDS

pericardium_seg_root = '/data/medical/cardiac/seg/pericardium'
pericardium_seg_root_images = os.path.join(pericardium_seg_root, 'images')
pericardium_seg_root_masks = os.path.join(pericardium_seg_root, 'masks')
pericardium_seg_root_renamed_masks = os.path.join(pericardium_seg_root, 'renamed_masks')


# 1.划分数据集
def split_ds():
    image_root = pericardium_seg_root_images
    out_config_dir = os.path.join(pericardium_seg_root, 'config/cta')
    os.makedirs(out_config_dir, exist_ok=True)

    DatasetsUtils.split_ds(image_root, out_config_dir, 0.8, 0.2)


class PericardiumSegDS(CommonSegDS):
    def __init__(self, data_root, config_file, crop_size):
        CommonSegDS.__init__(self, data_root, config_file, crop_size)


def test_PericardiumSegDS():

    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm

    data_root = pericardium_seg_root
    config_file = os.path.join(pericardium_seg_root, 'config/cta/train.txt')
    crop_size = [128, 128, 128]
    ds = PericardiumSegDS(data_root, config_file, crop_size)
    dataloader = DataLoader(ds, batch_size=2, pin_memory=True, num_workers=2, drop_last=True)
    for index, (images, masks, _, _) in tqdm(enumerate(dataloader)):
        print('images shape:\t', images.shape)


if __name__ == '__main__':
    # split_ds()
    test_PericardiumSegDS()