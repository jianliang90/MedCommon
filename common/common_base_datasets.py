import os
import sys

import torch
import torchvision

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root)
sys.path.append(os.path.join(root, 'external_lib/torchio'))

import torchio as tio
from torch.utils.data import Dataset, DataLoader

from utils.data_aug_utils import DATA_AUGMENTATION_UTILS

from glob import glob

'''
this super class for seg&gan task
'''

'''
'''

class COMMON_PAIRS_DS(Dataset):
    def __init__(self, src_files, dst_files, image_shape, transforms=None):
        subjects = []
        for (src, dst) in zip(src_files, dst_files):
            subject = tio.Subject(src=tio.ScalarImage(src), dst=tio.LabelMap(dst))
            # subject = tio.Subject(src=tio.ScalarImage(src), dst=tio.ScalarImage(dst))
            subjects.append(subject)
        if transforms is None:
            self.transforms = DATA_AUGMENTATION_UTILS.get_common_transform(image_shape, 'GAN')
        self.subjects_dataset = tio.SubjectsDataset(subjects, transform=self.transforms)    


    def __len__(self):
        return self.subjects_dataset.__len__()

    def __getitem__(self, item):
        return self.subjects_dataset.__getitem__(item)



'''
    该通用分割类，适用于如下构成的文件夹格式，images如果为dicom格式，请先转换为nii.gz的形式
    tree -L 2
    .
    ├── images
    │   ├── 1.2.840.113704.1.111.10192.1571886399.11.nii.gz
    │   ├── 1.2.840.113704.1.111.11692.1420599548.14.nii.gz
    │   ├── 1.2.840.113704.1.111.11716.1415240146.11.nii.gz
    │   ├── 1.2.840.113704.1.111.12576.1389599418.6.nii.gz
    │   ├── 1.2.840.113704.1.111.13172.1389599763.7.nii.gz
    │   ├── 1.2.840.113704.1.111.1384.1392885868.9.nii.gz
    │   ├── 1.2.840.113704.1.111.2452.1387439529.10.nii.gz
    │   ├── 1.2.840.113704.1.111.2632.1390443812.11.nii.gz
    │   ├── 1.2.840.113704.1.111.5624.1392092458.10.nii.gz
    │   ├── 1.2.840.113704.1.111.6756.1592183917.11.nii.gz
    │   ├── 1.2.840.113704.1.111.6896.1389252289.9.nii.gz
    │   ├── 1.2.840.113704.1.111.7780.1388040486.10.nii.gz
    │   ├── 1.2.840.113704.1.111.7956.1562030574.11.nii.gz
    │   ├── 1.2.840.113704.1.111.8660.1421889850.10.nii.gz
    │   ├── 1.2.840.113704.1.111.8776.1415860078.10.nii.gz
    │   ├── 1.2.840.113704.1.111.9536.1577060319.15.nii.gz
    │   ├── 1.3.46.670589.33.1.63700781943575774800001.5142437508376053996.nii.gz
    │   ├── 1.3.46.670589.33.1.63722560084727458900002.4851763629495772847.nii.gz
    │   └── 1.3.46.670589.33.1.63725405821017542900002.4919856832254375598.nii.gz
    └── masks
        ├── 1.2.840.113704.1.111.10192.1571886399.11.nii.gz
        ├── 1.2.840.113704.1.111.11692.1420599548.14.nii.gz
        ├── 1.2.840.113704.1.111.11716.1415240146.11.nii.gz
        ├── 1.2.840.113704.1.111.12576.1389599418.6.nii.gz
        ├── 1.2.840.113704.1.111.13172.1389599763.7.nii.gz
        ├── 1.2.840.113704.1.111.1384.1392885868.9.nii.gz
        ├── 1.2.840.113704.1.111.2452.1387439529.10.nii.gz
        ├── 1.2.840.113704.1.111.2632.1390443812.11.nii.gz
        ├── 1.2.840.113704.1.111.5624.1392092458.10.nii.gz
        ├── 1.2.840.113704.1.111.6756.1592183917.11.nii.gz
        ├── 1.2.840.113704.1.111.6896.1389252289.9.nii.gz
        ├── 1.2.840.113704.1.111.7780.1388040486.10.nii.gz
        ├── 1.2.840.113704.1.111.7956.1562030574.11.nii.gz
        ├── 1.2.840.113704.1.111.8660.1421889850.10.nii.gz
        ├── 1.2.840.113704.1.111.8776.1415860078.10.nii.gz
        ├── 1.2.840.113704.1.111.9536.1577060319.15.nii.gz
        ├── 1.3.46.670589.33.1.63700781943575774800001.5142437508376053996.nii.gz
        ├── 1.3.46.670589.33.1.63722560084727458900002.4851763629495772847.nii.gz
        └── 1.3.46.670589.33.1.63725405821017542900002.4919856832254375598.nii.gz
'''
class CommonSegmentationDS(COMMON_PAIRS_DS):
    def __init__(self, data_root, config_file, image_shape, mask_pattern='.nii.gz', transforms=None):
        '''
        config_file: generated by invoking 'split_ds' function as follows
        
        '''
        self.data_root = data_root
        self.config_file = config_file
        self.image_shape = image_shape
        self.transforms = transforms

        self.image_root = os.path.join(data_root, 'images')
        self.mask_root = os.path.join(data_root, 'masks')

        self.images_list = []
        self.masks_list = []

        series_uids = []
        with open(config_file) as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                series_uids.append(line)

        image_files = glob(os.path.join(self.image_root, '*.nii.gz'))
        # basenames = [os.path.basename(i).replace('.nii.gz', '') for i in image_files]
        basenames = series_uids
        for basename in basenames:
            image_file = os.path.join(self.image_root, '{}.nii.gz'.format(basename))
            mask_file = os.path.join(self.mask_root, '{}{}'.format(basename, mask_pattern))
            if not os.path.isfile(image_file):
                continue
            if not os.path.isfile(mask_file):
                continue
            self.images_list.append(image_file)
            self.masks_list.append(mask_file)

        print('====> data processed:\t{}'.format(len(self.images_list)))

        super().__init__(self.images_list, self.masks_list, self.image_shape, transforms)


    def __getitem__(self, item):
        subjects = super().__getitem__(item)
        image = subjects['src']['data'].float()
        mask = subjects['dst']['data'].squeeze(0).float()
        image_file = os.path.basename(subjects['src']['path'])
        mask_file = os.path.basename(subjects['dst']['path'])
        return image, mask, image_file, mask_file

    @staticmethod
    def get_inference_input(infile, image_shape):
        subject = tio.Subject(src=tio.ScalarImage(infile))
        trans = DATA_AUGMENTATION_UTILS.get_common_transform(image_shape, 'GAN_INFERENCE')
        s = trans(subject)
        return s

def test_torchio():
    src = '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/images/1.3.12.2.1107.5.1.4.60320.30000016092100163091900049818.nii.gz'
    subject = tio.Subject(src=tio.ScalarImage(src))
    trans = DATA_AUGMENTATION_UTILS.get_common_transform([32,32,32], 'GAN_INFERENCE')
    s = trans(subject)
    print('hello world!')

if __name__ == '__main__':
    test_torchio()