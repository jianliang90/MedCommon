import os
import sys

import torch
import torchvision

gan_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(gan_root, 'external_lib/torchio'))


import torchio as tio
from torch.utils.data import Dataset, DataLoader
from utils.data_aug_utils import DATA_AUGMENTATION_UTILS

def get_common_transform(image_shape, type='GAN'):
    default_transform = None
    if type == 'GAN':
        default_transform = tio.Compose([
            tio.RandomFlip(axes=[0,1,2]), 
            tio.RandomAnisotropy(downsampling=(1,2.5),scalars_only=False,p=0.25),              # make images look anisotropic 25% of times
            tio.OneOf({                                # either
                tio.RandomCropOrPad((image_shape[0], image_shape[1], image_shape[2])): 0.8,
                tio.CropOrPad((image_shape[0], image_shape[1], image_shape[2])):0.2,   # or random elastic deformation
            }),
            tio.RandomBlur(p=0.25),                    # blur 25% of times
            tio.RandomNoise(p=0.25)
        ])
    elif type == 'DEBUG':
        default_transform = tio.Compose([
            tio.RandomAnisotropy(p=0.999),              # make images look anisotropic 25% of times
            tio.CropOrPad((image_shape[0], image_shape[1], image_shape[2]))
        ])
    elif type == 'GAN_INFERENCE':
        default_transform = tio.Compose([
            tio.CropOrPad((image_shape[0], image_shape[1], image_shape[2]))
        ])        
    else:
        default_transform = tio.Compose([
            tio.RandomFlip(axes=[0,1,2]), 
            tio.RandomAnisotropy(p=0.25),              # make images look anisotropic 25% of times
            tio.CropOrPad((image_shape[0], image_shape[1], image_shape[2])),            # tight crop around brain
            tio.RandomBlur(p=0.99999),                    # blur 25% of times
            tio.RandomNoise(p=0.25),                   # Gaussian noise 25% of times
            tio.OneOf({                                # either
                tio.RandomAffine(): 0.8,               # random affine
                tio.RandomElasticDeformation(): 0.2,   # or random elastic deformation
            }, p=0.8),                                 # applied to 80% of images
            tio.RandomBiasField(p=0.3),                # magnetic field inhomogeneity 30% of times
            tio.OneOf({                                # either
                tio.RandomMotion(): 1,                 # random motion artifact
                tio.RandomSpike(): 2,                  # or spikes
                tio.RandomGhosting(): 2,               # or ghosts
            }, p=0.5), 
        ])
    return default_transform 

class GAN_COMMON_DS(Dataset):
    def __init__(self, data_root, src_pattern, dst_pattern, image_shape, transforms=None, mask_pattern=None):
        self.data_root = data_root
        self.src_pattern = src_pattern
        self.dst_pattern = dst_pattern
        self.mask_pattern = mask_pattern

        if transforms:
            self.transforms = transforms
        else:
            default_transform = tio.Compose([
                tio.RandomFlip(axes=[0,1,2]), 
                tio.RandomAnisotropy(p=0.25),              # make images look anisotropic 25% of times
                tio.CropOrPad((image_shape[0], image_shape[1], image_shape[2])),            # tight crop around brain
                tio.RandomBlur(p=0.99999),                    # blur 25% of times
                tio.RandomNoise(p=0.25),                   # Gaussian noise 25% of times
                tio.OneOf({                                # either
                    tio.RandomAffine(): 0.8,               # random affine
                    tio.RandomElasticDeformation(): 0.2,   # or random elastic deformation
                }, p=0.8),                                 # applied to 80% of images
                tio.RandomBiasField(p=0.3),                # magnetic field inhomogeneity 30% of times
                tio.OneOf({                                # either
                    tio.RandomMotion(): 1,                 # random motion artifact
                    tio.RandomSpike(): 2,                  # or spikes
                    tio.RandomGhosting(): 2,               # or ghosts
                }, p=0.5), 
            ])
            self.transforms = default_transform


        pids = os.listdir(data_root)

        src_files = []
        dst_files = []
        mask_files = []
        subjects = []
        for pid in pids:
            src_file = os.path.join(data_root, str(pid), src_pattern)
            dst_file = os.path.join(data_root, str(pid), dst_pattern)
            if self.mask_pattern:
                mask_file = os.path.join(data_root, str(pid), self.mask_pattern)
            exist = True
            if not os.path.isfile(src_file):
                print('====> {} not exist!'.format(src_file))
                exist = False
            if not os.path.isfile(dst_file):
                print('====> {} not exist!'.format(dst_file))
                exist = False
            if self.mask_pattern and not os.path.isfile(mask_file):
                print('====> {} not exist!'.format(mask_file))
                exist = False
            if not exist:
                continue
            src_files.append(src_file)
            dst_files.append(dst_file)
            if self.mask_pattern:
                mask_files.append(mask_file)


        if self.mask_pattern:
            for (src, dst, mask) in zip(src_files, dst_files, mask_files):
                subject = tio.Subject(src=tio.ScalarImage(src), dst=tio.LabelMap(dst), mask=tio.LabelMap(mask))
                subjects.append(subject)            
        else:
            for (src, dst) in zip(src_files, dst_files):
                subject = tio.Subject(src=tio.ScalarImage(src), dst=tio.LabelMap(dst))
                # subject = tio.Subject(src=tio.ScalarImage(src), dst=tio.ScalarImage(dst))
                subjects.append(subject)

        self.subjects_dataset = tio.SubjectsDataset(subjects, transform=self.transforms)

    
    def __len__(self):
        return self.subjects_dataset.__len__()

    def __getitem__(self, item):
        return self.subjects_dataset.__getitem__(item)

    @staticmethod
    def get_inference_input(infile, image_shape):
        subject = tio.Subject(src=tio.ScalarImage(infile))
        trans = DATA_AUGMENTATION_UTILS.get_common_transform(image_shape, 'GAN_INFERENCE')
        s = trans(subject)
        return s


class GANDataWrapper:
    def __init__(self):
        pass
    @staticmethod
    def get_processed_image(transforms, infile):
        subject = tio.Subject(src=tio.ScalarImage(infile))
        subjects = []
        subjects.append(subject)
        subjects_dataset = tio.SubjectsDataset(subjects, transform=transforms)
        return subjects_dataset.__getitem__(0)


