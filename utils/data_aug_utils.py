import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print('solution root:\t', root)

sys.path.append(root)
sys.path.append(os.path.join(root, 'external_lib/torchio'))

import torchio as tio

class DATA_AUGMENTATION_UTILS:
    def __init__(self):
        pass

    @staticmethod
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
