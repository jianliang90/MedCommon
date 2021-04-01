import os
import sys

import torch

# sys.path.append('/home/zhangwd/code/work/BrainSolution/gan/')
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import models
from models.pix2pix_3d_model_old import Pix2PixModel

class Options():
    def __init__(self):
        self.lr = 2e-4
        self.beta1 = 0.5
        self.gan_mode = 'lsgan'
        self.direction = 'AtoB'
        self.lambda_L1 = 2
        self.epochs = 10000
        self.num_workers = 16
        self.batch_size = 4
        self.pin_memory = True
        self.display = 2
        self.save_interval = 10
        self.model_save_interval = 25
        self.intermidiate_result_root = '../../data/gan/hospital_6_crop/experiment_registration2/8.2.out/train_result/intermidiate_result_{}'.format(__file__.split('.')[0])
        self.save_dir = '../../data/gan/hospital_6_crop/experiment_registration2/9.2.model_out/model_{}'.format(__file__.split('.')[0])
        # add patch discriminator
        self.patch_D = False
        self.num_patches_D = 5
        self.patch_size_D = [64, 64, 64]
        # crop_size
        self.crop_size = [64, 448, 448]
        # self.crop_size = [8, 8, 8]

        self.root_dir = '../../data/gan/hospital_6_crop/experiment_registration2/8.2.out'
        self.config_file = '../../data/gan/hospital_6_crop/experiment_registration2/8.2.out/config/anno_mask_ncct_to_dwi_bxxx_train_config_file.txt'
        self.check_point = None
        self.netG_model_path = '../../data/gan/hospital_6_crop/experiment_registration2/9.2.model_out/model_train_cta_to_dwi_bxxx_hospital6_nonmask_20200508/pixel2pixel_netG_epoch_950_loss_9.0383.pth'
        # self.netD_model_path = '../../data/gan/ncct2dwi/experiment_registration2/9.model_out/model_train_ncct_to_dwi_bxxx_20200421/pixel2pixel_netD_epoch_100_loss_0.2630.pth'
        self.netG_model_path = None
        self.netD_model_path = None


opt = Options()

model = Pix2PixModel(opt)

print(model.netG)

for i in range(50):
    real_a = torch.randn(1,1,32,448,224).float()
    real_b = torch.randn(1,1,32,448,224).float()
    input = {}
    input['A'] = real_a
    input['B'] = real_b
    input['A_paths'] = 'A'
    input['B_paths'] = 'B'
    model.set_input(input)
    model.optimize_parameters()