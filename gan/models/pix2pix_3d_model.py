import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.distributed as dist

from base_model import BaseModel
import gan_networks as networks

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(ROOT)

from utils.ssl_utils import SSL_Utils

class Pix2Pix3DModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1']
        if not self.opt.no_discriminator:
            self.loss_names.append('D_real')
            self.loss_names.append('D_fake')
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain and not opt.no_discriminator:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt)

        if self.isTrain and not opt.no_discriminator:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if not opt.no_discriminator:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))            
                self.optimizers.append(self.optimizer_D)
            if opt.ssl_sr:
                if self.opt.ssl_arch is not None and self.opt.ssl_pretrained_file is not None:
                    self.features_extractor = SSL_Utils.load_ssl_model(self.opt.ssl_arch, self.opt.ssl_pretrained_file)
                    self.features_extractor = torch.nn.Sequential(*list(self.features_extractor.children())[:1])
                    self.features_extractor.to(self.device)
                    self.loss_names.append('SR')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.mask = None
        if 'mask' in input:
            self.loss_names.append('G_L1_Mask')
            self.mask = input['mask'].to(self.device)
            if 'mask_label' in input:
                self.mask_label = input['mask_label']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        if not self.opt.no_discriminator:
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN = 0
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # Third, Mask
        if self.mask is not None:
            self.mask = (self.mask == self.mask_label)
            self.loss_G_L1_Mask = (torch.nn.L1Loss(reduction='none')(self.fake_B, self.real_B) * self.mask).sum() / (self.mask.sum()+1e-6)
            self.loss_G_L1_Mask *= self.opt.lambda_L1_Mask
            self.loss_G += self.loss_G_L1_Mask
        if self.opt.ssl_sr:
            with torch.no_grad():
                real_B_cls = self.features_extractor(self.real_B)
                fake_B_cls = self.features_extractor(self.fake_B)
                self.loss_SR = torch.nn.L1Loss()(real_B_cls, fake_B_cls)
                self.loss_G += self.loss_SR               
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        if not self.opt.no_discriminator:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        from collections import OrderedDict
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                tensor = getattr(self, name)
                coronal_name = '{}_coronal'.format(name)
                coronal_tensor = tensor[:,:,:,tensor.shape[-2]//3, :]
                sagital_name = '{}_sagital'.format(name)
                sagital_tensor = tensor[:,:,:,:, tensor.shape[-1]//3]
                axial_name = '{}_axial'.format(name)
                axial_tensor = tensor[:,:,tensor.shape[-3]//3,:,:]
                # visual_ret[name] = getattr(self, name)
                # visual_ret[coronal_name] = coronal_tensor
                # visual_ret[sagital_name] = sagital_tensor
                visual_ret[axial_name] = axial_tensor
        return visual_ret