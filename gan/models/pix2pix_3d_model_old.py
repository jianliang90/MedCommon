import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.distributed as dist

from base_model import BaseModel
import gan_networks as networks

class Pix2PixModel():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.netG_cpu = networks.ResnetGenerator(1,1, 32, n_blocks=6)
        self.netD_cpu = networks.PixelDiscriminator(2,8)
        if self.opt.patch_D:
            self.netD_P_cpu = PixelDiscriminator(2,8)

        if opt.netG_model_path is not None:
            self.netG_cpu.load_state_dict(torch.load(opt.netG_model_path))
        if opt.netD_model_path is not None:
            self.netD_cpu.load_state_dict(torch.load(opt.netD_model_path))
        
        self.isTrain = True
        self.optimizers=[]
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).cuda()
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionMaskLBP = MaskLBPLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG_cpu.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_cpu.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if self.opt.patch_D:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.netD_P = torch.nn.DataParallel(self.netD_P).cuda()
        self.netG = torch.nn.DataParallel(self.netG_cpu).cuda()
        self.netD = torch.nn.DataParallel(self.netD_cpu).cuda()


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].cuda()
        self.real_B = input['B' if AtoB else 'A'].cuda()
        if 'mask' in input:
            self.mask = input['mask'].cuda()
        else:
            self.mask = None
        if 'lbp_mask' in input:
            self.lbp_mask = input['lbp_mask']
        else:
            self.lbp_mask = None
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
    def forward(self):
        self.netG = torch.nn.DataParallel(self.netG_cpu).cuda()
        self.netD = torch.nn.DataParallel(self.netD_cpu).cuda()
        # self.netG.train()
        # self.netD.train()
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

        # add patch(block) discriminator calculate 
        # related params: patch_D(bool), num_patches_D(int), patch_size_D([z,y,x])
        if self.opt.patch_D:
            self.fake_B_patch = []
            self.real_B_patch = []
            self.real_A_patch = []
            d = self.real_A.size(2)
            h = self.real_A.size(3)
            w = self.real_A.size(4)

            for i in range(self.opt.num_patches_D):
                max_end_d = d - self.opt.patch_size_D[0]
                max_end_h = h - self.opt.patch_size_D[1]
                max_end_w = w - self.opt.patch_size_D[2]

                # print('d:{}\th:{}\tw:{}'.format(d,h,w))
                # print('max_end_d:{}\tmax_end_h:{}\tmax_end_w:{}'.format(max_end_d,max_end_h,max_end_w))
                start_d = np.random.randint(0, max_end_d)
                start_h = np.random.randint(0, max_end_h)
                start_w = np.random.randint(0, max_end_w)

                end_d = start_d + self.opt.patch_size_D[0]
                end_h = start_h + self.opt.patch_size_D[1]
                end_w = start_w + self.opt.patch_size_D[2]

                self.fake_B_patch.append(self.fake_B[:,:,start_d:end_d, start_h:end_h, start_w:end_w])
                self.real_B_patch.append(self.real_B[:,:,start_d:end_d, start_h:end_h, start_w:end_w])
                self.real_A_patch.append(self.real_A[:,:,start_d:end_d, start_h:end_h, start_w:end_w])

        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.mask is None:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        else:
            fake_AB = torch.cat((self.real_A*self.mask, self.fake_B*self.mask), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        if self.mask is None:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        else:
            real_AB = torch.cat((self.real_A*self.mask, self.real_B*self.mask), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_D_P(self):
        self.loss_D_fake_patch = 0
        self.loss_D_real_patch = 0
        #from .networks import cal_gradient_penalty
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        for i in range(self.opt.num_patches_D):
            fake_AB_patch = torch.cat((self.real_A_patch[i], self.fake_B_patch[i]), 1)
            pred_fake_AB_patch = self.netD_P(fake_AB_patch.detach())
            self.loss_D_fake_patch += self.criterionGAN(pred_fake_AB_patch, False)

        for i in range(self.opt.num_patches_D):
            real_AB_patch = torch.cat((self.real_A_patch[i], self.real_B_patch[i]), 1)
            pred_real_AB_patch = self.netD_P(real_AB_patch.detach())
            self.loss_D_real_patch += self.criterionGAN(pred_real_AB_patch, True)

        self.loss_D_fake_patch = self.loss_D_fake_patch/self.opt.num_patches_D
        self.loss_D_real_patch = self.loss_D_real_patch/self.opt.num_patches_D
        self.loss_D_patch = (self.loss_D_fake_patch + self.loss_D_real_patch)/2
        self.loss_D_patch.backward(retain_graph=False)



    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.mask is None:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        else:
            fake_AB = torch.cat((self.real_A*self.mask, self.fake_B*self.mask), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        if self.mask is None and self.lbp_mask is None:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        elif self.lbp_mask is not None:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 + self.criterionMaskLBP(self.fake_B, self.real_B, self.lbp_mask).float()
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B*self.mask, self.real_B*self.mask) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        
        # update D patches
        if self.opt.patch_D:
            self.set_requires_grad(self.netD_P, True)  # enable backprop for D
            self.optimizer_D_P.zero_grad()  # set D's gradients to zero
            self.backward_D_P()  # calculate gradients for D
            self.optimizer_D_P.step()  # update D's weights
        
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        if self.opt.patch_D:
            self.set_requires_grad(self.netD_P, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        
        # print(self.loss_G)
        # print(self.loss_D)
        # print('\n')


    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        netG_out_model_file = 'pixel2pixel_netG_epoch_{}_loss_{:.4f}.pth'.format(epoch, self.loss_G.detach().cpu().numpy())
        torch.save(self.netG_cpu.cpu().state_dict(), 
            os.path.join(self.save_dir, netG_out_model_file))
        netD_out_model_file = 'pixel2pixel_netD_epoch_{}_loss_{:.4f}.pth'.format(epoch, self.loss_D.detach().cpu().numpy())    
        torch.save(self.netD_cpu.cpu().state_dict(), 
            os.path.join(self.save_dir, netD_out_model_file))

        print('====> save model:\t{}'.format(netG_out_model_file))
        print('====> save model:\t{}'.format(netD_out_model_file))

