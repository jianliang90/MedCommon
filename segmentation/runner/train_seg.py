import os
import sys
import numpy as np

MEDCOMMON_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir)
print(MEDCOMMON_ROOT)
# CURRENT_EXP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir)
sys.path.append(MEDCOMMON_ROOT)
sys.path.append(os.path.join(MEDCOMMON_ROOT, 'segmentation/external_lib/MedicalZooPytorch'))
# sys.path.append(CURRENT_EXP_ROOT)

from segmentation.external_lib.MedicalZooPytorch.lib.medzoo.Unet3D import UNet3D
from segmentation.external_lib.MedicalZooPytorch.lib.losses3D.dice import DiceLoss
from segmentation.models.unet3d_auto_resample import ResampledUnet3D
# from datasets.lung_seg_datasets import AirwayCoarseSeg_DS
from utils.data_io_utils import DataIO

import torch
import torch.nn

from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.misc_utils import AverageMeter
import time
from tqdm import tqdm

import argparse
import SimpleITK as sitk

import fire

class SegmentationTrainer:
    def __init__(self):
        pass

    def train_one_epoch(self, dataloader, model, criterion, optimizer, epoch, display, phase='train'):
        # model.train()
        if phase == 'train':
            model.eval()
        else:
            model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        logger = []
        end = time.time()
        final_pred = None
        final_gt = None
        for num_iter, (images, masks, image_files, mask_files) in tqdm(enumerate(dataloader)):
            # print('images:\t', images.shape)
            # print('masks:\t', masks.shape)
            data_time.update(time.time() - end)
            output = model(images.cuda())
            final_pred = output
            final_gt = masks
            loss = criterion(output, masks.cuda())[0]
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_time.update(time.time()-end)
            end = time.time()
            losses.update(loss.detach().cpu().numpy(), len(images))
            if (num_iter+1)%display == 0:
                print_info = '[{}]\tEpoch: [{}][{}/{}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\tData {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\t'.format(
                    phase, epoch, num_iter, len(dataloader), batch_time=batch_time, data_time=data_time, loss=losses)
                print(print_info)
                logger.append(print_info)
        pred_mask = torch.nn.functional.sigmoid(output).argmax(1)    
        pred_mask_uint8 = np.array(pred_mask.detach().cpu().numpy(), np.uint8)
        gt_mask_uint8 = np.array(masks.numpy(), np.uint8)
        pred_mask_sitk = sitk.GetImageFromArray(pred_mask_uint8[0])
        gt_mask_sitk = sitk.GetImageFromArray(gt_mask_uint8[0])
        image_sitk = sitk.GetImageFromArray(np.array(images[0].squeeze().cpu().numpy(), dtype=np.int16))
        sitk.WriteImage(pred_mask_sitk, 'pred_mask_{}.nii.gz'.format(epoch))
        sitk.WriteImage(gt_mask_sitk, 'gt_mask_{}.nii.gz'.format(epoch))
        sitk.WriteImage(image_sitk, 'image_{}.nii.gz'.format(epoch))
        return losses.avg, logger

    

