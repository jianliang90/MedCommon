import os
import sys
import numpy as np

MEDCOMMON_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)
print(MEDCOMMON_ROOT)
CURRENT_EXP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir)
sys.path.append(MEDCOMMON_ROOT)
sys.path.append(os.path.join(MEDCOMMON_ROOT, 'segmentation/external_lib/MedicalZooPytorch'))
sys.path.append(CURRENT_EXP_ROOT)

from segmentation.external_lib.MedicalZooPytorch.lib.medzoo.Unet3D import UNet3D
from segmentation.external_lib.MedicalZooPytorch.lib.losses3D.dice import DiceLoss
from segmentation.models.unet3d_auto_resample import ResampledUnet3D
from datasets.lung_seg_datasets import AirwayCoarseSeg_DS
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

class Options():
    def __init__(self):
        self.lr = 1e-3
        self.epochs = 100
        self.lr_fix = 50
        self.display = 2
        self.model_dir = '../data/seg/model'


def train_one_epoch(dataloader, model, criterion, optimizer, epoch, display, phase='train'):
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
    sitk.WriteImage(pred_mask_sitk, 'pred_mask_{}.nii.gz'.format(epoch))
    sitk.WriteImage(gt_mask_sitk, 'gt_mask_{}.nii.gz'.format(epoch))
    return losses.avg, logger


def inference(img_path, model_pth, out_dir, is_dcm=True):
    '''
    debug cmd: inference('../data/seg/nii_file/1.3.12.2.1107.5.1.4.60320.30000015012900333934300003426/img.nii.gz', '../data/seg/model/cardiac_seg_train_0.013_val_0.020', '../data/seg/inference')
    debug cmd: inference('/fileser/zhangwd/data/lung/changzheng/airway/airway_20201030/images/1.2.840.113704.1.111.10192.1571886399.11', '../data/seg/model/cardiac_seg_train_0.020_val_0.052', '../data/seg/inference')
    '''
    
    if is_dcm:
        data = DataIO.load_dicom_series(img_path)
    else:
        data = DataIO.load_nii_image(img_path)
    image_arr = data['image']
    sitk_img = data['sitk_image']

    image_tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).float()
    num_classes = 4
    model = ResampledUnet3D(1, num_classes)
    assert model_pth is not None
    model.load_state_dict(torch.load(model_pth))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    output = model(image_tensor.cuda())
    pred_mask = torch.nn.functional.sigmoid(output).argmax(1)    
    pred_mask_uint8 = np.array(pred_mask.detach().cpu().numpy(), np.uint8)
    pred_mask_uint8 = pred_mask_uint8[0]
    in_sitk_mask = sitk.GetImageFromArray(pred_mask_uint8)
    in_sitk_mask.CopyInformation(sitk_img)

    os.makedirs(out_dir, exist_ok=True)
    sitk.WriteImage(sitk_img, os.path.join(out_dir, 'image_raw.nii.gz'))
    sitk.WriteImage(in_sitk_mask, os.path.join(out_dir, 'mask_pred.nii.gz'))
    print('hello world!')


def main():
    opts = Options()

    image_root = '/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/image_refine/ori_128_128_128'
    mask_root = '/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/mask_refine/ori_128_128_128'

    train_config_file = '/fileser/zhangfan/DataSet/airway_segment_data/csv/train_filename.csv'
    val_config_file = '/fileser/zhangfan/DataSet/airway_segment_data/csv/val_filename.csv'

    crop_size = [128, 128, 128]
    # crop_size = [64, 64, 64]
    num_classes = 4
    train_ds = AirwayCoarseSeg_DS(image_root, mask_root, train_config_file, crop_size)
    train_dataloader = DataLoader(train_ds, batch_size=8, pin_memory=True, num_workers=2, drop_last=True)
    
    val_ds = AirwayCoarseSeg_DS(image_root, mask_root, val_config_file, crop_size)
    val_dataloader = DataLoader(val_ds, batch_size=4, pin_memory=False, num_workers=2, drop_last=True)

    model = ResampledUnet3D(1, num_classes)
    criterion = DiceLoss(num_classes).cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=opts.lr, betas=(0.9, 0.999))
    best_loss = 1
    for epoch in range(opts.epochs):
        loss_train, _ = train_one_epoch(train_dataloader, torch.nn.DataParallel(model).cuda(), criterion, optimizer, epoch, opts.display)
        loss, _ = train_one_epoch(val_dataloader, torch.nn.DataParallel(model).cuda(), criterion, optimizer, epoch, opts.display, 'val')
        if loss < best_loss:
            best_loss = loss
            print('current best val loss is:\t{}'.format(best_loss))
            os.makedirs(opts.model_dir, exist_ok=True)
            saved_model_path = os.path.join(opts.model_dir, 'cardiac_seg_train_{:.3f}_val_{:.3f}'.format(loss_train, loss))
            torch.save(model.cpu().state_dict(), saved_model_path)
            print('====> save model:\t{}'.format(saved_model_path))





if __name__ == '__main__':
    # fire.Fire()
    # main()
    # inference('../data/seg/nii_file/1.3.12.2.1107.5.1.4.60320.30000015012900333934300003426/img.nii.gz', '../data/seg/model/cardiac_seg_train_0.013_val_0.020', '../data/seg/inference/test')
    # inference('../../data/changzheng/airway/airway_20201030/paires_croped_by_coarse_lung_seg/images/1.2.840.113704.1.111.10192.1571886399.11.nii.gz', '../data/seg/model/cardiac_seg_train_0.105_val_0.095', '../data/seg/inference/test')
    inference('/fileser/zhangwd/data/lung/changzheng/airway/airway_20201030/images/1.2.840.113704.1.111.10192.1571886399.11', '../data/seg/model/cardiac_seg_train_0.020_val_0.052', '../data/seg/inference')
