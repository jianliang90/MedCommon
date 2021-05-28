import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(ROOT)

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(ROOT, 'segmentation/external_lib/MedicalZooPytorch'))

from segmentation.external_lib.MedicalZooPytorch.lib.medzoo.Unet3D import UNet3D
from segmentation.external_lib.MedicalZooPytorch.lib.losses3D.dice import DiceLoss
from segmentation.datasets.common_seg_datasets import CommonSegDS
from segmentation.options.train_options import TrainOptions
from common.common_base_datasets import CommonSegmentationDS
from segmentation.runner.train_seg import SegmentationTrainer
from utils.datasets_utils import DatasetsUtils
from utils.distributed_utils import DistributedUtils
from segmentation.models.unet3d_auto_resample import ResampledUnet3D

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




seg_root = '/data/medical/cardiac/chamber/seg/chamber_seg_resampled_unified_256'


# class Options():
#     def __init__(self):
#         self.lr = 1e-3
#         self.epochs = 100
#         self.lr_fix = 50
#         self.display = 2
#         self.model_dir = './output/seg/model'


def main():
    # opts = Options()
    opts = TrainOptions().parse()

    DistributedUtils.init_distributed_mode(opts)
    if 'rank' not in opts:
        opts.rank = DistributedUtils.get_rank()

    print(opts)

    train_data_root = seg_root
    train_config_file = os.path.join(seg_root, 'config/train.txt')

    val_data_root = seg_root
    val_config_file = os.path.join(seg_root, 'config/val.txt')

    # 老版本算法
    # crop_size = [224, 224, 224]
    # train_ds = CommonSegDS(train_data_root, train_config_file, crop_size)
    # train_dataloader = DataLoader(train_ds, batch_size=1, pin_memory=True, num_workers=2, drop_last=True)
    # val_ds = CommonSegDS(val_data_root, val_config_file, crop_size)
    # val_dataloader = DataLoader(val_ds, batch_size=3, pin_memory=True, num_workers=2, drop_last=True)

    # 新版本算法
    crop_size = [352, 352, 256]
    # crop_size = [256, 256, 256]
    # crop_size = [64, 64, 64]
    train_ds = CommonSegmentationDS(train_data_root, train_config_file, crop_size)
    train_dataloader = DataLoader(train_ds, batch_size=1, pin_memory=True, num_workers=2, drop_last=True)
    val_ds = CommonSegmentationDS(val_data_root, val_config_file, crop_size)
    val_dataloader = DataLoader(val_ds, batch_size=1, pin_memory=False, num_workers=2, drop_last=True)




    # crop_size = [64, 64, 64]
    num_classes = 8

    model = ResampledUnet3D(1, num_classes, 4)

    # pretrained_file = './checkpoints/experiment_name/common_seg_epoch_9_train_0.095'
    # pretrained_file = './checkpoints/experiment_name/common_seg_epoch_98_train_0.045'
    # pretrained_file = './checkpoints/experiment_name/common_seg_epoch_99_train_0.037'
    # pretrained_file = './checkpoints/experiment_name/common_seg_epoch_87_train_0.028'
    # if pretrained_file:
    #     model.load_state_dict(torch.load(pretrained_file, map_location='cpu'))
    #     if opts.distributed:
    #         torch.distributed.barrier()

    trainer = SegmentationTrainer()
    criterion = DiceLoss(num_classes).cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=opts.lr, betas=(0.9, 0.999))
    best_loss = 1

    net = None
    if opts is not None and 'distributed' in opts and opts.distributed is True:
        import torch.distributed as dist
        torch.cuda.set_device(opts.gpu)
        model = model.cuda(opts.gpu)
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opts.gpu], output_device=opts.gpu)
    else:
        net = torch.nn.parallel.DataParallel(model)

    for epoch in range(opts.n_epochs):
        loss_train, _ = trainer.train_one_epoch(train_dataloader, net, criterion, optimizer, epoch, opts.display_freq, 'train', opts)
        # loss, _ = trainer.train_one_epoch(val_dataloader, net, criterion, optimizer, epoch, opts.display_freq, 'val')
        
        print(opts.rank)
        if opts.rank == 0:
            # if loss < best_loss:
            #     best_loss = loss
            #     print('current best val loss is:\t{}'.format(best_loss))
            #     os.makedirs(opts.model_dir, exist_ok=True)
            #     saved_model_path = os.path.join(opts.model_dir, 'common_seg_train_{:.3f}_val_{:.3f}'.format(loss_train, loss))
            #     torch.save(model.state_dict(), saved_model_path)
            #     print('====> save model:\t{}'.format(saved_model_path))

            if loss_train < best_loss:
                best_loss = loss_train
                print('current best train loss is:\t{}'.format(best_loss))
                model_dir = os.path.join(opts.checkpoints_dir, opts.name)
                os.makedirs(model_dir, exist_ok=True)                
                saved_model_path = os.path.join(model_dir, 'common_seg_epoch_{}_train_{:.3f}'.format(epoch, loss_train))
                torch.save(model.state_dict(), saved_model_path)
                print('====> save model:\t{}'.format(saved_model_path))

def load_cardic_inference_model(model_pth=None):
    if model_pth is None:
        model_pth = './common_seg_train_0.030_val_0.055'
    num_classes = 8
    model = ResampledUnet3D(1, num_classes)
    model.load_state_dict(torch.load(model_pth))
    return model


def inference():
    out_dir = './tmp_out'
    os.makedirs(out_dir, exist_ok=True)

    model_pth = './common_seg_train_0.030_val_0.055'
    series_path = '/home/proxima-sx12/data/data_category/NEWCTA/1315171/DS_CorAdSeq  0.75  Bv40  3  BestSyst 44 %'
    basename = os.path.basename(os.path.dirname(series_path))
    out_image_file = os.path.join(out_dir, '{}_image.nii.gz'.format(basename))
    out_mask_file = os.path.join(out_dir, '{}_mask_pred.nii.gz'.format(basename))

    num_classes = 8
    model = ResampledUnet3D(1, num_classes)
    model.load_state_dict(torch.load(model_pth))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    image, pred_mask = SegmentationTrainer.inference_one_case(model, series_path, is_dcm=True)

    sitk.WriteImage(image, out_image_file)
    sitk.WriteImage(pred_mask, out_mask_file)


    series_path = '/home/proxima-sx12/data/data_category/NEWMIP/1315171'
    basename = os.path.basename(os.path.dirname(series_path))
    out_image_file = os.path.join(out_dir, '{}_mip_image.nii.gz'.format(basename))
    out_mask_file = os.path.join(out_dir, '{}_mip_mask_pred.nii.gz'.format(basename))

    image, pred_mask = SegmentationTrainer.inference_one_case(model, series_path, is_dcm=True)

    sitk.WriteImage(image, out_image_file)
    sitk.WriteImage(pred_mask, out_mask_file)


if __name__ == '__main__':
    # DatasetsUtils.split_ds(os.path.join(seg_root, 'images'), os.path.join(seg_root, 'config'))
    # main()
    inference()