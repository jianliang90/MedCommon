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
from segmentation.options.train_options import TrainOptions
# from datasets.lung_seg_datasets import AirwayCoarseSeg_DS
from utils.data_io_utils import DataIO
from utils.datasets_utils import DatasetsUtils
from utils.distributed_utils import DistributedUtils
from utils.data_aug_utils import DATA_AUGMENTATION_UTILS
from common.common_base_datasets import CommonSegmentationDS

import torch
import torch.nn

from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.misc_utils import AverageMeter
from utils.datasets_utils import DatasetsUtils
import time
from tqdm import tqdm

import argparse
import SimpleITK as sitk

import fire

import math

class SegmentationTrainer:
    def __init__(self, amp=False):
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1, enabled=False)
        self.amp = amp

    def train_one_epoch(self, dataloader, model, criterion, optimizer, epoch, display, phase='train', opt=None):

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
        # for num_iter, (subjects) in tqdm(enumerate(dataloader)):
            # print('images:\t', images.shape)
            # print('masks:\t', masks.shape)
            data_time.update(time.time() - end)

            # output = model(images.cuda())
            # final_pred = output
            # final_gt = masks
            # loss = criterion(output, masks.cuda())[0]

            if self.amp:
                with torch.cuda.amp.autocast():
                    output = model(images.cuda())
                    final_pred = output
                    final_gt = masks
                    loss = criterion(output, masks.cuda())[0]
            else:
                output = model(images.cuda())
                final_pred = output
                final_gt = masks
                loss = criterion(output, masks.cuda())[0]                

            if opt is not None and opt.rank == 0:
                if (num_iter+1)%display == 0:
                    print('loss*:\t', loss.detach().cpu().numpy(), '\t', loss.detach().cpu().numpy().dtype)

            if phase == 'train':
                if self.amp:
                    optimizer.zero_grad()
                    scaled_loss = self.scaler.scale(loss)
                    print('before scale:\t', loss)
                    print('after scale:\t', scaled_loss)
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()                    


            batch_time.update(time.time()-end)
            end = time.time()
            losses.update(loss.detach().cpu().numpy(), len(images))
            if opt is not None and opt.rank == 0:
                if (num_iter+1)%display == 0:
                    print_info = '[{}]\tEpoch: [{}][{}/{}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\tData {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\t'.format(
                        phase, epoch, num_iter, len(dataloader), batch_time=batch_time, data_time=data_time, loss=losses)
                    print(print_info)
                    print(losses.sum, '\t', losses.count)
                    print(loss.detach().cpu().numpy())
                    print(self.scaler.state_dict())
                    logger.append(print_info)
        
                    # output middle result
                    # if opt.rank == 0:
                    #     pred_mask = torch.nn.functional.sigmoid(output).argmax(1)    
                    #     pred_mask_uint8 = np.array(pred_mask.detach().cpu().numpy(), np.uint8)
                    #     gt_mask_uint8 = np.array(masks.numpy(), np.uint8)
                    #     pred_mask_sitk = sitk.GetImageFromArray(pred_mask_uint8[0])
                    #     gt_mask_sitk = sitk.GetImageFromArray(gt_mask_uint8[0])
                    #     image_sitk = sitk.GetImageFromArray(np.array(images[0].squeeze().cpu().numpy(), dtype=np.int16))
                    #     gt_mask_sitk.CopyInformation(image_sitk)
                    #     pred_mask_sitk.CopyInformation(image_sitk)
                    #     sitk.WriteImage(pred_mask_sitk, 'pred_mask_{}_{}.nii.gz'.format(epoch, num_iter))
                    #     sitk.WriteImage(gt_mask_sitk, 'gt_mask_{}_{}.nii.gz'.format(epoch, num_iter))
                    #     sitk.WriteImage(image_sitk, 'image_{}_{}.nii.gz'.format(epoch, num_iter))
        return losses.avg, logger

    @staticmethod
    def load_model(opts):
        num_classes = opts.num_classes
        base_n_filter = opts.base_n_filter
        model = ResampledUnet3D(1, num_classes, base_n_filter)
        model.load_state_dict(torch.load(opts.weights, map_location='cpu'))
        return model

    @staticmethod
    def inference_one_case(model, series_path, is_dcm=True, dst_size = [128, 128, 128]):
        # 注意在处理分割模型时，模型的模式model.eval()，和模型训练时一致
        model.eval()

        if is_dcm:
            image_data = DataIO.load_dicom_series(series_path)
        else:
            image_data = DataIO.load_nii_image(series_path)
        image = image_data['sitk_image']
        # dst_size = [128, 128, 128]
        resampled_image = DatasetsUtils.resample_image_unsame_resolution(image, dst_size, sitk.sitkLinear)
        resampled_image_arr = sitk.GetArrayFromImage(resampled_image)
        resampled_image_tensor = torch.from_numpy(resampled_image_arr).unsqueeze(0).unsqueeze(0).float()
        output = model(resampled_image_tensor.cuda())

        pred_mask = torch.nn.functional.sigmoid(output).argmax(1)    
        pred_mask_uint8 = np.array(pred_mask.detach().cpu().numpy(), np.uint8)
        pred_mask_uint8 = pred_mask_uint8[0]
        pred_sitk_mask = sitk.GetImageFromArray(pred_mask_uint8)
        pred_sitk_mask.CopyInformation(resampled_image)

        ori_pred_sitk_mask = DatasetsUtils.restore_ori_image_from_resampled_image(pred_sitk_mask, image)
        
        ori_pred_sitk_mask.CopyInformation(image)

        return image, ori_pred_sitk_mask

    @staticmethod
    def inference_one_case1(model, series_path, crop_size = [128, 128, 128], out_root=None):
        subject = CommonSegmentationDS.get_inference_input(series_path, crop_size)
        input = subject['src']['data'].float().unsqueeze(0)
        model.eval()
        output = model(input)
        pred_mask = torch.nn.functional.sigmoid(output).argmax(1)    
        pred_mask_uint8 = np.array(pred_mask.detach().cpu().numpy(), np.uint8)
        pred_mask_uint8 = pred_mask_uint8[0]
        
        ori_image = sitk.ReadImage(series_path)
        cropped_image_arr = input.squeeze().cpu().numpy()
        cropped_image_arr = np.transpose(cropped_image_arr, [2,1,0])
        cropped_image_arr = np.array(cropped_image_arr, np.int16)

        cropped_image = sitk.GetImageFromArray(cropped_image_arr)
        cropped_image.SetSpacing(ori_image.GetSpacing())
        cropped_image.SetDirection(ori_image.GetDirection())
        cropped_image.SetOrigin(ori_image.GetOrigin())

        pred_mask_uint8 = np.transpose(pred_mask_uint8, [2,1,0])
        cropped_mask = sitk.GetImageFromArray(pred_mask_uint8)
        cropped_mask.CopyInformation(cropped_image)

        if out_root:
            os.makedirs(out_root, exist_ok=True)
            basename = os.path.basename(series_path).replace('.nii.gz', '')
            out_image_file = os.path.join(out_root, '{}_image.nii.gz'.format(basename))
            out_mask_file = os.path.join(out_root, '{}_mask.nii.gz'.format(basename))
            sitk.WriteImage(cropped_image, out_image_file)
            sitk.WriteImage(cropped_mask, out_mask_file)
        
        print('hello world!')




def train():
    opts = TrainOptions().parse()
    DistributedUtils.init_distributed_mode(opts)
    if 'rank' not in opts:
        opts.rank = DistributedUtils.get_rank()

    print(opts)

    seg_root = opts.dataroot
    train_data_root = seg_root
    train_config_file = os.path.join(seg_root, 'config/train.txt')

    val_data_root = seg_root
    val_config_file = os.path.join(seg_root, 'config/val.txt')

    transforms = None
    if opts.aug == 'inference':
        transforms = DATA_AUGMENTATION_UTILS.get_common_transform(opts.crop_size, 'GAN_INFERENCE')
    elif opts.aug == 'seg_train':
        transforms = DATA_AUGMENTATION_UTILS.get_common_transform(opts.crop_size, 'GAN')

    train_ds = CommonSegmentationDS(train_data_root, train_config_file, opts.crop_size)
    train_dataloader = DataLoader(train_ds, batch_size=1, pin_memory=True, num_workers=2, drop_last=True)
    # val_ds = CommonSegmentationDS(val_data_root, val_config_file, opts.crop_size)
    # val_dataloader = DataLoader(val_ds, batch_size=1, pin_memory=False, num_workers=2, drop_last=True)

    num_classes = opts.num_classes
    base_n_filter = opts.base_n_filter
    model = ResampledUnet3D(1, num_classes, base_n_filter)
    
    # pretrained_file = './checkpoints/experiment_name/common_seg_epoch_87_train_0.028'
    if opts.weights:
        model.load_state_dict(torch.load(opts.weights, map_location='cpu'))
        if opts.distributed:
            torch.distributed.barrier()
   
    
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
            if loss_train < best_loss:
                best_loss = loss_train
                print('current best train loss is:\t{}'.format(best_loss))
                model_dir = os.path.join(opts.checkpoints_dir, opts.name)
                os.makedirs(model_dir, exist_ok=True)                
                saved_model_path = os.path.join(model_dir, 'common_seg_epoch_{}_train_{:.3f}'.format(epoch, loss_train))
                torch.save(model.state_dict(), saved_model_path)
                print('====> save model:\t{}'.format(saved_model_path))

def inference(infile, out_root, cropped_size = [288, 288, 320], weights=None):
    opts = TrainOptions().parse()
    if weights:
        opts.weights = weights
    model = SegmentationTrainer.load_model(opts)
    SegmentationTrainer.inference_one_case1(model, infile, cropped_size, out_root)

def test_inference():
    opts = TrainOptions().parse()
    model = SegmentationTrainer.load_model(opts)
    series_path = '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/images/1.3.12.2.1107.5.1.4.60320.30000016092100163091900049818.nii.gz'
    series_path = '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/images/1.3.12.2.1107.5.1.4.60320.30000019042300091141600091595.nii.gz'
    series_path = '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/images/1.3.12.2.1107.5.1.4.60320.30000018122400284654800011514.nii.gz'
    series_path = '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/images/1.3.12.2.1107.5.1.4.60320.30000019011400245145200059829.nii.gz'
    
    out_root = '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/inference'
    cropped_size = [288, 288, 320]
    SegmentationTrainer.inference_one_case1(model, series_path, cropped_size, out_root)



if __name__ == '__main__':
   test_inference() 