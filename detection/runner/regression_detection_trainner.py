import os
import sys
import torch
import torch.nn as nn

MEDCOMMON_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir)
sys.path.append(MEDCOMMON_ROOT)
sys.path.append(os.path.join(MEDCOMMON_ROOT, 'external_lib'))

from tqdm import tqdm
import time
import numpy as np

from utils.misc_utils import AverageMeter
from detection.models.detection_auto_resample import RegressionDetecter
from detection.datasets.position_detection_common_ds import PositionDetectionDS
from utils.detection_utils import PYTORCH_TENSOR_DETECTION_UTILS
from utils.lr_adjust_utils import LR_ADJUST_UTILS
from detection.models.detection_auto_resample import RegressionDetecter
from detection.datasets.position_detection_common_ds import PositionDetectionDS
from utils.detection_utils import PYTORCH_TENSOR_DETECTION_UTILS
from torch.utils.data import Dataset, DataLoader

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='common detection algorithm')
    parser.add_argument('--dataroot', type=str, default='/data/medical/brain/cerebral_parenchyma/exp/cta', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--weights', type=str, default=None, help='pretrained weights file')

    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--boundary_info_file', type=str, default='/data/medical/brain/cerebral_parenchyma/exp/cta/config/mask_boundary_info.json', help='file to record mask boundary information')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--n_objects', type=int, default=1, help='n objects to be detected')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[128,128,128])
    parser.add_argument('--aug', default='inference')
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--n_epochs', type=int, default=100)
    return parser.parse_args()

class RegressionDetectionTrainer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def train_one_epoch(dataloader, model, criterion, optimizer, epoch, display, phase='train', opt=None):
        if phase == 'train':
            model.train()
        else:
            model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        logger = []
        ious = np.array([], dtype=np.float)
        for num_iter, (images, targets, names) in tqdm(enumerate(dataloader)):
            data_time.update(time.time() - end)
            output = model(images.cuda())
            if criterion is not None:
                loss = criterion(output, targets.cuda())
            else:
                iou,_ = PYTORCH_TENSOR_DETECTION_UTILS.calc_brick_iou(output, targets.view(targets.shape[0],-1).cuda())
                l1_loss = torch.nn.L1Loss()(output, targets.view(targets.shape[0],-1).cuda())
                loss = l1_loss + (1-iou.mean())
                ious = np.append(ious, iou.detach().cpu())
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   
            batch_time.update(time.time()-end)
            end = time.time()  
            losses.update(loss.detach().cpu().numpy(), len(images))
            if (opt is not None and opt.rank == 0) or (opt is None):
                if (num_iter+1)%display == 0:
                    print_info = '[{}]\tEpoch: [{}][{}/{}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\tData {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\t'.format(
                        phase, epoch, num_iter, len(dataloader), batch_time=batch_time, data_time=data_time, loss=losses)
                    print(print_info)
                    # print(losses.sum, '\t', losses.count)
                    # print(loss.detach().cpu().numpy())
                    logger.append(print_info)
        return ious                                         

def train():
    opts = parse_args()
    root = opts.dataroot
    boundary_info_file = opts.boundary_info_file
    ds = PositionDetectionDS(root, boundary_info_file=boundary_info_file)
    dataloader = DataLoader(ds, batch_size=opts.batch_size)

    n_objects = opts.n_objects
    net_args = {
        'input_size': [128, 128, 128], 
        'arch':opts.arch
    }
    model = RegressionDetecter(n_objects, net_args)

    criterion = None
    lr = opts.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)

    n_epochs = opts.n_epochs
    best_iou = 0
    for epoch in range(n_epochs):
        LR_ADJUST_UTILS.adjust_learning_rate(optimizer, epoch, opts.lr, True, n_epochs)
        ious = RegressionDetectionTrainer.train_one_epoch(dataloader, model.cuda(), criterion, optimizer, epoch, 1, phase='train', opt=None)
        if ious.mean() > best_iou:
            best_iou = ious.mean()
            print('current best train iou is:\t{}'.format(best_iou))
            model_dir = os.path.join(opts.checkpoints_dir, opts.name)
            os.makedirs(model_dir, exist_ok=True)                
            saved_model_path = os.path.join(model_dir, 'common_det_epoch_{}_train_{:.3f}'.format(epoch, best_iou))
            torch.save(model.state_dict(), saved_model_path)
            print('====> save model:\t{}'.format(saved_model_path))

        
    print('hello world!')
    


def test_RegressionDetectionTrainer():
    root = '/data/medical/brain/cerebral_parenchyma/exp/cta'
    boundary_info_file='/data/medical/brain/cerebral_parenchyma/exp/cta/config/mask_boundary_info.json'
    ds = PositionDetectionDS(root, boundary_info_file=boundary_info_file)
    dataloader = DataLoader(ds, batch_size=1)
    model = RegressionDetecter(1)

    criterion = None
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_epochs = 100
    best_iou = 0
    for epoch in range(n_epochs):
        LR_ADJUST_UTILS.adjust_learning_rate(optimizer, epoch, lr, True, n_epochs)
        ious = RegressionDetectionTrainer.train_one_epoch(dataloader, model.cuda(), criterion, optimizer, epoch, 1, phase='train', opt=None)
        if ious.mean() > best_iou:
            best_iou = ious.mean()
            print('current best train iou is:\t{}'.format(best_iou))
            model_dir = os.path.join(opts.checkpoints_dir, opts.name)



if __name__ == '__main__':
    # test_RegressionDetectionTrainer()
    train()