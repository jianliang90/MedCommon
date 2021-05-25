import os
import sys
import torch
import torch.nn as nn

MEDCOMMON_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir)
sys.path.append(MEDCOMMON_ROOT)
sys.path.append(os.path.join(MEDCOMMON_ROOT, 'external_lib'))

from tqdm import tqdm
import time

from utils.misc_utils import AverageMeter
from detection.models.detection_auto_resample import RegressionDetecter
from detection.datasets.position_detection_common_ds import PositionDetectionDS
from utils.detection_utils import PYTORCH_TENSOR_DETECTION_UTILS
from utils.lr_adjust_utils import LR_ADJUST_UTILS

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
        for num_iter, (images, targets, names) in tqdm(enumerate(dataloader)):
            data_time.update(time.time() - end)
            output = model(images.cuda())
            if criterion is not None:
                loss = criterion(output, targets.cuda())
            else:
                iou,_ = PYTORCH_TENSOR_DETECTION_UTILS.calc_brick_iou(output, targets.view(targets.shape[0],-1).cuda())
                l1_loss = torch.nn.L1Loss()(output, targets.view(targets.shape[0],-1).cuda())
                loss = l1_loss + (1-iou.mean())
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


def test_RegressionDetectionTrainer():
    from detection.models.detection_auto_resample import RegressionDetecter
    from detection.datasets.position_detection_common_ds import PositionDetectionDS
    from utils.detection_utils import PYTORCH_TENSOR_DETECTION_UTILS
    from torch.utils.data import Dataset, DataLoader
    
    root = '/data/medical/brain/cerebral_parenchyma/exp/cta'
    ds = PositionDetectionDS(root)
    dataloader = DataLoader(ds, batch_size=1)
    model = RegressionDetecter(1)

    criterion = None
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_epochs = 100
    for epoch in range(n_epochs):
        LR_ADJUST_UTILS.adjust_learning_rate(optimizer, epoch, lr, True, n_epochs)
        RegressionDetectionTrainer.train_one_epoch(dataloader, model.cuda(), criterion, optimizer, epoch, 1, phase='train', opt=None)


if __name__ == '__main__':
    test_RegressionDetectionTrainer()