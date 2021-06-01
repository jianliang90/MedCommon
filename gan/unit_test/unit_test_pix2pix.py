import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from options.train_options import TrainOptions
from util.visualizer import Visualizer
import models
from models.pix2pix_3d_model import Pix2Pix3DModel

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from utils.distributed_utils import DistributedUtils


from datasets.common_ds import GAN_COMMON_DS, get_common_transform



opt = TrainOptions().parse()

DistributedUtils.init_distributed_mode(opt)

if 'rank' not in opt:
    opt.rank = DistributedUtils.get_rank()

print(opt)


data_root = '/data/medical/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium'
transform = get_common_transform([352,352,160],'GAN')
ds = GAN_COMMON_DS(data_root, 'cropped_cta.nii.gz', 'cropped_mbf.nii.gz', [64,64,64], transform)
dataloader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=True, pin_memory=True)
dataset_size = len(dataloader)    # get the number of images in the dataset.

model = models.create_model(opt)
model.setup(opt)

visualizer = Visualizer(opt)
total_iters = 0                # the total number of training iterations

print(model.netD)

# model = Pix2Pix3DModel(opt)

for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    # 这一句重要到你无法想象
    model.eval()
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

    for index, (subjects) in enumerate(dataloader):
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size * DistributedUtils.get_world_size()
        epoch_iter += opt.batch_size * DistributedUtils.get_world_size()

        real_a = subjects['src']['data'].float()
        real_b = subjects['dst']['data'].float()
        input = {}
        input['A'] = real_a
        input['B'] = real_b
        input['A_paths'] = 'A'
        input['B_paths'] = 'B'
        model.set_input(input)
        model.optimize_parameters()
        
        if DistributedUtils.get_rank() == 0:
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()

                import matplotlib.pyplot as plt
                vis_result = model.get_current_visuals()
                for key in vis_result.keys():
                    if 'real_A' in key:
                        ww = 400
                        wl = 40
                        w_min = wl-ww/2
                        w_max = wl+ww/2
                        vis_result[key] = torch.clamp(vis_result[key], min=w_min, max=w_max)/ww
                    else:
                        ww = 150
                        wl = 75
                        w_min = wl-ww/2
                        w_max = wl+ww/2
                        vis_result[key] = torch.clamp(vis_result[key], min=w_min, max=w_max)/ww
                        vis_result[key] = plt.get_cmap('jet')(vis_result[key].detach().cpu().numpy()).squeeze()[...,:3]*255
                visualizer.display_current_results(vis_result, epoch_iter, False)


            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / (opt.batch_size)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)            
            
            iter_data_time = time.time()


    if DistributedUtils.get_rank() == 0:
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    model.update_learning_rate()                     # update learning rates at the end of every epoch.

print('hello world!')
