import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from options.train_options import TrainOptions
from options.test_options import TestOptions
from util.visualizer import Visualizer
import models
from models.pix2pix_3d_model import Pix2Pix3DModel

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
from common.common_base_datasets import CommonSegmentationDS
from utils.distributed_utils import DistributedUtils
from utils.image_show_utils import ImageShowUtils
from utils.metrics_utils import MetricsUtils

from datasets.common_ds import GAN_COMMON_DS, get_common_transform

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd

class GANTrainer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def train_one_epoch(model, dataloader, visualizer, total_iters, epoch, opt):
        # 这一句重要到你无法想象
        model.eval()
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset_size = len(dataloader) 
        for index, (subjects) in enumerate(dataloader):
            iter_start_time = time.time()  # timer for computation per iteration
            # if total_iters % opt.print_freq == 0:
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
            if opt.mask_pattern:
                mask = subjects['mask']['data']
                input['mask'] = mask
                input['mask_label'] = opt.mask_label
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
                            ww = opt.src_ww_wl[0]
                            wl = opt.src_ww_wl[1]
                            w_min = wl-ww/2
                            w_max = wl+ww/2
                            vis_result[key] = torch.clamp(vis_result[key], min=w_min, max=w_max)/ww
                        else:
                            ww = opt.dst_ww_wl[0]
                            wl = opt.dst_ww_wl[1]
                            w_min = wl-ww/2
                            w_max = wl+ww/2
                            vis_result[key] = torch.clamp(vis_result[key], min=w_min, max=w_max)/ww
                            if opt.dst_vis_lut:
                                vis_result[key] = plt.get_cmap(opt.dst_vis_lut)(vis_result[key].detach().cpu().numpy()).squeeze()[...,:3]*255
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
        model.update_learning_rate()  
        
        return total_iters

    
    @staticmethod
    def load_model(opts, weights_G):
        model = models.create_model(opts)
        # setup操作会加载模型，这里不用
        # model.setup(opts)
        netG = model.netG.module
        netG.load_state_dict(torch.load(weights_G, map_location='cpu'))
        return netG
        # return model

    @staticmethod
    def inference_onecase(model, series_path, crop_size = [128, 128, 128], out_root=None, opts=None):
        model.eval()
        subject = CommonSegmentationDS.get_inference_input(series_path, crop_size)
        input = subject['src']['data'].float().unsqueeze(0)
        real_a = input
        fake_b = model.netG(real_a.cuda())
        fake_b = fake_b.detach().squeeze().cpu().numpy()
        fake_b = np.transpose(fake_b, [2,1,0])
        info_img = sitk.ReadImage(series_path)
        spacing = info_img.GetSpacing()
        direction = info_img.GetDirection()
        origin = info_img.GetOrigin()
        os.makedirs(out_root, exist_ok=True)
        real_img_a = sitk.GetImageFromArray(real_a)
        real_img_a.SetSpacing(spacing)
        real_img_a.SetDirection(direction)
        real_img_a.SetOrigin(origin)
        fake_img_b = sitk.GetImageFromArray(fake_b)
        fake_img_b.CopyInformation(real_img_a)
        if out_root:
            sitk.WriteImage(real_img_a, os.path.join(out_root, 'real_a.nii.gz'))
            sitk.WriteImage(fake_img_b, os.path.join(out_root, 'fake_b.nii.gz'))
        return real_img_a, fake_img_b

    @staticmethod
    def inference_batch(model, data_root, out_dir, opts):
        '''
        必须设置：
            crop_size
            src_pattern
            dst_pattern
        '''
        if opts.inference_mode == 'train':
            model.train()
        else:
            model.eval()
        transform = get_common_transform(opts.crop_size,'GAN_INFERENCE')
        ds = GAN_COMMON_DS(data_root, opts.src_pattern, opts.dst_pattern, opts.crop_size, transform)
        dataloader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=True, pin_memory=True)
        dataset_size = len(dataloader) 
        for index, (subjects) in tqdm(enumerate(dataloader)):
            real_a = subjects['src']['data'].float()
            real_b = subjects['dst']['data'].float()
            # input = {}
            # input['A'] = real_a
            # input['B'] = real_b
            # input['A_paths'] = 'A'
            # input['B_paths'] = 'B'
            # model.set_input(input)
            # fake_b = model.netG(real_a.cuda())
            fake_b = model(real_a.cuda())
            fake_b = fake_b.detach().squeeze().cpu().numpy()
            real_a = real_a.squeeze().cpu().numpy()
            real_b = real_b.squeeze().cpu().numpy()
            real_a = np.transpose(real_a, [2,1,0])
            real_b = np.transpose(real_b, [2,1,0])
            fake_b = np.transpose(fake_b, [2,1,0])
            pid = subjects['src']['path'][0].split('/')[-2]
            info_img = sitk.ReadImage(subjects['src']['path'][0])
            spacing = info_img.GetSpacing()
            direction = info_img.GetDirection()
            origin = info_img.GetOrigin()
            out_sub_dir = os.path.join(out_dir, pid)
            os.makedirs(out_sub_dir, exist_ok=True)
            real_img_a = sitk.GetImageFromArray(real_a)
            real_img_a.SetSpacing(spacing)
            real_img_a.SetDirection(direction)
            real_img_a.SetOrigin(origin)
            real_img_b = sitk.GetImageFromArray(real_b)
            real_img_b.CopyInformation(real_img_a)
            fake_img_b = sitk.GetImageFromArray(fake_b)
            fake_img_b.CopyInformation(real_img_a)
            sitk.WriteImage(real_img_a, os.path.join(out_sub_dir, 'real_a.nii.gz'))
            sitk.WriteImage(real_img_b, os.path.join(out_sub_dir, 'real_b.nii.gz'))
            sitk.WriteImage(fake_img_b, os.path.join(out_sub_dir, 'fake_b.nii.gz'))        

    @staticmethod
    def export_slicemap_onecase(data_root, out_root, 
            src_ww=150, src_wl=75, dst_ww=150, dst_wl=75, 
            src_lut=None, dst_lut='jet'
        ):
        real_a_file = os.path.join(data_root, 'real_a.nii.gz')
        real_b_file = os.path.join(data_root, 'real_b.nii.gz')
        fake_b_file = os.path.join(data_root, 'fake_b.nii.gz')

        real_a_img = sitk.ReadImage(real_a_file)
        real_b_img = sitk.ReadImage(real_b_file)
        fake_b_img = sitk.ReadImage(fake_b_file)

        real_a_arr = sitk.GetArrayFromImage(real_a_img)
        real_b_arr = sitk.GetArrayFromImage(real_b_img)
        fake_b_arr = sitk.GetArrayFromImage(fake_b_img)

        ImageShowUtils.save_volume_to_jpg(real_a_arr, os.path.join(out_root, 'real_a'), src_ww, src_wl, axis=0, file_prefix='x', reverse=False, lut_name=src_lut)
        ImageShowUtils.save_volume_to_jpg(real_b_arr, os.path.join(out_root, 'real_b'), dst_ww, dst_wl, axis=0, file_prefix='x', reverse=False, lut_name=dst_lut)
        ImageShowUtils.save_volume_to_jpg(fake_b_arr, os.path.join(out_root, 'fake_b'), dst_ww, dst_wl, axis=0, file_prefix='x', reverse=False, lut_name=dst_lut)

    @staticmethod
    def export_slicemap_singletask(data_root, out_root, suids,
            src_ww=150, src_wl=75, dst_ww=150, dst_wl=75, 
            src_lut=None, dst_lut='jet'
        ):
        for suid in tqdm(suids):
            try:
                sub_data_root = os.path.join(data_root, suid)
                sub_out_root = os.path.join(out_root, suid)
                GANTrainer.export_slicemap_onecase(sub_data_root, sub_out_root, 
                    src_ww, src_wl, dst_ww, dst_wl, src_lut, dst_lut)
            except Exception as e:
                print('====> Error case:\t{}'.format(suid))
                print(e)

    @staticmethod
    def export_slicemap_multiprocessing(data_root, out_root, 
            src_ww=150, src_wl=75, dst_ww=150, dst_wl=75, 
            src_lut=None, dst_lut='jet',
            process_num=6
        ):
        series_uids = []
        series_uids = os.listdir(data_root)

        # print(series_uids)
        num_per_process = (len(series_uids) + process_num - 1)//process_num

        # this for single thread to debug
        # GANTrainer.export_slicemap_singletask(data_root, out_root, series_uids, src_ww, src_wl, dst_ww, dst_wl, src_lut, dst_lut)

        # this for run 
        import multiprocessing
        from multiprocessing import Process
        multiprocessing.freeze_support()

        pool = multiprocessing.Pool()

        results = []

        print(len(series_uids))
        for i in range(process_num):
            sub_series_uids = series_uids[num_per_process*i:min(num_per_process*(i+1), len(series_uids))]
            print(len(sub_series_uids))
            result = pool.apply_async(GANTrainer.export_slicemap_singletask, 
                args=(data_root, out_root, sub_series_uids, src_ww, src_wl, dst_ww, dst_wl, src_lut, dst_lut))
            results.append(result)

        pool.close()
        pool.join()


    @staticmethod
    def calc_mae(
            data_root='/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference_384x384x160_eval', 
            out_dir = '/data/medical/cardiac/cta2mbf/data_66_20210517/7.analysis_result', 
            out_file = 'mae_384x384x160_eval.csv'
        ):
        row_elems = []
        for suid in tqdm(os.listdir(data_root)):
            sub_data_root = os.path.join(data_root, suid)
            real_b_file = os.path.join(sub_data_root, 'real_b.nii.gz')
            fake_b_file = os.path.join(sub_data_root, 'fake_b.nii.gz') 
            _, mae = MetricsUtils.calc_mae_with_file(real_b_file, fake_b_file)
            row_elems.append(np.array([suid, mae]))
        df = pd.DataFrame(np.array(row_elems), columns=['suid', 'mae'])
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, out_file)
        df.to_csv(out_file)

    

def train():
    opt = TrainOptions().parse()
    DistributedUtils.init_distributed_mode(opt)
    if 'rank' not in opt:
        opt.rank = DistributedUtils.get_rank()
    print(opt)

    data_root = opt.dataroot
    crop_size = opt.crop_size
    transform = get_common_transform(crop_size, opt.aug)
    ds = GAN_COMMON_DS(data_root, opt.src_pattern, opt.dst_pattern, crop_size, transform, opt.mask_pattern)
    dataloader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=True, pin_memory=True)
    dataset_size = len(dataloader)    # get the number of images in the dataset.
    
    model = models.create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0                # the total number of training iterations
    print(model.netD)    

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        total_iters = GANTrainer.train_one_epoch(model, dataloader, visualizer, total_iters, epoch, opt)

def test_load_model():
    '''
        # --model pix2pix_3d \
        # --input_nc 1 \
        # --output_nc 1 \
        # --ngf 32 \
        # --netG resnet_6blocks \
        # --ndf 8 \
        # --no_dropout \
        # --netD pixel \
        # --norm batch \
    '''
    opts = TestOptions().parse()
    opts.model = 'pix2pix_3d'
    opts.input_nc = 1
    opts.output_nc = 1
    opts.ngf = 32
    opts.netG = 'resnet_6blocks'
    opts.ndf = 8
    opts.no_dropout = True
    opts.netD = 'pixel'
    opts.norm = 'batch'

    weights_G = '/home/zhangwd/code/work/MedCommon/gan/unit_test/checkpoints/experiment_name/430_net_G.pth'
    model = GANTrainer.load_model(opts, weights_G)
    print(model)

def inference_onecase(series_path, out_root, weights):
    opt = TrainOptions().parse()
    model = GANTrainer.load_model(opt, weights)
    GANTrainer.inference_onecase(model.cuda(), series_path, opt.crop_size, out_root, opt)

def inference(data_root, out_root, weights):
    opt = TrainOptions().parse()
    model = GANTrainer.load_model(opt, weights)
    GANTrainer.inference_batch(model.cuda(), data_root, out_root, opt)

if __name__ == '__main__':
    train()
    # test_load_model()
    # inference(
    #         '/data/medical/cardiac/cta2mbf/data_140_20210602/5.mbf_myocardium', 
    #         '/data/medical/cardiac/cta2mbf/data_140_20210602/6.inference_352x352x160_eval', 
    #         '/data/medical/cardiac/cta2mbf/data_114_20210318/checkpoints/cta2mbf/90_net_G.pth'
    #     )
    # inference(
    #         '/data/medical/cardiac/cta2mbf/data_140_20210602/5.mbf_myocardium', 
    #         '/data/medical/cardiac/cta2mbf/data_140_20210602/6.inference_352x352x160_train', 
    #         '/home/zhangwd/code/work/MedCommon/gan/unit_test/checkpoints/bk/train_latest/1140_net_G.pth'
    #     )
