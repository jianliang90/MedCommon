""" segmentation of lung and airway in fine resolution.
"""

import os
import sys
import warnings
current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
seg_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir, os.path.pardir)
seg_zf_root_dir = os.path.join(seg_root_dir, 'segmentation', 'zf')
print(seg_root_dir)
print(seg_zf_root_dir)
# sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
sys.path.append(seg_root_dir)
sys.path.append(seg_zf_root_dir)

warnings.filterwarnings("ignore")

import torch

# from network.unet import UNet
from network.unet_preprocess_deploy import UNet
from runner.runner import SegmentationModel
from runner.args import ModelOptions
from data_processor.data_loader import DataSetLoader

import fire


# ---------------------------------------------Args config-------------------------------------------------- #
net_config = {"num_class": 3,
              "nb_filter": [8, 16, 32, 64, 128],
              "input_size":[128, 128, 128],
              "clip_window":[-1200, 600],
              "use_checkpoint": False}

def parse_args():
    args = ModelOptions("segmentation of lung").parse()
    # args.image_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/image_refine/crop_256_192_256/"
    # args.mask_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/mask_refine/crop_256_192_256/"

    args.image_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/image_refine/ori_128_128_128/"
    args.mask_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/mask_refine/ori_128_128_128/"

    args.train_dataset = "/fileser/zhangfan/DataSet/airway_segment_data/csv/train_filename.csv"
    args.val_dataset = "/fileser/zhangfan/DataSet/airway_segment_data/csv/val_filename.csv"
    args.label = ["left_lung", "right_lung", "airway"]
    args.num_classes = 3
    args.batch_size = 4
    args.n_workers = 4
    args.lr = 1e-3
    args.epochs = 150
    # args.mode = "train"
    args.out_dir = "./output/lung_fine_seg"

    return args


# --------------------------------------------Init--------------------------------------------------------- #

def get_data_loader(args):
    train_dataLoader = DataSetLoader(csv_path=args.train_dataset, image_dir=args.image_dir,
                                    mask_dir=args.mask_dir, num_classes=args.num_classes, phase="train",
                                    window_level=[-1200, 600])
    val_dataLoader = DataSetLoader(csv_path=args.val_dataset, image_dir=args.image_dir,
                                mask_dir=args.mask_dir, num_classes=args.num_classes, phase="val",
                                window_level=[-1200, 600])
    return train_dataLoader, val_dataLoader

def init_model(args, net_config):
    torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
    network = UNet(net_config)
    model = SegmentationModel(args, network)

    return network, model


def inference_onecase(network, args, infile, out_mask_file=None, is_series=False):
    from data_processor.data_io import DataIO
    from utils.mask_utils import smooth_mask
    from utils.image_utils import clip_and_normalize
    import numpy as np
    import torch.nn.functional as F

    data_io = DataIO()

    model = SegmentationModel(args, network).network
    # model = model.cuda()

    model.train()

    if is_series:
        data_dict = data_io.load_dicom_series(infile)
    else:    
        data_dict = data_io.load_nii_image(infile)
    image_arr = data_dict['image']
    window_level = net_config['clip_window']
    image_zyx = clip_and_normalize(image_arr, min_window=window_level[0], max_window=window_level[1])
    print(image_zyx.max())
    print(image_zyx.min())
    image_arr = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).float().cuda()

    with torch.no_grad():
        output = model(image_arr)
    if args.activation == "softmax":
        output = F.softmax(output[0], dim=1)
    elif args.activation == "sigmoid":
        output = F.sigmoid(output[0])
    output = output.detach().cpu().numpy().squeeze()

    pred_mask_czyx = output
    pred_mask_czyx[pred_mask_czyx >= 0.5] = 1
    pred_mask_czyx[pred_mask_czyx < 0.5] = 0

    pred_mask_czyx = pred_mask_czyx[np.newaxis, ] if args.num_classes == 1 else pred_mask_czyx

    pred_mask_zyx = np.zeros(pred_mask_czyx.shape[1:], dtype=np.int8)

    for i in range(args.num_classes):
        out_mask = pred_mask_czyx[i, ].squeeze()
        if args.is_post_process:
            out_mask = smooth_mask(out_mask, area_least=2000, is_binary_close=False)
        pred_mask_zyx[out_mask != 0] = i + 1

    ranges = np.where(pred_mask_zyx > 0)
    min_z, min_y, min_x = np.min(np.array(ranges), axis=1)
    max_z, max_y, max_x = np.max(np.array(ranges), axis=1)
    print('lung dim:\tdepth:[{}]\theight:[{}]\twidth:[{}]'.format(max_z-min_z, max_y-min_y, max_x-min_x))

    if out_mask_file is not None:
        os.makedirs(os.path.dirname(out_mask_file), exist_ok=True)
        data_io.save_medical_info_and_data(pred_mask_zyx, data_dict['origin'], data_dict["spacing"],
                                                     data_dict["direction"], out_mask_file)


# --------------------------------------------Session------------------------------------------------------ #

def train():
    args = parse_args()
    network, model = init_model(args, net_config)
    train_dataLoader, val_dataLoader = get_data_loader(args)
    if args.mode == "train":
        print('train mode')
        model.train(train_dataLoader, val_dataLoader)
    elif args.mode == "val":
        print('validation mode')
        model.validate(val_dataLoader)
    elif args.mode == 'inference':
        # inference_onecase(network, args, '/data/medical/lung/LUNA/RAW_NII/1.3.6.1.4.1.14519.5.2.1.6279.6001.245546033414728092794968890929.nii.gz', '/data/medical/lung/changzheng/airway/airway_20201030/coarse_lung_masks/test.nii.gz')
        inference_onecase(network, args, '/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/image_refine/ori_128_128_128/1.3.6.1.4.1.14519.5.2.1.6279.6001.148935306123327835217659769212.nii.gz', '/data/medical/lung/changzheng/airway/airway_20201030/coarse_lung_masks/test.nii.gz')


if __name__ == "__main__":
    train()
    # fire.Fire()



    
