# exp 1
CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot /data/medical/lung/LUNA/lung_256 \
--num_classes 4 \
--base_n_filter 6 \
--aug seg_train \
--lr 2e-4 \
--n_epochs 400 \
--crop_size 256 256 256 \
--dynamic_size 256 256 256 \
--weights /data/medical/lung/LUNA/lung_256/checkpoints/lung/common_seg_epoch_91_train_0.024 \
--checkpoints_dir /data/medical/lung/LUNA/lung_256/checkpoints \
--name lung