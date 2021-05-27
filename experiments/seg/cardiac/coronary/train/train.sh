# exp 1
CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot /data/medical/cardiac/seg/coronary/coronary_cropped_by_mask \
--num_classes 2 \
--base_n_filter 6 \
--aug seg_train \
--lr 2e-4 \
--n_epochs 400 \
--crop_size 384 384 256 \
--checkpoints_dir /data/medical/cardiac/seg/coronary/coronary_cropped_by_mask/checkpoints \
--name coronary \
--weights /data/medical/cardiac/seg/coronary/coronary_cropped_by_mask/checkpoints/coronary/common_seg_epoch_28_train_0.069