# CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch \
# --master_addr='10.100.37.100' \
# --master_port='29503' \
# --nproc_per_node=2 \
# --nnodes=1 \
# --use_env \
# train_gan_3d.py \
# --dataroot /data/medical/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium \
# --model pix2pix_3d \
# --input_nc 1 \
# --output_nc 1 \
# --ngf 32 \
# --netG resnet_6blocks \
# --ndf 8 \
# --no_dropout \
# --netD pixel \
# --norm batch \
# --display_server='10.100.37.100' \
# --display_port=8098 \
# --display_id=1 \
# --lambda_L1=1 \
# --n_epochs=500 \
# --display_freq=10 \
# --print_freq=10 \
# --save_epoch_freq=10 \
# --lr_policy cosine \
# --lr 1e-4 \
# --checkpoints_dir /data/medical/cardiac/cta2mbf/data_114_20210318/checkpoints \
# --name cta2mbf \
# --crop_size 352 352 160 \
# --dst_vis_lut jet \
# --src_pattern cropped_cta.nii.gz \
# --dst_pattern cropped_mbf.nii.gz \
# --mask_pattern cropped_mbf_mask.nii.gz \
# --mask_label 1 \
# --lambda_L1_Mask 1.0 \
# --continue_train


# CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
# --master_addr='10.100.37.100' \
# --master_port='29505' \
# --nproc_per_node=2 \
# --nnodes=1 \
# --use_env \
# train_gan_3d.py \
# --dataroot /data/medical/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium \
# --model pix2pix_3d \
# --input_nc 1 \
# --output_nc 1 \
# --ngf 32 \
# --netG resnet_6blocks \
# --ndf 8 \
# --no_dropout \
# --netD pixel \
# --norm batch \
# --display_server='10.100.37.100' \
# --display_port=8900 \
# --display_id=1 \
# --lambda_L1=1 \
# --n_epochs=500 \
# --display_freq=10 \
# --print_freq=10 \
# --save_epoch_freq=10 \
# --lr_policy cosine \
# --lr 1e-4 \
# --checkpoints_dir /data/medical/cardiac/cta2mbf/data_114_20210318/checkpoints \
# --name cta2mbf_sr \
# --crop_size 384 384 160 \
# --dst_vis_lut jet \
# --src_pattern cropped_cta.nii.gz \
# --dst_pattern cropped_mbf.nii.gz \
# --mask_pattern cropped_mbf_mask.nii.gz \
# --mask_label 1 \
# --lambda_L1_Mask 1.0 \
# --no_discriminator \
# --ssl_sr \
# --ssl_arch resnet10 \
# --ssl_pretrained_file /data/medical/cardiac/cta2mbf/ssl/cropped_ori/checkpoints/mbf/resnet10/checkpoint_4520.pth.tar \
# --continue_train

CUDA_VISIBLE_DEVICES=1,3,6,7 python -m torch.distributed.launch \
--master_addr='10.100.37.100' \
--master_port='29505' \
--nproc_per_node=4 \
--nnodes=1 \
--use_env \
train_gan_3d.py \
--dataroot /data/medical/cardiac/cta2mbf/data_150_20210628/5.mbf_myocardium \
--model pix2pix_3d \
--input_nc 1 \
--output_nc 1 \
--ngf 32 \
--netG resnet_6blocks \
--ndf 8 \
--no_dropout \
--netD pixel \
--norm batch \
--display_server='10.100.37.100' \
--display_port=8900 \
--display_id=0 \
--lambda_L1=1 \
--n_epochs=500 \
--display_freq=10 \
--print_freq=10 \
--save_epoch_freq=10 \
--lr_policy cosine \
--lr 1e-4 \
--checkpoints_dir /data/medical/cardiac/cta2mbf/data_150_20210628/checkpoints \
--name cta2mbf_sr \
--crop_size 384 384 160 \
--dst_vis_lut jet \
--src_pattern cropped_cta.nii.gz \
--dst_pattern cropped_mbf.nii.gz \
--mask_pattern cropped_mbf_mask.nii.gz \
--mask_label 1 \
--lambda_L1_Mask 1.0 \
--no_discriminator \
--ssl_sr \
--ssl_arch resnet10 \
--ssl_pretrained_file /data/medical/cardiac/cta2mbf/ssl/cropped_ori/checkpoints/mbf/resnet10/checkpoint_9990.pth.tar \
--continue_train


# for inference
# CUDA_VISIBLE_DEVICES=6 python train_gan_3d.py \
# --dataroot /data/medical/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium \
# --model pix2pix_3d \
# --input_nc 1 \
# --output_nc 1 \
# --ngf 32 \
# --netG resnet_6blocks \
# --ndf 8 \
# --no_dropout \
# --netD pixel \
# --norm batch \
# --display_server='10.100.37.100' \
# --display_port=8098 \
# --display_id=1 \
# --lambda_L1=1 \
# --n_epochs=500 \
# --display_freq=10 \
# --print_freq=10 \
# --save_epoch_freq=10 \
# --lr_policy cosine \
# --lr 1e-4 \
# --checkpoints_dir /data/medical/cardiac/cta2mbf/data_114_20210318/checkpoints \
# --name cta2mbf \
# --crop_size 352 352 160 \
# --dst_vis_lut jet \
# --src_pattern cropped_cta.nii.gz \
# --dst_pattern cropped_mbf.nii.gz \
# --mask_pattern cropped_mask.nii.gz \
# --mask_label 6 \
# --lambda_L1_Mask 0.4 \
# --inference_mode train \
# --continue_train
