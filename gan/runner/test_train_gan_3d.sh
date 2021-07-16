# # mbf demo
# CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
# --master_addr='10.100.37.100' \
# --master_port='29501' \
# --nproc_per_node=1 \
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
# --display_port=8099 \
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
# --crop_size 64 64 64
# #--continue_train


# cta2dwi demo
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
--master_addr='10.100.37.100' \
--master_port='29501' \
--nproc_per_node=1 \
--nnodes=1 \
--use_env \
train_gan_3d.py \
--dataroot /data/medical/brain/gan/hospital_6_crop/experiment_registration2/8.common_gan \
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
--display_port=8099 \
--display_id=1 \
--lambda_L1=1 \
--n_epochs=5000 \
--display_freq=2 \
--print_freq=2 \
--save_epoch_freq=10 \
--lr_policy cosine \
--lr 1e-4 \
--checkpoints_dir /data/medical/brain/gan/hospital_6_crop/experiment_registration2/checkpoints \
--name cta2dwi \
--crop_size 416 416 128 \
--src_pattern CTA.nii.gz \
--dst_pattern BXXX.nii.gz \
--dst_ww_wl 400 200 \
# --continue_train

