CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot /data/medical/cardiac/seg/coronary/coronary_ori \
--checkpoints_dir /data/medical/cardiac/seg/coronary/coronary_ori/checkpoints/det \
--name coronary \
--boundary_info_file  /data/medical/cardiac/seg/coronary/coronary_ori/config/mask_boundary_info.json \
--weights /data/medical/cardiac/seg/coronary/coronary_ori/checkpoints/det/coronary/common_det_epoch_4_train_0.661