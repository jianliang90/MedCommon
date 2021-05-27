# CUDA_VISIBLE_DEVICES=1 python train.py \
# --dataroot /data/medical/brain/cerebral_parenchyma/exp/cta \
# --checkpoints_dir /data/medical/brain/cerebral_parenchyma/exp/cta/checkpoints/det \
# --name coronary_train \
# --boundary_info_file  /data/medical/brain/cerebral_parenchyma/exp/cta/config/mask_boundary_info.json \
# --pretrained /home/zhangwd/code/work/MedCommon_Self_Supervised_Learning/trainer/DetectionBrain/checkpoint_2390.pth.tar


CUDA_VISIBLE_DEVICES=0 python train.py \
--dataroot /data/medical/brain/cerebral_parenchyma/exp/cta_256 \
--checkpoints_dir /data/medical/brain/cerebral_parenchyma/exp/cta_256/checkpoints/det \
--name coronary_train \
--n_epochs 400 \
--batch_size 8 \
--boundary_info_file  /data/medical/brain/cerebral_parenchyma/exp/cta_256/config/mask_boundary_info.json \
--pretrained /home/zhangwd/code/work/MedCommon_Self_Supervised_Learning/trainer/DetectionBrain/checkpoint_2390.pth.tar

