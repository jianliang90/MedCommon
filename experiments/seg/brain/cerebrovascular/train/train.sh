# CUDA_VISIBLE_DEVICES=4 python train.py --dataroot /data/medical/brain/Cerebrovascular/segmenation_erode_k1 --num_classes 2 --base_n_filter 6 --lr 2e-4 --n_epochs 200 --crop_size 288 288 320 --checkpoints_dir /data/medical/brain/Cerebrovascular/segmenation_erode_k1/checkpoints --name cerebravascular --weights /data/medical/brain/Cerebrovascular/segmenation_erode_k1/checkpoints/cerebravascular/common_seg_epoch_25_train_0.084
# exp 1
CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot /data/medical/brain/Cerebrovascular/segmenation_erode_k1 \
--num_classes 2 \
--base_n_filter 6 \
--lr 2e-4 \
--n_epochs 400 \
--crop_size 288 288 320 \
--aug seg_train\
--checkpoints_dir \
/data/medical/brain/Cerebrovascular/segmenation_erode_k1/checkpoints \
--name cerebravascular \
--weights /data/medical/brain/Cerebrovascular/segmenation_erode_k1/checkpoints/cerebravascular/common_seg_epoch_187_train_0.044