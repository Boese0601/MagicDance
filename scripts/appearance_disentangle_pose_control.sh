CUDA_VISIBLE_DEVICES=0 torchrun --master_port 10000 train_tiktok.py \
--model_config model_lib/ControlNet/models/cldm_v15_reference_only_pose.yaml \
--init_path ./pretrained_weights/control_sd15_ini.ckpt \
--output_dir ./tiktok_train_log/magicdance \
--train_batch_size 8 \
--num_workers 1 \
--control_mode controlnet_important \
--img_bin_limit 29 \
--use_fp16 \
--control_type body+hand+face \
--train_dataset tiktok_video_arnold \
--with_text \
--wonoise \
--finetune_control \
--local_image_dir ./tiktok_train_log/image_log/magicdance \
--local_log_dir ./tiktok_train_log/tb_log/magicdance \
--image_pretrain_dir ./tiktok_train_log/appearance_control_pretraining \
--pose_pretrain_dir ./pretrained_weights/control_v11p_sd15_openpose.pth \
$@

