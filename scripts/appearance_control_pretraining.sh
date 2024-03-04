CUDA_VISIBLE_DEVICES=0 torchrun --master_port 10000 train_tiktok.py \
--model_config model_lib/ControlNet/models/cldm_v15_reference_only.yaml \
--init_path ./pretrained_weights/control_sd15_ini.ckpt \
--output_dir ./tiktok_train_log/appearance_control_pretraining \
--train_batch_size 32 \
--num_workers 1 \
--img_bin_limit 15 \
--use_fp16 \
--control_mode controlnet_important \
--control_type body+hand+face \
--train_dataset tiktok_video_arnold \
--v4 \
--with_text \
--wonoise \
--finetune_control \
--local_image_dir ./tiktok_train_log/image_log/appearance_control_pretraining \
--local_log_dir ./tiktok_train_log/tb_log/appearance_control_pretraining \
$@

