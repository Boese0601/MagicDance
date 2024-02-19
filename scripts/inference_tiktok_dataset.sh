CUDA_VISIBLE_DEVICES=1 torchrun --master_port 10000 test_tiktok.py \
--model_config model_lib/ControlNet/models/cldm_v15_reference_only_pose.yaml \
--num_train_steps 10 \
--img_bin_limit all \
--train_batch_size 1 \
--use_fp16 \
--control_mode controlnet_important \
--control_type body+hand+face \
--train_dataset tiktok_video_arnold \
--with_text \
--wonoise \
--local_image_dir ./tiktok_test_log/image_log/magicdance \
--local_log_dir ./tiktok_test_log/tb_log/magicdance \
--image_pretrain_dir ./pretrained_weights/model_state-110000.th
$@

