imagename="181020"
posepath="001"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 18102 test_any_image_pose.py \
--model_config model_lib/ControlNet/models/cldm_v15_reference_only_pose.yaml \
--num_train_steps 1 \
--img_bin_limit all \
--train_batch_size 1 \
--use_fp16 \
--control_mode controlnet_important \
--control_type body+hand+face \
--train_dataset tiktok_video_arnold \
--v4 \
--with_text \
--wonoise \
--local_image_dir ./tiktok_test_log/image_log/$imagename/$posepath/image \
--local_log_dir ./tiktok_test_log/tb_log/$imagename/$posepath/log \
--image_pretrain_dir ./pretrained_weights/model_state-110000.th \
--local_pose_path ./example_data/pose_sequence/$posepath \
--local_cond_image_path ./example_data/image/out-of-domain/$imagename.png \
$@

