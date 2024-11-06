python source/train_CAE.py \
--config_file_path "config_CAE_denoising.yaml" \
--model_checkpoint "model_state_CAE_denoising.pth" \
--learning_rate 0.0002 \
--batch_size 600 \
--num_epochs 1000 \
--dataset_directory "D:/playgroundv25/downsample_images" \
--min_loss_path "min_loss_CAE.json"

#--batch_size 530 \